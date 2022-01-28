from glob import glob
import os.path
from pprint import pprint
from typing import Iterable, List
import albumentations as A
import cv2
import numpy as np
import scipy
from sklearn import datasets
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch import nn as nn
from torchvision import transforms
import itertools
from architectures import fornet
from architectures.fornet import FeatureExtractor
from blazeface import FaceExtractor, BlazeFace, VideoReader
import pandas as pd
from tqdm import tqdm

formats = ['image','video']
models = ['TimmV2','TimmV2ST','ViT','ViTST']
datasets = ['ffpp','dfdc','celeb']


def get_transformer(face_policy: str, patch_size: int, net_normalizer: transforms.Normalize, train: bool):
    # Transformers and traindb
    if face_policy == 'scale':
        # The loader crops the face isotropically then scales to a square of size patch_size_load
        loading_transformations = [
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0,always_apply=True),
            A.Resize(height=patch_size,width=patch_size,always_apply=True),
        ]
        if train:
            downsample_train_transformations = [
                A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
            ]
        else:
            downsample_train_transformations = []
    elif face_policy == 'tight':
        # The loader crops the face tightly without any scaling
        loading_transformations = [
            A.LongestMaxSize(max_size=patch_size, always_apply=True),
            A.PadIfNeeded(min_height=patch_size, min_width=patch_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0,always_apply=True),
        ]
        if train:
            downsample_train_transformations = [
                A.Downscale(scale_max=0.5, scale_min=0.5, p=0.5),  # replaces scaled dataset
            ]
        else:
            downsample_train_transformations = []
    else:
        raise ValueError('Unknown value for face_policy: {}'.format(face_policy))

    if train:
        aug_transformations = [
            A.Compose([
                A.HorizontalFlip(),
                A.OneOf([
                    A.RandomBrightnessContrast(),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
                ]),
                A.OneOf([
                    A.ISONoise(),
                    A.IAAAdditiveGaussianNoise(scale=(0.01 * 255, 0.03 * 255)),
                ]),
                A.Downscale(scale_min=0.7, scale_max=0.9, interpolation=cv2.INTER_LINEAR),
                A.ImageCompression(quality_lower=50, quality_upper=99),
            ], )
        ]
    else:
        aug_transformations = []

    # Common final transformations
    final_transformations = [
        A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std, ),
        ToTensorV2(),
    ]
    transf = A.Compose(
        loading_transformations + downsample_train_transformations + aug_transformations + final_transformations)
    return transf

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    
def get_video_paths(data_dir,num_videos):
    video_paths = glob(data_dir + "/**/*.mp4",recursive=True)
    video_idxs = [x for x in range(0,num_videos)]  # if num_videos is 3, then video_idxs = [0,1,2] i.e we will test videos at index 0,1,2 in file_names
    file_names = []
    for i in video_paths:
        file_names.append(os.path.basename(i))
    file_names.sort()
    return file_names, video_idxs

def get_model_paths(model,model_dir,dataset,choices):
    model_paths = glob(  model_dir + '/**/*.pth', recursive=True)
    models_for_dataset = []
    for i in model_paths:
        if(i.split("/")[-1].startswith(dataset)):
            models_for_dataset.append(i)
    model_paths = models_for_dataset
    a = []
    for i in model_paths:
        if( choices[os.path.basename(i).split(".")[0].split("_")[1]] in model ):
            a.append(i)
    model_paths = a
    return model_paths    



def load_weights(model_paths,choices,device):
    model_list = []
    for i in tqdm(model_paths, desc="Loading Models"):
        net_name = choices[i.split("/")[-1].split("_")[1].split(".")[0]]
        net_class = getattr(fornet, net_name)
        net: FeatureExtractor = net_class().eval().to(device)
        net.load_state_dict(torch.load(
            i, map_location='cpu')['net'])
        model_list.append(net)
    return model_list



def load_face_extractor(blazeface_dir,device,fpv):
    
    facedet = BlazeFace().to(device)
    facedet.load_weights(blazeface_dir+"blazeface.pth")
    facedet.load_anchors(blazeface_dir+"anchors.npy")
    
    videoreader = VideoReader(verbose=False)
    def video_read_fn(x): return videoreader.read_frames(
    x, num_frames=fpv)
    
    return FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)
    

def extract_faces(data_dir,file_names,video_idxs,transformer,face_extractor,num_videos,fpv):
    faces = face_extractor.process_videos(input_dir=data_dir, filenames=file_names, video_idxs=video_idxs)

    faces_frames = [fpv*x for x in range(0, num_videos+1)]   # [0,32,64,96]

    faces_hc = torch.stack([transformer(image=frame['faces'][0])['image'] for frame in faces if len(frame['faces'])])
    
    return faces_hc, faces_frames
    
    
def predict(ensemble_models,data_dir,file_names,video_idxs,num_videos,faces,faces_frames,model,save_csv=True,true_class=False):
    predictions = {}
    with torch.no_grad():
        for i in tqdm(range(0, num_videos),desc='Predicting: '):  # (0,3) i.e 0,1,2
            score = ensemble_models(faces[faces_frames[i]:faces_frames[i+1]])
            if(not true_class):
                predictions[data_dir+file_names[video_idxs[i]]] = [score, {'ensemble_score': sum(score.values())/(len(model))  }, {
                    'predicted_class': 'real' if sum(score.values())/(len(model)) < 0.3 else 'fake'}]
            else:
                predictions[data_dir+file_names[video_idxs[i]]] = [score, {'ensemble_score': sum(score.values())/(len(model))  }, {
                    'predicted_class': 'real' if sum(score.values())/(len(model)) < 0.3 else 'fake', 'true_class': input_dir.split("/")[3]}]
    if(save_csv):
        pclass = []
        for preds in predictions:
            predicted_class = predictions[preds][2]['predicted_class']
            pclass.append(predicted_class)
        data = {'video_path': [x for x in predictions.keys()],
               'prediction':  pclass   } 
        df = pd.DataFrame(data)
        df.to_csv('predictions.csv')
        print("Predictions saved to predictions.csv")
    return predictions
    