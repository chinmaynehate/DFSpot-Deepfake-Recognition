from glob import glob
from json.tool import main
from statistics import mode
from cv2 import VIDEOWRITER_PROP_FRAMEBYTES
from utils import utils,ensemble
from architectures.fornet import FeatureExtractor
from architectures import fornet
from blazeface import FaceExtractor, BlazeFace, VideoReader
import sys
from scipy.special import expit
import matplotlib.pyplot as plt
import torch
import argparse
from pathlib import Path
import pprint
import os.path
sys.path.append('..')


def main():
    # Args
    parser = argparse.ArgumentParser()

    parser.add_argument('--media_type', type=str,
                        help='Image or Video', choices=utils.formats, required=True)
    parser.add_argument('--data_dir', type=str,
                        help='Path to the directory where image/video is stored. Eg: ../sample_videos/ffpp/real/', required=True)
    parser.add_argument('--dataset', type=str, help='Image/Video is from which dataset',
                        choices=utils.datasets, required=True)
    parser.add_argument('--model', action='append',
                        help="Mention models to use among ['TimmV2','TimmV2ST','ViT','ViTST']. Use this arg multiple times to add multiple models", choices=utils.models, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--blazeface_dir',type=str,default='blazeface/')
    parser.add_argument('--face_policy', default='scale')
    parser.add_argument('--face_size', type=int, default=224)
    parser.add_argument('--fpv', type=int,
                        help='frames per video to extract', default=32)
    parser.add_argument('--device', type=int, help='GPU device', default=0)
    parser.add_argument('--n',help='Number of videos to test',type=int,default=2)

    args = parser.parse_args()

    media_type = args.media_type
    data_dir = args.data_dir
    dataset = args.dataset
    model = args.model
    model_dir = args.model_dir
    blazeface_dir = args.blazeface_dir
    face_policy = args.face_policy
    face_size = args.face_size
    fpv = args.fpv
    device = torch.device('cuda:{:d}'.format(
        args.device)) if torch.cuda.is_available() else torch.device('cpu')
    num_videos = args.n


    info = {'Type of Media: ': media_type, 'Path to media: ': data_dir, 'Dataset chosen: ': dataset, 'Models: ': model, 'Path to the directory of the models: ': model_dir, 'Face policy: ': face_policy, 'Face size: ': face_size, 'Frames per video: ': fpv, 'Device: ': device}
    print("Input information:")
    pprint.pprint(info)

    model_choices = {'v2': 'TimmV2', 'v2st': 'TimmV2ST', 'vit': 'ViT', 'vitst': 'ViTST'}
    
    
    file_names, video_idxs = utils.get_video_paths(data_dir, num_videos)
    print("File names: ", file_names)
    print("Video ids: ", video_idxs)
    
    
    face_extractor = utils.load_face_extractor(blazeface_dir, device,fpv)
    print('Face extractor Loaded!')
    
    model_paths = utils.get_model_paths(model, model_dir, dataset, choices=model_choices)
    print("Model paths: ", model_paths)
    
    models_loaded = utils.load_weights(model_paths,model_choices,device)
    print("Models Loaded!")
    
    transformer = utils.get_transformer(
    face_policy, face_size, models_loaded[0].get_normalizer(), train=False)
    print('Transformer Loaded!')
    
    faces, faces_frames = utils.extract_faces(data_dir, file_names, video_idxs ,transformer,face_extractor,num_videos,fpv)
    print("Faces extracted and transformed!")
    
    ensemble_models = ensemble.ensemble(models_loaded,device)
    print("Ensemble Ready!")
    
    predictions =  utils.predict(ensemble_models,data_dir,file_names,video_idxs,num_videos,faces,faces_frames,model,save_csv=True,true_class=False)
    print("Predictions:")
    pprint.pprint(predictions)
    



if __name__ == '__main__':
    main()


#  python3 spot.py --media_type video --data_dir ../sample_videos/ffpp/real --dataset ffpp --model TimmV2 --model TimmV2ST --model ViT --model ViTST --model_dir ../models --n 2