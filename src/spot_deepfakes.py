from glob import glob
from json.tool import main
from statistics import mode
import cv2
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
import distutils
from distutils import util
import os
import os.path
sys.path.append('..')


def main():
    # Args
    parser = argparse.ArgumentParser()

    parser.add_argument('--media_type', type=str,
                        help='Image or Video', choices=utils.formats, default ='video')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the directory where image/video is stored. Eg: ../sample_videos/ffpp/real/', required=True)
    parser.add_argument('--dataset', type=str, help='Image/Video is from which dataset',
                        choices=utils.datasets, required=True)
    parser.add_argument('--model', nargs='+',
                        help="Mention models to use among ['TimmV2','TimmV2ST','ViT','ViTST']. Use this arg multiple times to add multiple models", choices=utils.models, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--blazeface_dir',type=str,default='blazeface/')
    parser.add_argument('--face_policy', default='scale')
    parser.add_argument('--face_size', type=int, default=224)
    parser.add_argument('--fpv', type=int,
                        help='frames per video to extract', default=32)
    parser.add_argument('--device', type=int, help='GPU device', default=0)
    parser.add_argument('--n',help='Number of videos to test',type=int,default=2)
    parser.add_argument('--annotate',type= str, help='Set True to save the video in output/ directory which is annotated with frame level predictions',choices=['True','False'], default='False')
    parser.add_argument('--output_dir',type=str,help='Path to directory where the annotate video is saved', default='output/')
    parser.add_argument('--video_id',nargs='+',type=int,help = "Index of the videos in data_dir that have to be checked")
    

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
    annotate = bool(distutils.util.strtobool(args.annotate))
    output_dir = args.output_dir
    video_idxs = args.video_id
    
    info = {'Type of Media: ': media_type, 'Path to media: ': data_dir, 'Dataset chosen: ': dataset, 'Models: ': model, 'Path to the directory of the models: ': model_dir, 'Face policy: ': face_policy, 'Face size: ': face_size, 'Frames per video: ': fpv, 'Device: ': device,'Annotate: ': annotate}
    print("Input information:")
    pprint.pprint(info)

    
    file_names, video_glob = utils.get_video_paths(data_dir, num_videos)    

    model_choices = {'v2': 'TimmV2', 'v2st': 'TimmV2ST', 'vit': 'ViT', 'vitst': 'ViTST'}
    
    model_paths = utils.get_model_paths(model, model_dir, dataset, choices=model_choices)

    
    models_loaded = utils.load_weights(model_paths,model_choices,device)
    pprint.pprint({"Models Loaded": model})
    
    ensemble_models = ensemble.ensemble(models_loaded,device)

    
    transformer = utils.get_transformer(face_policy, face_size, models_loaded[0].get_normalizer(), train=False)

    if(annotate):
        predictions = utils.extract_predict_annotate(output_dir,ensemble_models, video_glob, video_idxs, transformer, blazeface_dir,device, models_loaded)
        #print("Predictions:")
        #pprint.pprint(predictions)
    else:
        face_extractor = utils.load_face_extractor(blazeface_dir, device,fpv)
        print('Face extractor Loaded!')

        faces, faces_frames = utils.extract_faces(data_dir, file_names, video_idxs ,transformer,face_extractor,num_videos,fpv)
        print("Faces extracted and transformed!")

        predictions, predictions_frames =  utils.predict(ensemble_models,data_dir,file_names,video_idxs,num_videos,faces,faces_frames,model,save_csv=True,true_class=False)
        #print("Predictions:")
        #pprint.pprint(predictions)
        
        

if __name__ == '__main__':
    main()
