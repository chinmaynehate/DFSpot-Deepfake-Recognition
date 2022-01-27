import glob
from utils import utils
from architectures.fornet import FeatureExtractor
from architectures import fornet
from blazeface import FaceExtractor, BlazeFace, VideoReader
import sys
from scipy.special import expit
import matplotlib.pyplot as plt
import torch
sys.path.append('..')

net_choices = ['TimmV2', 'TimmV2ST', 'ViT', 'ViTST']

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32

dataset = "ffpp"
net_name = net_choices[0]
net_class = getattr(fornet, net_name)
model_path = "../models/" + dataset + "_" + "v2.pth"



net: FeatureExtractor = net_class().eval().to(device)
net.load_state_dict(torch.load(model_path, map_location='cpu')['net'])


transf = utils.get_transformer(
    face_policy, face_size, net.get_normalizer(), train=False)
facedet = BlazeFace().to(device)
facedet.load_weights("blazeface/blazeface.pth")
facedet.load_anchors("blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)


def video_read_fn(x): return videoreader.read_frames(
    x, num_frames=frames_per_video)


face_extractor = FaceExtractor(video_read_fn=video_read_fn, facedet=facedet)



video_paths = glob.glob('../sample_videos/ffpp/real/**/*.mp4', recursive=True)
file_names = []
for i in video_paths:
    file_names.append(i.split("/")[4])
file_names.sort()


video_idxs = [1, 3]

input_dir = '../sample_videos/ffpp/real/'


faces = face_extractor.process_videos(
    input_dir=input_dir, filenames=file_names, video_idxs=video_idxs)
total_videos = len(video_idxs)


faces_frames = [frames_per_video *
                x for x in range(0, total_videos+1)]   # [0,32,64,96]

faces_hc = torch.stack([transf(image=frame['faces'][0])['image']
                       for frame in faces if len(frame['faces'])])


predictions = {}
with torch.no_grad():
    for i in range(0, total_videos):  # (0,3) i.e 0,1,2
        pred = net(faces_hc[faces_frames[i]:faces_frames[i+1]
                            ].to(device)).cpu().numpy().flatten()
        score = expit(pred.mean())
        predictions[input_dir+file_names[video_idxs[i]]
                    ] = [round(score, 3), 'real' if score < 0.1 else 'fake']
        predictions[input_dir+file_names[video_idxs[i]]] = [round(score, 3), {
            'predicted_class': 'real' if score < 0.1 else 'fake', 'true_class': input_dir.split("/")[3]}]

res = list(predictions.values())

print(res)
