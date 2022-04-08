cd src

: '
Lets say you have 3 videos: a.mp4, b.mp4 & c.mp4. ou want to check their authenticity. In order to do so, place them in the sample_videos/ folder. 
Now, you have 3 options: either use models trained on FFPP, DFDC or CelebDF dataset. 
For each dataset, we have 4 models: EfficientNetV2 (V2), EfficientNetV2 ST (V2 ST), Vision Transformer (ViT) & Vision Transformer ST (VIT ST).
Consider, you decided to use the models trained on DFDC dataset. So in order to download the models and setup the files, run the setup.sh file by executing
./setup.sh -m dfdc

The -m flag with dfdc will download the models trained on DFDC dataset. Once the execution of setup.sh is finished, you can start with running the main code.

In the code, the 3 videos which are placed in the sample_videos folder are first arranged alphabetically and indexed as:
a.mp4 0
b.mp4 1
c.mp4 2

These indexes 0,1,2 are used as --video_id argument while running the main code. See the example below:
'

python3 spot_deepfakes.py \
    --media_type video \
    --data_dir ../sample_videos/ \
    --dataset dfdc \
    --model TimmV2 TimmV2ST ViT ViTST \
    --model_dir ../models \
    --video_id 0 1 2 \
    --annotate True

: '
In the above command, all the 3 videos i.e a.mp4, b.mp4 and c.mp4 are chosen via the --video_id argument. The --model argument specifies a list of
models that have to be ensembled in order to make predictions. The --data_dir argements points to the path where the 3 videos that have to be analysed are
saved. After running this code, the annotated video is stored in the src/output/ folder and the predictions are stored in src/output/predictions.csv
'

# Some other examples:
python3 spot_deepfakes.py \
    --media_type video \
    --data_dir ../sample_videos/ffpp/fake \
    --dataset ffpp \
    --model ViT ViTST \
    --model_dir ../models \
    --video_id 1 2 \
    --annotate True

python3 spot_deepfakes.py \
    --media_type video \
    --data_dir ../sample_videos/celeb/fake \
    --dataset celeb \
    --model TimmV2 TimmV2ST ViT ViTST \
    --model_dir ../models \
    --video_id 3 \
    --annotate True

python3 spot_deepfakes.py \
    --media_type video \
    --data_dir ../sample_videos/dfdc/real \
    --dataset dfdc \
    --model TimmV2ST ViTST \
    --model_dir ../models \
    --video_id 0 1 \
    --annotate True

python3 spot_deepfakes.py \
    --media_type video \
    --data_dir ../sample_videos/ffpp/fake \
    --dataset dfdc \
    --model TimmV2 TimmV2ST ViT ViTST \
    --model_dir ../models \
    --video_id 2 3 \
    --annotate True

python3 spot_deepfakes.py \
    --media_type video \
    --data_dir ../sample_videos/celeb/real \
    --dataset ffpp \
    --model ViT ViTST \
    --model_dir ../models \
    --video_id 1 \
    --annotate True



: '
If you want to test authenticity of an image, then place the image in the sample_images folder. And run the below command. The predictions will be stored in src/output/img_predictions.json file.

'

python3 spot_deepfakes.py \
    --media_type image \
    --data_dir ../sample_images/ \
    --dataset ffpp \
    --model TimmV2 TimmV2ST ViT ViTST \
    --model_dir ../models \
    --device 2  


