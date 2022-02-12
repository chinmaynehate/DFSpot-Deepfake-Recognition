cd src

python3 spot_deepfakes.py \
--media_type video \
--data_dir ../sample_videos/ffpp/fake \
--dataset ffpp \
--model TimmV2 TimmV2ST ViT ViTST \
--model_dir ../models \
--video_id 0 3 6  \
--annotate True 

python3 spot_deepfakes.py \
--media_type video \
--data_dir ../sample_videos/ffpp/real \
--dataset ffpp \
--model TimmV2ST ViT ViTST \
--model_dir ../models \
--video_id 0 8 11  \
--annotate True 
