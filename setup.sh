pip3 install gdown
gdown https://drive.google.com/uc?id=1xnF7xwUitiXvkRYk1Y2hRRGKNrUrtlv0 #Sample videos
gdown https://drive.google.com/uc?id=1hCZ9V1CnzjX9a33_uB3ZJMSqioGooFSS #Models
unzip -qq sample_videos.zip -d .
rm -rf sample_videos.zip
unzip -qq models.zip -d .
rm -rf models.zip
pip3 install -r requirements.txt
