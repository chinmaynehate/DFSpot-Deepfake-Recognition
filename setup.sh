#!/bin/bash 

pip3 install gdown

mkdir -p models
mkdir -p sample_videos

path=$PWD
models="/models"
sample_videos="/sample_videos"
src="/src"
utils="/src/utils"

models_path=$path$models
sample_videos_path=$path$sample_videos
src_path=$path$src
utils_path=$path$utils

cd $models_path
gdown https://drive.google.com/uc?id=1gVOsRnSuvObgzLItDoxNUQOvlH2xzr4h # celeb
gdown https://drive.google.com/uc?id=1GhKJKKhYUwtvOSRIqOQ3DCiw6Cz2BOh6 # dfdc
gdown https://drive.google.com/uc?id=1Kz4ls7ghGZuEsXrpRQeSCYl1er7cmj4Z # ffpp

cd $sample_videos_path
gdown https://drive.google.com/uc?id=134pojUQrObF5sSDqaR2BX1xMx_VZn4ZD # sample videos

cd $utils_path
python3 extract.py --f $models_path/celeb_models.zip $models_path/dfdc_models.zip $models_path/ffpp_models.zip --d $models_path
python3 extract.py --f $sample_videos_path/sample_videos.zip --d $sample_videos_path

cd $models_path
rm -rf celeb_models.zip dfdc_models.zip ffpp_models.zip

cd $sample_videos_path
rm -rf sample_videos.zip
mv  sample_videos/* .
rm -rf sample_videos

pip3 install -r requirements.txt
