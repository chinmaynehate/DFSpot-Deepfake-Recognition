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
#gdown https://drive.google.com/uc?id=1gVOsRnSuvObgzLItDoxNUQOvlH2xzr4h # celeb
#gdown https://drive.google.com/uc?id=1GhKJKKhYUwtvOSRIqOQ3DCiw6Cz2BOh6 # dfdc
#gdown https://drive.google.com/uc?id=1Kz4ls7ghGZuEsXrpRQeSCYl1er7cmj4Z # ffpp

wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1gVOsRnSuvObgzLItDoxNUQOvlH2xzr4h' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gVOsRnSuvObgzLItDoxNUQOvlH2xzr4h" -O celeb_models.zip && rm -rf /tmp/cookies.txt # celeb

wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1GhKJKKhYUwtvOSRIqOQ3DCiw6Cz2BOh6' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1GhKJKKhYUwtvOSRIqOQ3DCiw6Cz2BOh6" -O dfdc_models.zip && rm -rf /tmp/cookies.txt # dfdc

wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1Kz4ls7ghGZuEsXrpRQeSCYl1er7cmj4Z' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Kz4ls7ghGZuEsXrpRQeSCYl1er7cmj4Z" -O ffpp_models.zip && rm -rf /tmp/cookies.txt # ffpp


cd $sample_videos_path
#gdown https://drive.google.com/uc?id=1zbR_SyRVR6ZPDF-CWT5B8DN96YKuau1m # sample videos

https://drive.google.com/file/d/1zbR_SyRVR6ZPDF-CWT5B8DN96YKuau1m/view?usp=sharing
wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1zbR_SyRVR6ZPDF-CWT5B8DN96YKuau1m' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zbR_SyRVR6ZPDF-CWT5B8DN96YKuau1m" -O sample_videos.zip && rm -rf /tmp/cookies.txt # sample videos


cd $utils_path
python3 extract.py --f $models_path/celeb_models.zip $models_path/dfdc_models.zip $models_path/ffpp_models.zip --d $models_path
python3 extract.py --f $sample_videos_path/sample_videos.zip --d $sample_videos_path

cd $models_path
rm -rf celeb_models.zip dfdc_models.zip ffpp_models.zip

cd $sample_videos_path
rm -rf sample_videos.zip


cd $path
pip3 install -r requirements.txt
