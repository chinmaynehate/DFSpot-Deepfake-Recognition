#!/bin/bash

pip3 install gdown

mkdir -p models
mkdir -p sample_videos
mkdir -p sample_output_videos

path=$PWD
models="/models"
sample_videos="/sample_videos"
src="/src"
utils="/src/utils"
sample_output_videos="/sample_output_videos"

models_path=$path$models
sample_videos_path=$path$sample_videos
sample_output_videos_path=$path$sample_output_videos
src_path=$path$src
utils_path=$path$utils

while getopts m: flag; do
    case "${flag}" in
    m) model_dataset=${OPTARG} ;;
    esac
done

if [[ $model_dataset = "celeb" ]]; then
    echo "Models V2, V2ST, ViT & ViTST trained on: $model_dataset dataset will be downloaded"
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo "Around 14GB of free space is required"

    cd $models_path
    gdown https://drive.google.com/uc?id=1l4CpqaUb3e5bGfkDEJRaGJsFHQnJdNpM || wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1l4CpqaUb3e5bGfkDEJRaGJsFHQnJdNpM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1l4CpqaUb3e5bGfkDEJRaGJsFHQnJdNpM" -O celeb_models.zip && rm -rf /tmp/cookies.txt || echo " Download of models failed. This happens as downloading a file from Google Drive is restricted to some limit. Try running this script after 24 hours or manually download the models."
    cd $utils_path
    python3 extract.py --f $models_path/celeb_models.zip --d $models_path
    cd $models_path
    rm -rf celeb_models.zip

fi

if [[ $model_dataset = "dfdc" ]]; then
    echo "Models V2, V2ST, ViT & ViTST trained on: $model_dataset dataset will be downloaded"
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo "Around 14GB of free space is required"

    cd $models_path
    gdown https://drive.google.com/uc?id=1_Ofv_TBOeTDSaXhjnUi1BgNT3HWUmMPI || wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1_Ofv_TBOeTDSaXhjnUi1BgNT3HWUmMPI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_Ofv_TBOeTDSaXhjnUi1BgNT3HWUmMPI" -O dfdc_models.zip && rm -rf /tmp/cookies.txt || echo " Download of models failed. This happens as downloading a file from Google Drive is restricted to some limit. Try running this script after 24 hours or manually download the models."
    cd $utils_path
    python3 extract.py --f $models_path/dfdc_models.zip --d $models_path
    cd $models_path
    rm -rf dfdc_models.zip

fi

if [[ $model_dataset = "ffpp" ]]; then
    echo "Models V2, V2ST, ViT & ViTST trained on: $model_dataset dataset will be downloaded"
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo "Around 14GB of free space is required"

    cd $models_path
    gdown https://drive.google.com/uc?id=1fvFJ6uRaZBEn_AF4KRyge2R1j3YokwZH || wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1fvFJ6uRaZBEn_AF4KRyge2R1j3YokwZH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fvFJ6uRaZBEn_AF4KRyge2R1j3YokwZH" -O ffpp_models.zip && rm -rf /tmp/cookies.txt || echo " Download of models failed. This happens as downloading a file from Google Drive is restricted to some limit. Try running this script after 24 hours or manually download the models."
    cd $utils_path
    python3 extract.py --f $models_path/ffpp_models.zip --d $models_path
    cd $models_path
    rm -rf ffpp_models.zip

fi

if [[ $model_dataset = "all" ]]; then

    echo "Models V2, V2ST, ViT & ViTST trained on: $model_dataset datasets i.e CelebDf(v2), FaceForensics++ & DFDC will be downloaded"
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo "Around 40GB of free space is required"

    cd $models_path
    gdown https://drive.google.com/uc?id=1l4CpqaUb3e5bGfkDEJRaGJsFHQnJdNpM || wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1l4CpqaUb3e5bGfkDEJRaGJsFHQnJdNpM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1l4CpqaUb3e5bGfkDEJRaGJsFHQnJdNpM" -O celeb_models.zip && rm -rf /tmp/cookies.txt || echo " Download of models failed. This happens as downloading a file from Google Drive is restricted to some limit. Try running this script after 24 hours or manually download the models."
    cd $utils_path
    python3 extract.py --f $models_path/celeb_models.zip --d $models_path
    cd $models_path
    rm -rf celeb_models.zip

    cd $models_path
    gdown https://drive.google.com/uc?id=1_Ofv_TBOeTDSaXhjnUi1BgNT3HWUmMPI || wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1_Ofv_TBOeTDSaXhjnUi1BgNT3HWUmMPI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_Ofv_TBOeTDSaXhjnUi1BgNT3HWUmMPI" -O dfdc_models.zip && rm -rf /tmp/cookies.txt || echo " Download of models failed. This happens as downloading a file from Google Drive is restricted to some limit. Try running this script after 24 hours or manually download the models."
    cd $utils_path
    python3 extract.py --f $models_path/dfdc_models.zip --d $models_path
    cd $models_path
    rm -rf dfdc_models.zip

    cd $models_path
    gdown https://drive.google.com/uc?id=1fvFJ6uRaZBEn_AF4KRyge2R1j3YokwZH || wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1fvFJ6uRaZBEn_AF4KRyge2R1j3YokwZH' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fvFJ6uRaZBEn_AF4KRyge2R1j3YokwZH" -O ffpp_models.zip && rm -rf /tmp/cookies.txt || echo " Download of models failed. This happens as downloading a file from Google Drive is restricted to some limit. Try running this script after 24 hours or manually download the models."
    cd $utils_path
    python3 extract.py --f $models_path/ffpp_models.zip --d $models_path
    cd $models_path
    rm -rf ffpp_models.zip

fi

cd $sample_videos_path
gdown https://drive.google.com/uc?id=1p51Usf4CFDOKhp9Sl4bkKQcaY1eKPsS8 || wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1p51Usf4CFDOKhp9Sl4bkKQcaY1eKPsS8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1p51Usf4CFDOKhp9Sl4bkKQcaY1eKPsS8" -O sample_videos.zip && rm -rf /tmp/cookies.txt || echo " Download of sample videos failed. This happens as downloading a file from Google Drive is restricted to some limit. Try running this script after 24 hours or manually download the models."
cd $utils_path
python3 extract.py --f $sample_videos_path/sample_videos.zip --d $sample_videos_path
cd $sample_videos_path
rm -rf sample_videos.zip

cd $sample_output_videos_path
gdown https://drive.google.com/uc?id=1sNsjijol1a3Pft5M4h0K0OxgIn85Jkvm || wget -c --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1sNsjijol1a3Pft5M4h0K0OxgIn85Jkvm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sNsjijol1a3Pft5M4h0K0OxgIn85Jkvm" -O sample_output_videos.zip && rm -rf /tmp/cookies.txt || echo " Download of sample output videos failed. This happens as downloading a file from Google Drive is restricted to some limit. Try running this script after 24 hours or manually download the models."
cd $utils_path
python3 extract.py --f $sample_output_videos_path/sample_output_videos.zip --d $sample_output_videos_path
cd $sample_output_videos_path
rm -rf sample_output_videos.zip

cd $path
pip3 install -r requirements.txt
