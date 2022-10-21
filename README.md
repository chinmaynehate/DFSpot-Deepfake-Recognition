<div id="top"></div>

<!-- PROJECT SHIELDS -->


[![PR](https://img.shields.io/badge/PRs-Welcome-<COLOR>.svg)][pullreq-url]
[![Maintenance](https://img.shields.io/badge/Maintained%3F-Yes-<COLOR>.svg)](https://https://github.com/chinmaynehate/DeepFake-Spot)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/chinmaynehate/DeepFake-Spot/blob/master/LICENSE)
[![Made with](https://img.shields.io/badge/Made%20with-Python-<COLOR>.svg)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1s0e0OO_Xcyw7S81s8GydTDtTQXJvJPpL?usp=sharing)
[![PyTorch](https://img.shields.io/badge/Uses-PyTorch-<COLOR>.svg)](https://pytorch.org/)



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/chinmaynehate/DeepFake-Spot">
    <img src="https://i.imgur.com/BhxJxjh.jpg" alt="Logo" >
  </a>

<h2 align="center">DFSpot-Deepfake-Recognition</h2>

  <p align="center">
    Determine whether a given video sequence has been manipulated or synthetically generated
    <br />
    <a href="https://github.com/chinmaynehate/DeepFake-Spot/issues">Report Bug</a>
    ·
    <a href="https://github.com/chinmaynehate/DeepFake-Spot/issues">Request Feature</a>
  </p>
</div>

<h3 align="center">⚡️ Try the demo here ⚡️</h3>
<div align="center">

  <a href="https://colab.research.google.com/drive/1s0e0OO_Xcyw7S81s8GydTDtTQXJvJPpL?usp=sharing">![example1](https://colab.research.google.com/assets/colab-badge.svg)</a>

</div>


<table >
<thead>
  <tr>
    <th "><img src="assets/gifs/celeb_fake.gif" alt="drawing" width="600" height="300"/> </th>
    <th><img src="assets/gifs/ffpp_fake.gif" alt="drawing" width="600" height="300"/></th>
  </tr>
</thead>
<tbody>
  
  </tr>
</tbody>
</table>

<div align="center">                                                                                   
    <strong >Ensemble of 4 models produce the above results on test videos from datasets such as Celeb-DF(v2), FaceForensics++ and DFDC </strong> 
                   <br/>
</div>                                                                               
                                                                                   
<br/>                                                                                   
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#repo-file-structure">Repo file structure</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>    
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## 📖 About The Project

PyTorch code for DF-Spot, a model ensemble that determines if an input video/image is real or fraudulent. To identify deepfakes, this study proposes an ensemble-based metric learning technique based on a siamese network architecture, in which four models are built beginning from a base network. This method has been validated using publicly available datasets such as Celeb-DF (v2), FaceForensics++, and DFDC.

<p align="right">(<a href="#top">back to top</a>)</p>



### :bricks: Built With

* [Python 3.6.9](https://www.python.org/)
* [PyTorch 1.8.0](https://pytorch.org/)
* [Timm 0.4.9](https://github.com/rwightman/pytorch-image-models)
* [OpenCV 4.2](https://opencv.org/)
* [Albumentations 0.5.2](https://albumentations.ai/)
* [Blazeface](https://github.com/tensorflow/tfjs-models/tree/master/blazeface)



<!-- GETTING STARTED -->
## ⚡️ Getting Started
Set up the project on your local machine by following the instructions below. You can also run the demo on Google Colab [here](https://colab.research.google.com/drive/1s0e0OO_Xcyw7S81s8GydTDtTQXJvJPpL?usp=sharing)
### :heavy_check_mark: Prerequisites
* Update system and install pip3
   ```sh
   sudo apt update
   sudo apt -y install python3-pip
   ```

* Python virtual environment (optional)
   ```sh
   sudo apt install python3-venv   
   ```


### ⚙️ Installation

1. Create a python virtual environment (optional)
   ```sh
   mkdir df_spot
   cd df_spot
   python3 -m venv df_spot_env
   source df_spot_env/bin/activate
   ```
2. Clone the repo
   ```sh
   git clone https://github.com/chinmaynehate/DeepFake-Spot.git
   ```
3. Install dependencies
   ```sh
   cd DFSpot-Deepfake-Recognition
   sudo chmod +x setup.sh
   ```                                
> **🔔 Note**
The following command, which runs the setup.sh file, requires the `-m` parameter, which accepts either dfdc, celeb, ffpp, or all as inputs. If the flag -m is used with the option dfdc, setup.sh will download the models trained on the dfdc dataset. The models are currently saved on Google Drive and there appears to be a limit to the number of files that may be downloaded using the command-line utility tool `gdown`. As a result, it is possible that this limit has been reached and you are unable to download the models. If this occurs, try running the script again after 24 hours. You can also manually download the models by visiting the google drive link for the models from the `setup.sh` file.     
Downloading the models manually is recommended.
                                                            
   ```
   ./setup.sh -m <dataset>
   ```
For eg. If you want to download models trained on dfdc dataset, then run:                                                            
   ```
   ./setup.sh -m dfdc
   ```
  The other options are: celeb, ffpp or all                              

                                



### :floppy_disk: Project file structure
After running the requirements, prerequisites and installation scripts, the directory structure of 'DFSpot-Deepfake-Recognition/' is as follows
```sh
|-- assets # contains images & gifs for readme
|-- examples.sh # contains example for running spot_deepfakes.py 
|-- models # contains twelve .pth files. These are downloaded using gdown and extracted in setup.sh
|   |-- celeb_v2.pth
|   |-- dfdc_v2st.pth
|   |-- ffpp_v2.pth
|-- README.md
|-- requirements.txt
|-- sample_images # contains sample images from test set of ffpp, celebdf & dfdc dataset. Save the images that have to be tested in this folder           
|-- sample_output_videos # contains sample output videos that are obtained after running the code 
|-- sample_videos # contains all the sample videos downloaded using gdown and extracted in setup.sh. Save the video files that have to be tested in this folder
|   |-- abc.mp4 # video whose authenticity has to be tested
|   |-- pqr.mp4 # video whose authenticity has to be tested
|-- setup.sh # downloads all the models, sample_videos and installs dependencies
|-- src
    |-- architectures # contains definitions of models
    |-- blazeface # for face extraction
    |-- ensemble_model.ipynb 
    |-- output # contains the annotated video files generated by running spot_deepfakes.py
    |   |-- abc.avi # annotated video with frame-level predictions done by the ensemble of models for sample_videos/abc.mp4
    |   |-- pqr.avi # annotated video with frame-level predictions done by the ensemble of models for sample_videos/pqr.mp4
    |   |-- predictions.csv # final prediction class of abc.mp4 and pqr.mp4 i.e real or fake is stored as csv
    |-- spot_deepfakes.py # main()
    |-- utils # contains functions for extraction of faces from videos in sample_videos, loading models, ensemble of models and annotation
```
<p align="right">(<a href="#top">back to top</a>)</p>
                                
<!-- USAGE EXAMPLES -->
## ⚡️ Usage
### :video_camera: For videos
                                
1. When `setup.sh` is executed, a few example videos from the test set of datasets such as DFDC, FFPP, and CelebDF(V2) are saved in `sample videos/` folder. Assume you run the `setup.sh` file with the -m flag option dfdc. If so, then pass dfdc as the `--dataset` argument, and the code will check for models trained on the dfdc dataset in the models directory specified by the `--model dir` argument. Command to check for deepfakes in these videos using models trained on dfdc dataset is:
```sh
python3 spot_deepfakes.py --media_type video --data_dir ../sample_videos/dfdc/fake/ --dataset dfdc --model TimmV2 TimmV2ST ViT ViTST  --model_dir ../models/ --video_id 2 3 4 --annotate True --device 0 --output_dir output/  
```
The predictions are stored in `output/predictions.csv` and video with frame level annotations of predictions made by individual models and ensemble of models is stored in `output/` folder.

2. Say you have three videos- video1.mp4, video2.mp4 and video3.mp4 and you want to check their authenticity. Place these three videos in the `sample_videos/` folder and run:
```sh
python3 spot_deepfakes.py --media_type video --data_dir ../sample_videos/ --dataset ffpp --model TimmV2 TimmV2ST ViT ViTST  --model_dir ../models/ --video_id 0 1 2 --annotate True --device 0 --output_dir output/  
```
The predictions are stored in `output/predictions.csv` and video with frame level annotations of predictions made by individual models and ensemble of models is stored in `output/` folder.

### :framed_picture: For images
                                
1. By running `setup.sh` during installation, few sample images from test set of datasets like DFDC, FFPP and CelebDF(V2) are saved in `sample_images/`. To check the authenticity of these images, run:
```sh
python3 spot_deepfakes.py --media_type image --data_dir ../sample_images/ --dataset dfdc --model TimmV2 TimmV2ST ViT ViTST --model_dir ../models  --device 0 --output_dir output/  
```
2. Say you have a few images and you need to check their authenticity. Place them in the `sample_images/` folder and run the following command:
``` sh
python3 spot_deepfakes.py --media_type image --data_dir ../sample_images/ --dataset dfdc --model TimmV2 TimmV2ST ViT ViTST --model_dir ../models  --device 0 --output_dir output/  
```
                                
The predictions are stored in `output/img_predictions.json`
 
**_For more examples, please refer to [examples.sh](https://github.com/chinmaynehate/DeepFake-Spot/blob/master/examples.sh)_**

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- CONTRIBUTING -->
## ⭐️ Contributing

Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## ⚠️ License

Distributed under the MIT License. See [`LICENSE`](https://github.com/chinmaynehate/DFSpot-Deepfake-Recognition/blob/master/LICENSE) for more information.
<p align="right">(<a href="#top">back to top</a>)</p>
                                
<!-- ACKNOWLEDGMENTS -->
## :thumbsup: Acknowledgments

* [Deepware](https://github.com/deepware/deepfake-scanner)
* [Image and Sound Processing Lab - Politecnico di Milano](https://github.com/polimi-ispl/icpr2020dfdc)
* [Triplet loss tutorial](https://omoindrot.github.io/triplet-loss)
* Facebook's DeepFake Detection Challenge (DFDC) dataset [Paper](https://arxiv.org/abs/2006.07397)
* FaceForensics++ [Paper](https://arxiv.org/abs/1901.08971)
* Celeb-DF (v2) [Paper](https://arxiv.org/abs/1909.12962) 
* [Other papers](https://docs.google.com/document/d/1vW7ksUl4JFrQwUJIJ63SSmxxsMK84iVyGbEmjOCh__c/edit?usp=sharing)

<p align="right">(<a href="#top">back to top</a>)</p>

## :star: How to cite
Plain text:
```
C. Nehate, P. Dalia, S. Naik and A. Bhan, "Exposing DeepFakes using Siamese Training," 2022 IEEE India Council International Subsections Conference (INDISCON), 2022, pp. 1-6, doi: 10.1109/INDISCON54605.2022.9862825.
```

Bibtex:
```bibtex
@INPROCEEDINGS{9862825,
  author={Nehate, Chinmay and Dalia, Parth and Naik, Saket and Bhan, Aditya},
  booktitle={2022 IEEE India Council International Subsections Conference (INDISCON)}, 
  title={Exposing DeepFakes using Siamese Training}, 
  year={2022},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/INDISCON54605.2022.9862825}}
```

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[pullreq-url]:https://github.com/chinmaynehate/DeepFake-Spot/pulls
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
