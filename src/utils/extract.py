
from zipfile import ZipFile
import os
import sys
IN_COLAB = 'google.colab' in sys.modules
if(IN_COLAB):
    from tqdm import tqdm_notebook as tqdm
elif(not IN_COLAB):
    from tqdm import tqdm

import gdown

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--f', nargs='+', type=str,
                        help='Path to zip file that has to be extracted', required=True)

parser.add_argument('--d', type=str,
                        help='Path to the destination where contents of zip files are to be stored', required=True)


args = parser.parse_args()

zip_files = args.f
destination = args.d
    

    
def extract_zips(zipfile_path,destination_path):
    for a,i in enumerate(zipfile_path):
        filetoextract = zipfile_path[a]   
        basename = os.path.basename(i)
        with ZipFile(filetoextract,"r") as zip_ref:
             for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()) ,desc = "Extracting "+basename):
                zip_ref.extract(file,destination_path)

if __name__=="__main__":
    extract_zips(zip_files,destination)