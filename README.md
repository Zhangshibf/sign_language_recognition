# Sign Language Rcognition and Translation

## Dataset
Please download the dataset [here](https://drive.google.com/file/d/1C7k_m2m4n5VzI4lljMoezc-uowDEgIUh/view)

## Construct enviroment using Conda
'. .bashrc
conda env create -f env.yml
conda activate slr'

## Cut videos into frames
python cut_frames.py --pathIn=the path of the directory that contains videos --pathOut=the path of the directory where you would like to save the frames
