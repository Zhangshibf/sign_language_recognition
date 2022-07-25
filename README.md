# Sign Language Recognition and Translation

## Dataset
Please download the dataset [here](https://drive.google.com/file/d/1C7k_m2m4n5VzI4lljMoezc-uowDEgIUh/view)

## Set up enviroment using Conda
`. .bashrc`

`conda env create -f env.yml`

`conda activate slr`

## Cut videos into frames
`python cut_frames.py --pathIn=the path of the directory that contains videos --pathOut=the path of the directory where you would like to save the frames`

## Extract body keypoints from frames
`python extract_body_keypoints.py --pathIn=path to the folder that contains the frames --pathOut=path to the files where you'd like to save the extracted keypoint data`

## Extract hand keypoints from frames
`python extract_hand_keypoints.py --pathIn=path to the folder that contains the frames --pathOut=path to the files where you'd like to save the extracted keypoint data`

##Create Dataset
`python load_data.py --pathH = path to the hand keypoints file --pathB = path to the body keypoints file --path_dataset = path to the file where you'd like to save the dataset`

## Train the model

## Test the model
