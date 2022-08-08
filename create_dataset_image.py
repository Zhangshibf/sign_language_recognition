import torch
import cv2
import torchvision.transforms as transforms
import os
import argparse

def create_image_dataset(path_frames,path_dataset):

    f = open(path_dataset, "a", encoding="utf-8")
    nr = 0
    names = list()
    for root, dirs, files in os.walk(path_frames, topdown=False):
        for name in files:
            image_path = os.path.join(root, name)
            img = cv2.imread(image_path)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            transform = transforms.Compose([transforms.ToTensor()])
            tensor = transform(imgRGB)
            flat = torch.flatten(tensor).tolist()
            temporal_joined = ",".join(flat)
            line = str(name + "," + temporal_joined + "\n")
            f.write(line)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to the folder that contains the frames")
    a.add_argument("--pathOut", help="path to the files where you'd like to save the data")
    args = a.parse_args()
    print(args)
    create_image_dataset(args.pathIn, args.pathOut)