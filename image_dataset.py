import pandas as pd
import argparse
import os
import torch
import pickle
import torchvision.transforms as T
from PIL import Image
def create_image_dataset(path_frames):
    images = list()
    names = list()
    convertor = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    for root, dirs, files in os.walk(path_frames, topdown=False):
        for name in files:
            names.append(name)
            image_path = os.path.join(root, name)
            orig_img = Image.open(image_path)
            resized_img = T.functional.resize(orig_img, [27, 48])
            images.append(convertor(resized_img))

        with open("/home/CE/zhangshi/signlanguage/image_dataset.pickle", 'wb') as outp:
            pickle.dump(images, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()

        with open("/home/CE/zhangshi/signlanguage/image_dataset_names.pickle", 'wb') as outp:
            pickle.dump(names, outp, pickle.HIGHEST_PROTOCOL)
            outp.close()

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathB", help="path to the files where you'd like to save the data")
#    a.add_argument("--path_images", help="path to the files where you'd like to save the image data")
#    a.add_argument("--path_names", help="path to the files where you'd like to save the name data")
    args = a.parse_args()
    print(args)
    create_image_dataset(args.pathB)