import pandas as pd
import argparse
import os
import torch
import pickle
from collections import Counter
import torchvision.transforms as T
from PIL import Image

def create_image_dataset(pathFrames):
    images = list()
    names = list()
    convertor = T.Compose(
        [T.ToTensor(),
         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    for root, dirs, files in os.walk(pathFrames, topdown=False):
        for name in files:
            names.append(name)
            image_path = os.path.join(root, name)
            orig_img = Image.open(image_path)
            resized_img = T.functional.resize(orig_img, [54, 96])
            tensor_img = convertor(resized_img)
            images.append(tensor_img)
    return images,names

class Dataset():

    def __init__(self,images,names):
        videos = list(Counter([i[:11] for i in names]))
        videos.sort()
        labels = list()
        instances = list()
        for video_name in videos:
            labels.append(video_name)
            # find all frames of the same video, group them together. That's an instance
            frame_names = [i for i in names if video_name in i]
            frame_names_dict = dict()
            for item in frame_names:
                frame_names_dict[item] = int(item.split("_")[-1].rstrip(".jpg"))
            frame_name_tuples = sorted(((v,k) for k,v in frame_names_dict.items()), reverse=False)
            frame_name_sorted = [i[1] for i in frame_name_tuples]
        #    frame_names.sort(key=int)  # to make sure an instance is composed of [frame1,frame2,frame3...] in stead of random order
            frame_idxs = [i for i, x in enumerate(names) if x in frame_name_sorted]
            video_feature = list()
            for frame_idx in frame_idxs:
                video_feature.append(images[frame_idx])
            instances.append(video_feature)

        targets = self.create_targets(labels)
        signer = self.create_signer_list(labels)

        self.signers = signer
        self.labels = [i-1 for i in targets]
        self.instances = instances

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        instance = self.instances[idx]
        sample = {"Data": instance, "Class": label}

        return sample

    def create_test_set(self):
        # signer 10 is used as test set
        idx_signer10 = [i for i, x in enumerate(self.signers) if x == 10]
        self.test_x = [self.instances[idx] for idx in idx_signer10]
        self.test_y = [self.labels[idx] for idx in idx_signer10]

    def train_dev_split(self, dev):
        # dev is the ID of the signer to be used as dev set.
        idx_signer_dev = [i for i, x in enumerate(self.signers) if x == dev]
        self.dev_x = [self.instances[idx] for idx in idx_signer_dev]
        self.dev_y = [self.labels[idx] for idx in idx_signer_dev]

        idx_signer_train = [i for i, x in enumerate(self.signers) if x not in [dev, 10]]
        self.train_x = [self.instances[idx] for idx in idx_signer_train]
        self.train_y = [self.labels[idx] for idx in idx_signer_train]
        #这里得加个shuffle

    def create_targets(self, idx):
        y = list()
        for i in idx:
            video = i[:3]
            video_int = int(video.lstrip("0"))
            y.append(video_int)
        return y

    def create_signer_list(self, idx):
        signers = list()
        for i in idx:
            signer = i[4:7]
            signer_int = int(signer.lstrip("0"))
            signers.append(signer_int)
        return signers


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathFrames", help="path to the folder of frames")
 #   a.add_argument("--pathDataset",help = "path where you'd like to pickle the dataset")
    args = a.parse_args()
    print(args)
    images,names = create_image_dataset(args.pathFrames)
    print(names)
    dataset = Dataset(images,names)

    with open("/home/CE/zhangshi/signlanguage/image_dataset.pickle", 'wb') as outp:
        pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()