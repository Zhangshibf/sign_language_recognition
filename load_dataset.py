import pickle
from collections import Counter
import argparse
import pandas as pd

class Dataset():

    def __init__(self, path_dataset):
        instances = list()
        labels = list()
        dataset_df = pd.read_csv(path_dataset,header=None)
        idxs = dataset_df[0].tolist()
        del dataset_df[0]
        features = dataset_df.values.tolist()


        # each row corresponds to a frame. Now we need to group all frame data of one video together. Each video should have only one label
        videos = list(Counter([i[:11] for i in idxs]))
        videos.sort()
        videos = videos[1:]

        for video_name in videos:
            labels.append(video_name)
            # find all frames of the same video, group them together
            frame_names = [i for i in idxs if video_name in i]
            # make sure an instance is composed of [frame1,frame2,frame3...] in stead of random order
            frame_names_dict = dict()
            for item in frame_names:
                frame_names_dict[item] = int(item.split("_")[-1].rstrip(".jpg"))
            frame_name_tuples = sorted(((v, k) for k, v in frame_names_dict.items()), reverse=False)
            frame_name_sorted = [i[1] for i in frame_name_tuples]
            frame_idxs=list()
            for frame_name in frame_name_sorted:
                frame_idxs.append(idxs.index(frame_name))
            video_feature = list()
            for frame_idx in frame_idxs:
                video_feature.append(features[frame_idx])
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

    def train_dev_test_split(self, dev=9,test=10):
        # dev is the ID of the signer to be used as dev set.
        idx_signer_dev = [i for i, x in enumerate(self.signers) if x == dev]
        dev_x = [self.instances[idx] for idx in idx_signer_dev]
        dev_y = [self.labels[idx] for idx in idx_signer_dev]
        self.dev_x,self.dev_y = sklearn.utils.shuffle(dev_x,dev_y)

        idx_signer_test = [i for i, x in enumerate(self.signers) if x == test]
        test_x = [self.instances[idx] for idx in idx_signer_test]
        test_y = [self.labels[idx] for idx in idx_signer_test]
        self.test_x,self.test_y = sklearn.utils.shuffle(test_x,test_y)

        idx_signer_train = [i for i, x in enumerate(self.signers) if x not in [dev, 10]]
        train_x = [self.instances[idx] for idx in idx_signer_train]
        train_y = [self.labels[idx] for idx in idx_signer_train]
        train_x,train_y = self.data_augmentation(train_x,train_y)
        self.train_x, self.train_y= sklearn.utils.shuffle(train_x,train_y)

    def data_augmentation(self,x,y):
        augmented_x = list()
        for x in x:
            x1 = x[1::2]
            x2 = x[0::1]
            x3 = x[3:][:-2]
            augmented_x.append(x1)
            augmented_x.append(x2)
            augmented_x.append(x3)
            augmented_x.append(x)

        augmented_y =list((itertools.chain.from_iterable(itertools.repeat(x, 4) for x in y)))#remember to change the number!

        return augmented_x,augmented_y

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
    a.add_argument("--pathDataset", help="path to the Dataset")
    a.add_argument("--pickledFilePath",help = "path of the pickled dataset object")
    args = a.parse_args()
    dataset = Dataset(args.pathDataset)
    with open(args.pickledFilePath, 'wb') as outp:
        pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()
