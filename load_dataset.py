import pickle
from collections import Counter
import argparse
import pandas as pd

class Dataset():

    def __init__(self, path_dataset):
        instances = list()
        labels = list()
#        f = open(path_dataset)
#        data = f.read()
#        rows = data.split("\n")
        dataset_df = pd.read_csv(path_dataset,header=None)
        idxs = list(dataset_df[0])
        del dataset_df[0]
        features = dataset_df.values.tolist()
#        for row in rows:
#            row_data = row.split(",")
#            idxs.append(row_data[0])
#            features.append(row_data[1:])
#remember to delete this part
        for row,idx in zip(features,idxs):
            if len(row)!=48:
                print(len(row))
                print(idx)

        print("all good!")
        pass

        # each row corresponds to a frame. Now we need to group all frame data of one video together. Each video should have only one label
        # 047_001_001_18.jpg
        videos = list(Counter([i[:11] for i in idxs]))
        videos.sort()
        videos = videos[1:]

        for video_name in videos:
            labels.append(video_name)
            # find all frames of the same video, group them together. That's an instance
            frame_names = [i for i in idxs if video_name in i]
            frame_names.sort()  # to make sure an instance is composed of [frame1,frame2,frame3...] in stead of random order
            frame_idxs = [i for i, x in enumerate(idxs) if x in frame_names]
            video_feature = list()
            for frame_idx in frame_idxs:
                video_feature.append(features[frame_idx])
            instances.append(video_feature)

        targets = self.create_targets(labels)
        signer = self.create_signer_list(labels)

        self.signers = signer
        self.labels = targets
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
    a.add_argument("--pathDataset", help="path to the Dataset")
    a.add_argument("--pickledFilePath",help = "path of the pickled dataset object")
    args = a.parse_args()
    dataset = Dataset(args.pathDataset)
    with open(args.pickledFilePath, 'wb') as outp:
        pickle.dump(dataset, outp, pickle.HIGHEST_PROTOCOL)
        outp.close()