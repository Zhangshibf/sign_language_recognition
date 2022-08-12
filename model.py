import pickle5 as pickle
import torch
import torch.nn as nn
import itertools
import torch.optim as optim
import argparse
from collections import Counter
import sklearn
from sklearn.utils import shuffle

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


class sign_translator(nn.Module):

    def __init__(self, hidden_size,output_size, bidirectional=True, num_layers=2, dropout=0.3):

        super(sign_translator, self).__init__()
        self.layer1 = nn.LSTM(48,hidden_size,bidirectional = bidirectional,num_layers=num_layers,dropout=dropout)      #The LSTM layer
        self.layer2 = nn.Linear(hidden_size*2 if bidirectional == True else hidden_size,output_size)  #The linear layer
        self.layer3 = nn.Softmax(dim=1)   #The output layer with softmax

    def forward(self, vectors):
        # reshape the input for LSTM layer. The size of the expected input is [sequence length x 1 x 48]
        video_input = torch.reshape(vectors, [vectors.shape[0], 1, 48])
        output_layer1, (hidden, cell) = self.layer1(video_input )
        output_layer2 = self.layer2(output_layer1)
        last_output = output_layer2[-1]
        #we should take the last output of layer2.
        prediction = self.layer3(last_output)

        return prediction



def train_model(model,x,y,device,optimizer,loss_function,shuffle =True):
    print("-------------------------------Training-------------------------------------------")
    model.train()
    epoch_loss = 0
    epoch_accuracy = 0
    data_num = len(x)
    if shuffle == True:
        x, y = sklearn.utils.shuffle(x, y)
    for train_x,train_y in zip(x,y):
        train_x = torch.tensor(train_x)
        train_y = torch.tensor(train_y)
        train_y = torch.reshape(train_y, [1])
        train_x, train_y = train_x.to(device), train_y.to(device)
        optimizer.zero_grad()
        model_prediction = model(train_x)
        loss_per_batch = loss_function(model_prediction, train_y)
        epoch_accuracy += correct_or_not(model_prediction, train_y)
        epoch_loss += loss_per_batch.item()

        loss_per_batch.backward()
        optimizer.step()

    accuracy = epoch_accuracy / data_num
    loss = epoch_loss / data_num
    print(f"The averaged loss per instance is {loss}")
    print(f"The averaged accuracy per instance is {accuracy}")


def correct_or_not(prediction,y):
    prediction = torch.max(prediction,1)[1]
    if prediction == y:
        return 1
    else:
        return 0


def evaluate_model(model, x,y, device,loss_function):
    print("------------------------------------Evaluation---------------------------------------------")
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    data_num = len(x)
    with torch.no_grad():
        for dev_x,dev_y in zip(x,y):
            dev_x = torch.tensor(dev_x)
            dev_y = torch.tensor(dev_y)
            dev_y = torch.reshape(dev_y, [1])
            dev_x, dev_y = dev_x.to(device), dev_y.to(device)

            model_prediction = model(dev_x)
            loss = loss_function(model_prediction, dev_y)
            epoch_accuracy += correct_or_not(model_prediction, dev_y)
            epoch_loss += loss.item()

        accuracy = epoch_accuracy /data_num
        loss = epoch_loss /data_num
        print(f"The averaged loss is {loss}")
        print(f"The averaged accuracy is {accuracy}")


def train_test(pathDataset,lr= 0.001,epoch=20):
    with open(pathDataset, 'rb') as inp:
        dataset = pickle.load(inp)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_function = nn.functional.cross_entropy
    model = sign_translator(hidden_size=64, output_size=64)
    model.to(device)
    dataset.train_dev_test_split()
    for i in range(1,epoch+1):
        print(f"--------------Epoch {i}---------------")
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        train_model(model, dataset.train_x,dataset.train_y,device, optimizer, loss_function)
        evaluate_model(model, dataset.dev_x,dataset.dev_y,device,loss_function)
        print("--------------Evaluate on Test---------------")
        evaluate_model(model, dataset.test_x,dataset.test_y,device, loss_function)


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathDataset", help="path to the pickled dataset object")
    args = a.parse_args()
    train_test(args.pathDataset)
