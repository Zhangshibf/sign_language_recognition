from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import argparse

def normalize_dataset(pathH,pathB,path_dataset):
    hand = pd.read_csv(pathH,sep=",") #each frame correspond to a 64-dimensional vector
    body = pd.read_csv(pathB,sep=",").iloc[:,45:69] #delete keypoints related to face and hand. Each frame correspond to a 24-dimensional vector

    #normalization
    pca = PCA(n_components=24)
    hand_standard = StandardScaler().fit_transform(hand.iloc[:, 1:])
    hand_24 = pca.fit_transform(hand_standard)
    hand_24_pd = pd.DataFrame(hand_24)
    hand_24_pd["index"] = hand.iloc[:, 0]
    hand_mean = list(hand_24_pd.mean(axis=0))

    body = pd.read_csv(pathB, sep=",")
    upper_body = body.iloc[:,
                 45:69]  # delete keypoints related to face and hand. Each frame correspond to a 24-dimensional vector
    body_24 = StandardScaler().fit_transform(upper_body)
    body_24_pd = pd.DataFrame(body_24)
    body_24_pd["index"] = body.iloc[:, 0]

    #concatenate hand data with body data
    """
    Every frame has a body position estimation, however, about one third of frames does not have hand keypoint data because the recognition wasn't successful
    Say that we have 5 frames for one video, namely a, b, c, d, e
    body keypoint vectors, w1, w2, w3, w4, w5
    only 2 hand keypoint vectors v2,v4
    v1, v3 and v5 are going to be the averaged values of hand df
    """
    dataset = list()
    blank_row = [0] * 24
    body_idx = list(hand_24_pd["index"])
    hand_idx = list(hand_24_pd["index"])
    for b_idx in body_idx:
        row = list()
        row.extend(b_idx)
        if b_idx in hand_idx:
            row.extend(body_24_pd.loc[body_24_pd['index'] == b_idx].values.flatten().tolist()[:-1])
            row.extend(hand_24_pd.loc[hand_24_pd['index'] == b_idx].values.flatten().tolist()[:-1])
            dataset.append(row)
        else:
            row.extend(body_24_pd.loc[body_24_pd['index'] == b_idx].values.flatten().tolist()[:-1])
            row.extend(hand_mean)
            dataset.append(row)

    #and now the dataset is ready!
    f = open(path_dataset, "a", encoding="utf-8")
    for row in dataset:
        line = ",".join(row)
        f.write(line)
    f.close()

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathH", help="path to the folder that contains the frames")
    a.add_argument("--pathB", help="path to the files where you'd like to save the data")
    a.add_argument("--path_dataset", help="path to the files where you'd like to save the data")
    args = a.parse_args()
    print(args)
    normalize_dataset(args.pathH,args.pathB,args.path_dataset)

#split hand data and body data into train, dev, and test
#I want to use videos of subject 7 as dev set and 8 as test set.
#    body_train = does_not_contains(body_24, "_008_|_007_")
#    body_dev = get_rows(body_24, "_007_")
#    body_test = get_rows(body_24, "_008_")

#    hand_train = does_not_contains(hand_24, "_007_|_008_")
#    hand_dev = get_rows(hand_24, "_007_")
#    hand_test = get_rows(hand_24, "_008_")






def create_targets(df):
    idx = list(df.iloc[:, 0])
    y = list()
    for i in idx:
        y.append(int(i[:3]))

    return y


def get_rows(df,string):
    return df[df.frame_name.str.contains(string,regex=False)]

def does_not_contains(df,string):
    return df[~df.frame_name.str.contains(string)]
