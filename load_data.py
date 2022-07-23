from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
def load_data(pathH,pathB):
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
    v1, v3 and v5 are estimate in different ways, v1 = v2, v3 = (v2+v4)/2, v5=v4
    Now we concatenate each wn with each vn. Our final dataset looks like this:
    
    frame_index     features
    a                w1+v1
    b                w2+v2
    c                w3+v3
    d                w4+v4
    e                w5+v5
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



#split hand data and body data into train, dev, and test
#I want to use videos of subject 7 as dev set and 8 as test set.
    body_train = does_not_contains(body_24, "_008_|_007_")
    body_dev = get_rows(body_24, "_007_")
    body_test = get_rows(body_24, "_008_")

    hand_train = does_not_contains(hand_24, "_007_|_008_")
    hand_dev = get_rows(hand_24, "_007_")
    hand_test = get_rows(hand_24, "_008_")






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
