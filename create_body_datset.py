from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import argparse


def create_dataset(pathB,path_dataset):
    body = pd.read_csv(pathB, sep=",")
    upper_body = body.iloc[:,
                 45:]  # delete keypoints related to face
    body_24 = StandardScaler().fit_transform(upper_body)
    body_24_pd = pd.DataFrame(body_24)
    body_24_pd["index"] = body.iloc[:, 0]


    dataset = list()
    body_idx = list(body_24_pd["index"])

    for b_idx in body_idx:
        row = list()
        row.append(b_idx)
        row.extend(
                    [float(i) for i in body_24_pd.loc[body_24_pd['index'] == b_idx].values.flatten().tolist()[:-1]])
        dataset.append(row)

    #and now the dataset is ready!
    f = open(path_dataset, "a", encoding="utf-8")
    for row in dataset:
        line = ','.join(map(str, row))
        line_n = line+"\n"
        f.write(line_n)
    f.close()

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathB", help="path to the files where you'd like to save the data")
    a.add_argument("--path_dataset", help="path to the files where you'd like to save the data")
    args = a.parse_args()
    print(args)
    create_dataset(args.pathB,args.path_dataset)