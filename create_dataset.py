from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import argparse


def create_dataset(pathH,pathB,path_dataset):
    hand = pd.read_csv(pathH,sep=",") #each frame correspond to a 64-dimensional vector

    #normalization
    pca = PCA(n_components=24)
    hand_standard = StandardScaler().fit_transform(hand.iloc[:, 1:])
    hand_24 = pca.fit_transform(hand_standard)
    hand_24_pd = pd.DataFrame(hand_24)
    hand_24_pd["index"] = hand.iloc[:, 0]
    hand_mean = list(hand_24_pd.mean(axis=0))
#    hand_mean_str = [str(i) for i in hand_mean]
    body = pd.read_csv(pathB, sep=",")
    upper_body = body.iloc[:,
                 45:69]  # delete keypoints related to face and hand. Each frame correspond to a 24-dimensional vector
    body_24 = StandardScaler().fit_transform(upper_body)
    body_24_pd = pd.DataFrame(body_24)
    body_24_pd["index"] = body.iloc[:, 0]

    #concatenate hand data with body data
    idx_both = ['029', '031', '032', '034', '035', '036', '043', '044', '045', '048', '049', '050', '051', '053', '054',
                '055', '056', '057', '058', '060', '061', '063']
    idx_left = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015',
                '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '030', '033',
                '037', '038', '039', '040', '041', '042', '046', '047', '052', '059', '062', '064']

    dataset = list()
    body_idx = list(body_24_pd["index"])
    hand_idx = list(hand_24_pd["index"])
#    hand_zero = ["0"] * 24
    for b_idx in body_idx:

        if b_idx[:3] in idx_left:
            row = list()
            row.append(b_idx)
            if b_idx in hand_idx:
                row.extend(
                    [float(i) for i in body_24_pd.loc[body_24_pd['index'] == b_idx].values.flatten().tolist()[:-1]])
                row.extend(
                    [float(i) for i in hand_24_pd.loc[hand_24_pd['index'] == b_idx].values.flatten().tolist()[:-1]])
                dataset.append(row)

            else:
                row.extend(
                    [float(i) for i in body_24_pd.loc[body_24_pd['index'] == b_idx].values.flatten().tolist()[:-1]])
                row.extend(hand_mean)
                dataset.append(row)


        elif b_idx[:3] in idx_both:
            row1 = list()
            row1.append(b_idx)
            row2 = list()
            row2.append(b_idx)  # both hand frames need two rows

            if b_idx in hand_idx:
                row1.extend(
                    [float(i) for i in body_24_pd.loc[body_24_pd['index'] == b_idx].values.flatten().tolist()[:-1]])

                row2.extend(
                    [float(i) for i in body_24_pd.loc[body_24_pd['index'] == b_idx].values.flatten().tolist()[:-1]])

                h_vector = [i for i in hand_24_pd.loc[hand_24_pd['index'] == b_idx].values.flatten().tolist()[:-1]]
                row1.extend([float(i) for i in h_vector[:24]])
                row2.extend([float(i) for i in h_vector[25:]])

                dataset.append(row1)
                dataset.append(row2)

            else:
                row1.extend(
                    [float(i) for i in body_24_pd.loc[body_24_pd['index'] == b_idx].values.flatten().tolist()[:-1]])
                row1.extend(hand_mean)
                dataset.append(row1)
                dataset.append(row1)



    #and now the dataset is ready!
    f = open(path_dataset, "a", encoding="utf-8")
    for row in dataset:
        line = ','.join(map(str, row))
        line_n = line+"\n"
        f.write(line_n)
    f.close()

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathH", help="path to the folder that contains the frames")
    a.add_argument("--pathB", help="path to the files where you'd like to save the data")
    a.add_argument("--path_dataset", help="path to the files where you'd like to save the data")
    args = a.parse_args()
    print(args)
    create_dataset(args.pathH,args.pathB,args.path_dataset)







