import pandas as pd

def load_data(pathH,pathB):
    hand = pd.read_csv(pathH,sep=",")
    body = pd.read_csv(pathB,sep=",")


class sign_translator(nn.Module):
    pass


def train_model(model,train_data,optimizer,loss_function):
    pass

def calculate_accuracy_per_batch(prediction,y):

    prediction = torch.max(prediction,1)[1]
    correct = 0
    for i,j in zip(prediction,y):
        if i==j:
            correct+=1
    accuracy_per_batch = correct/len(y)

    return accuracy_per_batch