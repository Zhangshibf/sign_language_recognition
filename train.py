import model
import argparse
if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathDataset", help="path to the Dataset")
    args = a.parse_args()
    model.cross_val(args.pathDataset)