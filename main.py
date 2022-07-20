import model as m
import argparse

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathInH", help="path to the hand keypoint dataset file")
    a.add_argument("--pathInB", help="path to the body keypoint dataset file")
    args = a.parse_args()
    m.train_model(args.pathInH, args.pathInB)