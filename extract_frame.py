import os
import cv2
import argparse

def extractImages(pathIn, pathOut):
    for root, dirs, files in os.walk(pathIn, topdown=False):
        for name in files:
            video_path = os.path.join(root, name)
            count = 0
            n = 1
            vidcap = cv2.VideoCapture(video_path)
            success,image = vidcap.read()
            while success:
                vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
                success,image = vidcap.read()
                if success==True:
                    directory = pathOut
                    os.chdir(directory)
                    name = name.replace(".mp4","")
                    filename = "{filename}_{num}.jpg".format(filename=name, num=n)
                    cv2.imwrite(filename, image)
                    count = count + 0.2
                    n+=1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to the folder that contains videos")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)

