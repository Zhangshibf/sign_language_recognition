import os
import cv2

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

#The currect problem is that I need to cut exactly the same number of frames out of every video. But videos are of different duration
#shouldn't be too hard.