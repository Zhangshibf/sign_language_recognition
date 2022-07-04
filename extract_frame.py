import os
import cv2

def extractImages(pathIn, pathOut):
    for root, dirs, files in os.walk(pathIn, topdown=False):
        for name in files:
            a=123
            video_path = os.path.join(root, name)
#            count = 0
            n = 1
            vidcap = cv2.VideoCapture(video_path)
            success,image = vidcap.read()
            while success:
                directory = pathOut
                os.chdir(directory)
                name = name.replace(".mp4","")
                filename = "{filename}_{num}.jpg".format(filename=name, num=n)
                cv2.imwrite(filename, image)  # save frame as JPEG file
                success, image = vidcap.read()

#                vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
#                success,image = vidcap.read()
#                if success==True:
#                    directory = pathOut
#                    os.chdir(directory)
#                    name = name.replace(".mp4","")
#                    filename = "{filename}_{num}.jpg".format(filename=name, num=n)
#                    cv2.imwrite(filename, image)
#                    count = count + 0.1
#                    n+=1

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to the folder that contains videos")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)

