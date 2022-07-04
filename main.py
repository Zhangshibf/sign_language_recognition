import extract_frame as e
import argparse

print("Here we go!")

"""
def extractImages(pathIn, pathOut):
    count = 0
    n = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    print(success)
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line 
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        directory = pathOut
        os.chdir(directory)
        filename = "frame%d.jpg" % n
        print(filename)
        if not cv2.imwrite(filename, image):
            raise Exception("Oh no!")
        count = count + 0.3
        n+=1
"""
if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to the folder that contains videos")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print(args)
    e.extractImages(args.pathIn, args.pathOut)