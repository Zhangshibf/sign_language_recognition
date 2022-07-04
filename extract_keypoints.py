import mediapipe as mp
import cv2
import os


#Pose
def extract_pose_keypoints(path_dataset, path_frames):
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    points = mpPose.PoseLandmark
    d = []
    m = 0
    for point in points:
        x = str(point)[13:]
        d.append(x + "_x")
        d.append(x + "_y")
        d.append(x + "_z")
        d.append(x + "_vis")
        m += 1
        if m == 23:
            break  # we only need information from upper body

    f = open(path_dataset, "a", encoding="utf-8")
    joined_d = ",".join(d)
    f.write(str("frame_name," + joined_d + "\n"))
    f.close()

    f = open(path_dataset, "a", encoding="utf-8")
    nr = 0
    names = list()
    for root, dirs, files in os.walk(path_frames, topdown=False):
        for name in files:
            temporal = []
            image_path = os.path.join(root, name)
            img = cv2.imread(image_path)
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(imgRGB)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                m = 0
                for i, j in zip(points, landmarks):
                    temporal.extend([str(j.x), str(j.y), str(j.z), str(j.visibility)])
                    m += 1
                    if m == 23:
                        break  # we only need information from upper body
                temporal_joined = ",".join(temporal)
                line = str(name + "," + temporal_joined + "\n")
                f.write(line)
            else:
                nr+=1
                names.append(name)
    print("Not recognized")
    print(nr)
    print(names)
    f.close()

if __name__=="__main__":
    extract_pose_keypoints(path_dataset="D:\\LST\\第二学期\\Sign_Language_Recognition_and_Translation\\lsa64_cut_keypoints.txt",path_frames="D:\\lsa64_cut_frames" )