import cv2
import mediapipe as mp
import os

def extract_hand_keypoints(path_input,path_output):
    #write head of the dataset
    d=[]
    for point in mp.solutions.hands.HandLandmark:
        x = str(point)[13:]
        d.append(x + "_x")
        d.append(x + "_y")
        d.append(x + "_z")

    f = open(path_output, "a", encoding="utf-8")
    joined_d = ",".join(d)
    f.write(str("frame_name," + joined_d + "\n"))
    f.close()

    not_recognized = list()
    idx_both = ['029', '031', '032', '034', '035', '036', '043', '044', '045', '048', '049', '050', '051', '053', '054',
                '055', '056', '057', '058', '060', '061', '063']
    idx_left = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '030', '033', '037', '038', '039', '040', '041', '042', '046', '047', '052', '059', '062', '064']

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True,
                           max_num_hands=2,
                           min_detection_confidence=0.5)
    f = open(path_output, "a", encoding="utf-8")

    for root, dirs, files in os.walk(path_input, topdown=False):
        for idx, name in enumerate(files):
            if name[:3] in idx_left:

                image_path = os.path.join(root, name)
                img = cv2.imread(image_path, 1)
                image = cv2.flip(img, 1)
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.multi_hand_landmarks:
                    not_recognized.append(name)
                    continue

                temporal = []
                for point in results.multi_hand_landmarks[0].landmark:
                    temporal.extend([str(point.x), str(point.y), str(point.z)])
                temporal_joined = ",".join(temporal)
                line = str(name + "," + temporal_joined + "\n")
                f.write(line)

            elif name[:3] in idx_both:

                image_path = os.path.join(root, name)
                img = cv2.imread(image_path, 1)
                image = cv2.flip(img, 1)
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.multi_hand_landmarks:
                    not_recognized.append(name)
                    continue

                if len(results.multi_hand_landmarks) == 2:  # save the result only if both hands are recognized
                    temporal = []
                    for point in results.multi_hand_landmarks[0].landmark:
                        temporal.extend([str(point.x), str(point.y), str(point.z)])
                    temporal_joined = ",".join(temporal)
                    line = str(name + "," + temporal_joined + "\n")
                    f.write(line)

                    temporal = []
                    for point in results.multi_hand_landmarks[1].landmark:
                        temporal.extend([str(point.x), str(point.y), str(point.z)])
                    temporal_joined = ",".join(temporal)
                    line = str(name + "," + temporal_joined + "\n")
                    f.write(line)
                else:
                    not_recognized.append(name)
    f.close()
    print(not_recognized)

if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to the folder that contains the frames")
    a.add_argument("--pathOut", help="path to the files where you'd like to save the data")
    args = a.parse_args()
    print(args)
    extractImages(args.pathIn, args.pathOut)
