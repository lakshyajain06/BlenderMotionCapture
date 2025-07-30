import cv2 as cv
import mediapipe as mp
import matplotlib as m

from StereoNormalPointSolver import StereoNormalPointSolver


handSolution = mp.solutions.hands
hands = handSolution.Hands()

solver = StereoNormalPointSolver(1.53125, (368, 368), 29)

# define camera 1
capture1 = cv.VideoCapture(0)
capture1.set(cv.CAP_PROP_FRAME_WIDTH, 368)
capture1.set(cv.CAP_PROP_FRAME_HEIGHT, 368)

# define camera 2
capture2 = cv.VideoCapture(1)
capture2.set(cv.CAP_PROP_FRAME_WIDTH, 368)
capture2.set(cv.CAP_PROP_FRAME_HEIGHT, 368)

# Create two named windows first
cv.namedWindow("cam1", cv.WINDOW_NORMAL)
cv.namedWindow("cam2", cv.WINDOW_NORMAL)

# Move them side by side
cv.moveWindow("cam1", 100, 100)       # x=100, y=100
cv.moveWindow("cam2", 800, 100)   

while True:
    isTrue, frame1 = capture1.read()
    isTrue, frame2 = capture2.read() 

    rgb_frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
    rgb_frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)

    hands_found1 = hands.process(rgb_frame1,)
    hands_found2 = hands.process(rgb_frame2,)

    text = "Not Found"

    foundInLeft = False
    foundInRight = False

    if hands_found1.multi_hand_landmarks and hands_found2.multi_hand_landmarks:
        hand = hands_found1.multi_hand_landmarks[0]
        h, w, c = frame1.shape
        point = hand.landmark[8]
        x, y = int(point.x * w), int(point.y * h)

        coord1 = (x, y)

        hand = hands_found2.multi_hand_landmarks[0]
        h, w, c = frame2.shape
        point = hand.landmark[8]
        x, y = int(point.x * w), int(point.y * h)

        coord2 = (x, y)

        pos = solver.get_coordinate(coord1, coord2)
        distance = solver.get_distance_from_cam(pos)

    if hands_found1.multi_hand_landmarks:
        foundInLeft = True

        hand = hands_found1.multi_hand_landmarks
        h, w, c = frame1.shape
        coords1 = []
        for point in hand.landmark:
            x, y = int(point.x * w), int(point.y * h)
            coords1.append((x, y))
            cv.circle(frame1, (x, y), 10, (255, 0, 255), cv.FILLED)

    if hands_found2.multi_hand_landmarks:
        foundInRight = True
        hand = hands_found2.multi_hand_landmarks[0]
        h, w, c = frame2.shape
        coords2 = []
        for point in hand.landmark:
            x, y = int(point.x * w), int(point.y * h)
            coords2.append((x, y))
            cv.circle(frame2, (x, y), 10, (255, 0, 255), cv.FILLED)

    if foundInLeft and foundInRight:
        coords3d = []
        if len(coords1) == len(coords2):
            for i in range(len(coords1)):
                pos = solver.get_coordinate(coord1[i], coord2[i])
                distance = solver.get_distance_from_cam(pos)
                coords3d.append(pos)
            text = "Found In Both"
        else:
            print("ERROR, number of landmarks not same")
    elif foundInLeft:
        text = "Found In Left Only"
    elif foundInRight:
        text = "Found In Right Only"

    cv.putText(frame1, text, (184, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv.imshow('cam1', frame1)
    cv.imshow('cam2', frame2)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture1.release()
capture2.release()
cv.destroyAllWindows()
