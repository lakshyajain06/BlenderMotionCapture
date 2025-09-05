import mediapipe as mp


class MediaPipeDetector:
    def __init__(self):
        handSolution = mp.solutions.hands
        self.hands = handSolution.Hands()
 
    def get_poses(self, frame1, frame2):

        hands_found1 = self.hands.process(frame1,)
        hands_found2 = self.hands.process(frame2,)

        foundInLeft = False
        foundInRight = False

        coords1 = []
        coords2 = []

        if hands_found1.multi_hand_landmarks:
            foundInLeft = True
            hand = hands_found1.multi_hand_landmarks[0]
            h, w, c = frame1.shape
            for point in hand.landmark:
                x, y = int(point.x * w), int(point.y * h)
                coords1.append((x, y))

        if hands_found2.multi_hand_landmarks:
            foundInRight = True
            hand = hands_found2.multi_hand_landmarks[0]
            h, w, c = frame2.shape
            for point in hand.landmark:
                x, y = int(point.x * w), int(point.y * h)
                coords2.append((x, y))
        
        return foundInLeft, foundInRight, coords1, coords2