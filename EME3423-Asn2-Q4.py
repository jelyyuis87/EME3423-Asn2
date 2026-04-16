import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:
    while cam.isOpened():
        ret, img = cam.read()
        if not ret:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        gesture = "None"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 取 landmark 座標
                lm = hand_landmarks.landmark

                # 食指 (8 vs 6)，中指 (12 vs 10)
                index_up = lm[8].y < lm[6].y
                middle_up = lm[12].y < lm[10].y

                # 拇指 (4 vs 3)，無名指 (16 vs 14)，小指 (20 vs 18)
                thumb_fold = lm[4].x < lm[3].x  # 簡單判斷拇指收起
                ring_down = lm[16].y > lm[14].y
                pinky_down = lm[20].y > lm[18].y

                if index_up and middle_up and ring_down and pinky_down:
                    gesture = "V Sign"

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 顯示結果
        cv2.rectangle(img, (20, 20), (300, 100), (0, 0, 0), -1)
        cv2.putText(img, f'Gesture: {gesture}', (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Gesture Detection", img)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
cam.release()
cv2.destroyAllWindows()
