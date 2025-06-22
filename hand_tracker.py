import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            h, w, _ = img.shape

            # Get landmark positions
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks,
                                   mp_hands.HAND_CONNECTIONS)

            if lm_list:
                fingers = []

                # Thumb (check X direction, assumes right hand)
                if lm_list[4][1] > lm_list[3][1]:
                    fingers.append(1)  # up
                else:
                    fingers.append(0)

                # Other 4 fingers (check Y direction)
                tips = [8, 12, 16, 20]
                base_joints = [6, 10, 14, 18]
                for tip, base in zip(tips, base_joints):
                    if lm_list[tip][2] < lm_list[base][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Gesture interpretation
                total_up = fingers.count(1)
                if fingers == [0, 0, 0, 0, 0]:
                    gesture = "✊ Fist"
                elif fingers == [1, 1, 0, 0, 0]:
                    gesture = "✌️ Peace"
                elif fingers == [1, 0, 0, 0, 0]:
                    gesture = "👍 Thumbs Up"
                elif fingers == [1, 1, 1, 1, 1]:
                    gesture = "🖐 Open Hand"
                else:
                    gesture = f"{total_up} fingers up"

                # Draw gesture text
                cv2.putText(img, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3)

    # Show the result
    cv2.imshow("Hand Gesture Detector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
