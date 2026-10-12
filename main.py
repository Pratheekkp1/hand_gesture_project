import argparse
import time

import cv2
import mediapipe as mp

from gestures import GESTURE_SETS, classify, finger_states


def parse_args():
    p = argparse.ArgumentParser(description="Real-time hand gesture recognition from a webcam.")
    p.add_argument("--mode", choices=GESTURE_SETS.keys(), default="basic",
                   help="which gesture vocabulary to recognize")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--max-hands", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    prev = time.time()
    with mp_hands.Hands(max_num_hands=args.max_hands,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror so on-screen movement matches the user's hand like a selfie view
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            h, w, _ = frame.shape

            if result.multi_hand_landmarks:
                for hand_lms, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                    side = hand_info.classification[0].label
                    states = finger_states(hand_lms.landmark, side)
                    label = f"{side}: {classify(states, args.mode)}"

                    # Label above the wrist
                    wrist = hand_lms.landmark[0]
                    x, y = int(wrist.x * w), int(wrist.y * h) + 40
                    cv2.putText(frame, label, (max(x - 80, 10), min(y, h - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            now = time.time()
            fps = 1 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"{args.mode} | {fps:.0f} FPS | q to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
