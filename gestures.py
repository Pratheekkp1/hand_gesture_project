"""Rule-based hand gesture classification from MediaPipe hand landmarks.

MediaPipe returns 21 normalized landmarks per hand:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models
"""

THUMB_TIP, THUMB_IP = 4, 3
FINGER_TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky
FINGER_PIPS = [6, 10, 14, 18]   # middle joint of each finger

# Each vocabulary maps a finger-state tuple (thumb, index, middle, ring, pinky) to a label.
GESTURE_SETS = {
    "basic": {
        (0, 0, 0, 0, 0): "Fist",
        (0, 1, 1, 0, 0): "Peace",
        (1, 0, 0, 0, 0): "Thumbs Up",
        (1, 1, 1, 1, 1): "Open Hand",
        (0, 1, 0, 0, 0): "Pointing",
        (1, 1, 0, 0, 1): "Rock On",
    },
    # Simplified volleyball referee / bench signals
    "volleyball": {
        (1, 1, 1, 1, 1): "Time-out",
        (1, 0, 0, 0, 0): "Substitution",
        (0, 1, 1, 0, 0): "Serve",
        (0, 0, 0, 0, 0): "Out",
    },
}


def finger_states(landmarks, handedness):
    """Return a (thumb, index, middle, ring, pinky) tuple of 1 (extended) / 0 (folded).

    `landmarks` are MediaPipe normalized landmarks (x grows right, y grows down).
    `handedness` is "Left" or "Right" as reported by MediaPipe for the
    (already mirrored) frame, so the thumb test flips with the hand.
    """
    tip_x, ip_x = landmarks[THUMB_TIP].x, landmarks[THUMB_IP].x
    thumb_out = tip_x < ip_x if handedness == "Right" else tip_x > ip_x

    fingers = [int(landmarks[tip].y < landmarks[pip].y)
               for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)]
    return (int(thumb_out), *fingers)


def classify(states, gesture_set="basic"):
    """Map finger states to a gesture label, falling back to a finger count."""
    label = GESTURE_SETS[gesture_set].get(states)
    if label:
        return label
    count = sum(states)
    return f"{count} finger{'s' if count != 1 else ''} up"
