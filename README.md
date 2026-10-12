# Real-Time Hand Gesture Recognition

Recognizes hand gestures from a live webcam feed with **MediaPipe hand tracking** and **OpenCV**, with a mode that interprets **volleyball referee and bench signals**.

## How it works

1. **Hand tracking.** MediaPipe Hands detects up to two hands per frame and returns 21 3D landmarks each, plus whether it's a left or right hand.
2. **Finger state extraction** (`gestures.py`). Each finger is classified as extended or folded:
   - Index to pinky: the fingertip is above its middle (PIP) joint.
   - Thumb: the tip is outside the IP joint on the x-axis, **mirrored by handedness** so it works for both hands.
3. **Gesture classification.** The 5-bit finger state (e.g. `(0,1,1,0,0)`) is looked up in a gesture vocabulary, falling back to a finger count.
4. **Overlay.** Landmarks, per-hand labels and live FPS are drawn on a mirrored selfie view.

| Mode | Gestures |
|---|---|
| `basic` | Fist, Peace, Thumbs Up, Open Hand, Pointing, Rock On |
| `volleyball` | Time-out (open hand), Substitution (thumbs up), Serve (two fingers), Out (fist) |

Adding a gesture is one line in `GESTURE_SETS`.

## Usage

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python main.py                      # basic gestures
python main.py --mode volleyball    # volleyball signals
```

Options: `--camera N` to pick a webcam, `--max-hands N`. Press `q` to quit.

## Tech

Python · MediaPipe · OpenCV

## Next steps

- Replace the hand-written rules with a small classifier (k-NN or MLP) trained on recorded landmark vectors, to support dynamic and two-handed signals
- Smooth predictions over several frames to remove flicker between gestures
