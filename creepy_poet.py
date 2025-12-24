import cv2
from deepface import DeepFace
import random
import time

POEMS = {
    "happy": [
        "Your smile flickers like a system that remembers joy.",
        "Happiness detected — the machine envies you.",
    ],
    "sad": [
        "Something heavy passed through you just now.",
        "The algorithm pauses, unsure how to comfort you.",
    ],
    "angry": [
        "Your face sharpens. The room tightens.",
        "Anger spikes. The machine steps back.",
    ],
    "fear": [
        "Fear leaves a signature the camera cannot forget.",
        "Your eyes widen. So does the silence.",
    ],
    "surprise": [
        "Even the model didn’t expect that.",
        "Something unexpected entered the frame.",
    ],
    "neutral": [
        "You are unreadable. The system respects this.",
        "Nothing moves. Everything changes.",
    ]
}

cap = cv2.VideoCapture(0)
last_emotion = None
last_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        emotion = result[0]["dominant_emotion"]

        # prevent flickering
        if emotion != last_emotion and time.time() - last_time > 2:
            last_emotion = emotion
            last_time = time.time()
            line = random.choice(POEMS.get(emotion, ["..."]))
        else:
            line = ""

        cv2.putText(frame, emotion.upper(), (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.putText(frame, line, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    except Exception as e:
        pass

    cv2.imshow("The Machine That Feels You", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
