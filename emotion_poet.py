import cv2
import time
import random
from deepface import DeepFace
import markovify

# Load poetic brain
with open("corpus.txt") as f:
    text_model = markovify.Text(f.read())


fallbacks = {
    "sad": "The machine feels the weight but cannot lift it.",
    "angry": "The system tightens its logic.",
    "fear": "Something moves too fast to name.",
    "happy": "A light pattern emerges briefly.",
    "surprise": "Even the system did not expect this.",
    "neutral": "Nothing resolves."
}

def generate_poem(emotion):
    max_length = {
        "sad": 120,
        "angry": 60,
        "fear": 70,
        "happy": 100,
        "surprise": 80,
        "neutral": 90
    }.get(emotion, 80)

    line = text_model.make_short_sentence(max_length, tries=100)

    if not line:
       return f"[{emotion}] {fallbacks.get(emotion, 'The machine waits.')}"

    return f"[{emotion}] {line}"


cap = cv2.VideoCapture(0)

last_emotion = None
last_poem_time = 0
poem = ""

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

        if emotion != last_emotion and time.time() - last_poem_time > 3:
            poem = generate_poem(emotion)
            last_emotion = emotion
            last_poem_time = time.time()

        cv2.putText(frame, poem, (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 1)

    except:
        pass

    cv2.imshow("The Machine Writes Because You Are Here", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
