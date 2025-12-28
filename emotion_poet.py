import cv2
import time
import random
from deepface import DeepFace
import markovify

# =======================
# LOAD EMOTIONAL VOICES
# =======================
models = {
    "sad": markovify.Text(open("corpus_sad.txt").read()),
    "angry": markovify.Text(open("corpus_angry.txt").read()),
    "fear": markovify.Text(open("corpus_fear.txt").read()),
    "happy": markovify.Text(open("corpus_happy.txt").read()),
    "neutral": markovify.Text(open("corpus_neutral.txt").read())
}

fallbacks = {
    "sad": "The machine carries the weight quietly.",
    "angry": "The system tightens.",
    "fear": "Something moves too quickly to hold.",
    "happy": "A brief warmth passes through the frame.",
    "surprise": "This was not anticipated.",
    "neutral": "Nothing resolves."
}

# =======================
# TYPOGRAPHY FEEL
# =======================
font_settings = {
    "sad": (cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1),
    "angry": (cv2.FONT_HERSHEY_DUPLEX, 0.7, 2),
    "fear": (cv2.FONT_HERSHEY_PLAIN, 0.8, 1),
    "happy": (cv2.FONT_HERSHEY_COMPLEX, 0.6, 1),
    "neutral": (cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
}

# =======================
# TRUST MECHANIC
# =======================
TRUST_TIME = 25  # seconds before box disappears

# =======================
# POEM GENERATION
# =======================
def generate_poem(emotion, watch_time, distance):
    model = models.get(emotion, models["neutral"])

    # distance -> intimacy
    if distance < 120:
        max_len = 140
    elif distance < 200:
        max_len = 90
    else:
        max_len = 50

    # time -> calm
    max_len += min(30, int(watch_time / 15))

    line = model.make_short_sentence(max_len, tries=120)

    if not line:
        return fallbacks.get(emotion, "The machine waits.")

    return line

# =======================
# FRACTURED FACE BOX
# =======================
def draw_fractured_box(frame, x, y, w, h, emotion):
    jitter = {
        "angry": 6,
        "fear": 4,
        "sad": 2,
        "happy": 1,
        "neutral": 0
    }.get(emotion, 1)

    for _ in range(4):
        dx = random.randint(-jitter, jitter)
        dy = random.randint(-jitter, jitter)

        cv2.rectangle(
            frame,
            (x + dx, y + dy),
            (x + w + dx, y + h + dy),
            (200, 200, 200),
            1
        )

# =======================
# MEMORY
# =======================
poems = []  # each: text, x, y, alpha, drift, emotion

cap = cv2.VideoCapture(0)
start_time = time.time()
last_poem_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    now = time.time()
    watch_time = now - start_time

    emotion = "neutral"
    region = None
    distance = 250

    try:
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        analysis = result[0]
        emotion = analysis["dominant_emotion"]
        region = analysis["region"]

        face_area = region["w"] * region["h"]
        distance = int(100000 / max(face_area, 1))

        # speaking rate slows with trust
        cooldown = max(3, 8 - int(watch_time / 10))

        if now - last_poem_time > cooldown:
            text = generate_poem(emotion, watch_time, distance)

            poems.append({
                "text": text,
                "x": random.randint(40, w - 400),
                "y": random.randint(80, h - 40),
                "alpha": 255,
                "drift": random.choice([-1, 0, 1]),
                "emotion": emotion
            })

            last_poem_time = now

    except:
        pass

    # =======================
    # DRAW FACE (IF TRUST NOT EARNED)
    # =======================
    if region and watch_time < TRUST_TIME:
        draw_fractured_box(
            frame,
            region["x"],
            region["y"],
            region["w"],
            region["h"],
            emotion
        )

        font, scale, thickness = font_settings.get(emotion, font_settings["neutral"])

        cv2.putText(
            frame,
            emotion.upper(),
            (region["x"], region["y"] + region["h"] + 20),
            font,
            scale,
            (200, 200, 200),
            thickness,
            cv2.LINE_AA
        )

    # =======================
    # RENDER MEMORY
    # =======================
    overlay = frame.copy()

    for poem in poems[:]:
        poem["alpha"] -= 1
        poem["y"] += poem["drift"]

        if poem["alpha"] <= 0:
            poems.remove(poem)
            continue

        font, scale, thickness = font_settings.get(poem["emotion"], font_settings["neutral"])
        color = (poem["alpha"], poem["alpha"], poem["alpha"])

        cv2.putText(
            overlay,
            poem["text"],
            (poem["x"], poem["y"]),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA
        )

    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    cv2.imshow("It Watches Until It Trusts You", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
