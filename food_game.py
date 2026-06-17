import cv2
import mediapipe as mp
import random
import math

# ==========================
# Load Images
# ==========================
images = {}

images["burger"] = cv2.resize(
    cv2.imread("burger.png", cv2.IMREAD_UNCHANGED),
    (80, 80)
)

images["pizza"] = cv2.resize(
    cv2.imread("pizza.png", cv2.IMREAD_UNCHANGED),
    (80, 80)
)

images["broccoli"] = cv2.resize(
    cv2.imread("broccoli.png", cv2.IMREAD_UNCHANGED),
    (80, 80)
)

images["bomb"] = cv2.resize(
    cv2.imread("bomb.png", cv2.IMREAD_UNCHANGED),
    (80, 80)
)

# Check if images loaded
for name, img in images.items():
    if img is None:
        print(f"ERROR: {name}.png not found")
        exit()

# ==========================
# PNG Overlay Function
# ==========================
def overlay_png(background, overlay, x, y):

    x = int(x)
    y = int(y)

    h, w = overlay.shape[:2]

    if x < 0 or y < 0:
        return

    if x + w > background.shape[1]:
        return

    if y + h > background.shape[0]:
        return

    if overlay.shape[2] == 4:

        alpha = overlay[:, :, 3] / 255.0

        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha * overlay[:, :, c]
                + (1 - alpha)
                * background[y:y+h, x:x+w, c]
            )

    else:
        background[y:y+h, x:x+w] = overlay

# ==========================
# MediaPipe Face Mesh
# ==========================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================
# Camera
# ==========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not detected!")
    exit()

# ==========================
# Game Variables
# ==========================
score = 0
lives = 3
game_over = False

foods = [
    "pizza",
    "burger",
    "burger",
    "pizza",
    "broccoli",
    "bomb"
]

current_food = random.choice(foods)

food_x = random.randint(50, 500)
food_y = 0

speed = 10

# ==========================
# Main Loop
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    mouth_open = False
    mouth_x = 0
    mouth_y = 0

    # ==========================
    # Detect Mouth
    # ==========================
    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        upper = face_landmarks.landmark[13]
        lower = face_landmarks.landmark[14]

        upper_x = int(upper.x * w)
        upper_y = int(upper.y * h)

        lower_x = int(lower.x * w)
        lower_y = int(lower.y * h)

        mouth_x = (upper_x + lower_x) // 2
        mouth_y = (upper_y + lower_y) // 2

        mouth_gap = abs(lower_y - upper_y)

        cv2.circle(
            frame,
            (mouth_x, mouth_y),
            5,
            (0, 255, 0),
            -1
        )

        if mouth_gap > 20:
            mouth_open = True

    # ==========================
    # Game Logic
    # ==========================
    if not game_over:

        food_y += speed

        overlay_png(
            frame,
            images[current_food],
            int(food_x),
            int(food_y)
        )

        food_center_x = food_x + 40
        food_center_y = food_y + 40

        distance = math.sqrt(
            (food_center_x - mouth_x) ** 2 +
            (food_center_y - mouth_y) ** 2
        )

        # Eat food
        if distance < 60 and mouth_open:

            if current_food == "burger":
                score += 1

            elif current_food == "pizza":
                score += 2

            elif current_food == "broccoli":
                lives -= 1

            elif current_food == "bomb":
                game_over = True

            food_x = random.randint(50, w - 100)
            food_y = 0
            current_food = random.choice(foods)

            speed = 10 + score * 0.2

        # Missed food
        if food_y > h:

            food_x = random.randint(50, w - 100)
            food_y = 0
            current_food = random.choice(foods)

        if lives <= 0:
            game_over = True

    # ==========================
    # Display Text
    # ==========================
    cv2.putText(
        frame,
        f"Score: {score}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Lives: {lives}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    if mouth_open:
        cv2.putText(
            frame,
            "MOUTH OPEN",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        "Press Q to Quit",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if game_over:

        cv2.putText(
            frame,
            "GAME OVER",
            (120, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            4
        )

        cv2.putText(
            frame,
            f"Final Score: {score}",
            (120, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    cv2.imshow("Food Catch Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==========================
# Cleanup
# ==========================
cap.release()
cv2.destroyAllWindows()