import cv2
import mediapipe as mp
import pandas as pd

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define gestures
GESTURES = [
    "Hello", "Sorry", "I Love You", "Thank You",  # 1-4
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", 
    "L", "M", "N", "O", "P", 
    "Q", "R", "S", "T", "U",
    "V", "W", "X", "Y", "Z"
]  

# Split gestures for left & right
LEFT_GESTURES = GESTURES[:15]
RIGHT_GESTURES = GESTURES[15:]

data = []  # Store collected gesture data
cap = cv2.VideoCapture(0)

print("Press a number key (1-4) or a letter (a-z) for gestures. Press 'Esc' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmarks = []  

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:  # ✅ Loop through all detected hands
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract 21 landmark positions (x, y, z)
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

    # ✅ Display left-side gestures
    for i, gesture in enumerate(LEFT_GESTURES, 1):
        key_display = str(i) if i <= 4 else chr(ord('a') + (i - 5))
        cv2.putText(frame, f"{key_display}: {gesture}", (10, 20 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # ✅ Display right-side gestures
    for i, gesture in enumerate(RIGHT_GESTURES, 1):  
        key_display = chr(ord('a') + (i - 1 + 11))
        cv2.putText(frame, f"{key_display}: {gesture}", (frame.shape[1] - 200, 20 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in [ord(str(i)) for i in range(1, 5)]:  # 1-4
        label = GESTURES[int(chr(key)) - 1]
        if landmarks:  # ✅ Only save if hand data exists
            data.append([label] + landmarks)
            print(f"Saved: {label}")

    elif key in range(ord('a'), ord('z') + 1):  # a-z for gestures 5+
        index = key - ord('a') + 4
        if index < len(GESTURES) and landmarks:
            label = GESTURES[index]
            data.append([label] + landmarks)
            print(f"Saved: {label}")

    if key == 27:  # Escape key
        break

# ✅ Dynamically create column names based on the **maximum** number of recorded landmarks
if data:
    max_landmarks = max(len(row) - 1 for row in data)  # Find the largest landmark set
    columns = ["Label"] + [f"LM{i}" for i in range(max_landmarks)]

    # ✅ Pad shorter rows with NaN to match column length
    padded_data = [row + [None] * (max_landmarks - len(row) + 1) for row in data]

    df = pd.DataFrame(padded_data, columns=columns)
    df.to_csv("gesture_data.csv", index=False)
    print(f"Dataset saved as gesture_data.csv with {df.shape[1]} columns.")

cap.release()
cv2.destroyAllWindows()
