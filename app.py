from kivy.lang import Builder
from kivymd.app import MDApp
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.graphics.texture import Texture
import cv2
import mediapipe as mp
import joblib

# Load trained model
model = joblib.load("gesture_model.pkl")

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

KV = """
ScreenManager:
    MainScreen:

<MainScreen>:
    name: "main"
    MDFloatLayout:
        md_bg_color: 0.08, 0.08, 0.08, 1  # Dark theme

        MDLabel:
            text: "Hand Gesture Recognition"
            font_style: "H5"
            halign: "center"
            theme_text_color: "Custom"
            text_color: 1, 1, 1, 1
            pos_hint: {"center_x": 0.5, "top": 0.95}

        MDCard:
            size_hint: 0.92, 0.6
            pos_hint: {"center_x": 0.5, "center_y": 0.68}
            elevation: 10
            radius: [20, 20, 20, 20]
            md_bg_color: 0.15, 0.15, 0.15, 1
            padding: dp(10)

            BoxLayout:
                orientation: 'vertical'
                padding: dp(15)
                spacing: dp(10)

                MDLabel:
                    text: "Detected Gesture:"
                    font_style: "H6"
                    halign: "center"
                    theme_text_color: "Custom"
                    text_color: 1, 1, 1, 1
                    size_hint_y: None
                    height: dp(30)

                MDLabel:
                    id: gesture_label
                    text: "Show a Hand Gesture"
                    font_style: "H4"
                    halign: "center"
                    theme_text_color: "Custom"
                    text_color: 1, 1, 0, 1  # Yellow text
                    padding: dp(5)

                Image:
                    id: webcam_feed
                    allow_stretch: True
                    keep_ratio: True
                    size_hint_y: None
                    height: dp(250)

        MDCard:
            size_hint: 0.9, 0.15
            pos_hint: {"center_x": 0.5, "center_y": 0.25}
            md_bg_color: 0.1, 0.1, 0.1, 1
            padding: dp(15)

            MDScrollView:
                do_scroll_x: False
                do_scroll_y: True

                MDTextField:
                    id: sentence_label
                    hint_text: "Constructed Sentence"
                    multiline: True
                    readonly: True
                    mode: "fill"
                    text_color: 1, 1, 1, 1
                    size_hint_y: None
                    height: dp(100)
                    padding: dp(10)

        BoxLayout:
            size_hint: 0.9, None
            height: dp(50)
            pos_hint: {"center_x": 0.5, "y": 0.02}
            spacing: dp(15)
            padding: dp(10)

            MDRaisedButton:
                text: "Add Word"
                on_release: app.add_word()
                md_bg_color: 0, 0.6, 1, 1  # Blue

            MDRaisedButton:
                text: "Add Space"
                on_release: app.add_space()
                md_bg_color: 0, 0.8, 0.4, 1  # Green

            MDRaisedButton:
                text: "Delete"
                on_release: app.backspace()
                md_bg_color: 1, 0.5, 0, 1  # Orange

            MDRaisedButton:
                text: "Clear All"
                on_release: app.clear_sentence()
                md_bg_color: 1, 0, 0, 1  # Red
"""

class MainScreen(Screen):
    pass

class SignTextApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "BlueGray"
        self.sm = Builder.load_string(KV)
        self.sentence = ""
        self.detected_gesture = None
        Clock.schedule_interval(self.recognize_gesture, 1.0 / 10)
        return self.sm

    def recognize_gesture(self, dt):
        ret, frame = cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        landmarks = []
        detected_gesture = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
            while len(landmarks) < 126:
                landmarks.append(0)

            detected_gesture = model.predict([landmarks])[0]

        if detected_gesture and detected_gesture != "No Gesture Detected":
            self.detected_gesture = detected_gesture
        else:
            self.detected_gesture = None

        self.sm.get_screen("main").ids.gesture_label.text = f"[b]{self.detected_gesture or 'Show a Hand Gesture'}[/b]"
        self.sm.get_screen("main").ids.gesture_label.markup = True  

        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.sm.get_screen("main").ids.webcam_feed.texture = texture

    def add_word(self):
        if self.detected_gesture:
            self.sentence += "" + self.detected_gesture
            self.sm.get_screen("main").ids.sentence_label.text = self.sentence.strip()
            self.detected_gesture = None

    def add_space(self):
        if self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
        self.sm.get_screen("main").ids.sentence_label.text = self.sentence

    def backspace(self):
        words = self.sentence.split()
        if words:
            words.pop()
            self.sentence = " ".join(words)
        self.sm.get_screen("main").ids.sentence_label.text = self.sentence

    def clear_sentence(self):
        self.sentence = ""
        self.sm.get_screen("main").ids.sentence_label.text = ""

    def on_stop(self):
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    SignTextApp().run()
