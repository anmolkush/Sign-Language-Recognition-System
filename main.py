import csv
import copy
import cv2 as cv
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_landmarks, get_args, pre_process_landmark

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition")
        self.root.geometry("1000x700")
        self.root.configure(bg='#2c3e50')

        self.cap = None
        self.running = False
        self.recognized_word = ""  # To store the recognized word

        self.create_widgets()
        self.setup_model()

    def create_widgets(self):
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=8)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_rowconfigure(5, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Title Label
        self.title_label = tk.Label(self.root, text="Sign Language Recognition", font=("Helvetica", 24, "bold"), bg='#2c3e50', fg='#ecf0f1')
        self.title_label.grid(row=0, column=0, pady=20, sticky="n")

        # Video Frame
        self.video_frame = tk.Frame(self.root, bg='#34495e')
        self.video_frame.grid(row=1, column=0, pady=10, sticky="nsew")
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(fill="both", expand=True)

        # Instructions Label
        self.instructions_label = tk.Label(self.root, text="Press 'Start' to begin recognition and 'Stop' to end.", font=("Helvetica", 14), bg='#2c3e50', fg='#ecf0f1')
        self.instructions_label.grid(row=2, column=0, pady=10, sticky="n")

        # Buttons Frame
        self.buttons_frame = tk.Frame(self.root, bg='#2c3e50')
        self.buttons_frame.grid(row=3, column=0, pady=20, sticky="n")

        self.start_button = tk.Button(self.buttons_frame, text="Start", font=("Helvetica", 14), command=self.start_recognition, bg='#27ae60', fg='#ecf0f1', activebackground='#2ecc71', activeforeground='#ecf0f1')
        self.start_button.grid(row=0, column=0, padx=20, pady=10)

        self.stop_button = tk.Button(self.buttons_frame, text="Stop", font=("Helvetica", 14), command=self.stop_recognition, bg='#c0392b', fg='#ecf0f1', activebackground='#e74c3c', activeforeground='#ecf0f1')
        self.stop_button.grid(row=0, column=1, padx=20, pady=10)

        self.exit_button = tk.Button(self.buttons_frame, text="Exit", font=("Helvetica", 14), command=self.exit_app, bg='#8e44ad', fg='#ecf0f1', activebackground='#9b59b6', activeforeground='#ecf0f1')
        self.exit_button.grid(row=0, column=2, padx=20, pady=10)

        # Status Bar
        self.status_bar = tk.Label(self.root, text="Status: Ready", font=("Helvetica", 12), bg='#2c3e50', fg='#ecf0f1', anchor=tk.W)
        self.status_bar.grid(row=4, column=0, sticky="ew")

        # Footer
        self.footer = tk.Label(self.root, text="Developed by Anmol Kushwaha and Akhand Pratap Singh\n© 2024", font=("Helvetica", 10), bg='#2c3e50', fg='#ecf0f1', anchor=tk.CENTER)
        self.footer.grid(row=5, column=0, pady=10, sticky="s")

    def setup_model(self):
        args = get_args()
        self.cap_device = args.device
        self.cap_width = args.width
        self.cap_height = args.height
        self.use_static_image_mode = args.use_static_image_mode
        self.min_detection_confidence = args.min_detection_confidence
        self.min_tracking_confidence = args.min_tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.keypoint_classifier = KeyPointClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    def start_recognition(self):
        if not self.running:
            self.cap = cv.VideoCapture(self.cap_device)
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.cap_width)
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.cap_height)
            self.running = True
            self.update_status("Status: Running")
            self.update_frame()

    def stop_recognition(self):
        if self.running:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            cv.destroyAllWindows()
            self.update_status("Status: Stopped")

    def exit_app(self):
        self.stop_recognition()
        self.root.quit()

    def update_frame(self):
        if self.running:
            ret, image = self.cap.read()
            if ret:
                image = cv.flip(image, 1)
                debug_image = copy.deepcopy(image)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

                image.flags.writeable = False
                results = self.hands.process(image)
                image.flags.writeable = True

                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)

                        hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)

                        # Add the recognized sign to the word
                        if hand_sign_id != -1:
                            self.recognized_word = self.keypoint_classifier_labels[hand_sign_id]

                        debug_image = draw_landmarks(debug_image, landmark_list)
                        debug_image = draw_info_text(
                            debug_image,
                            handedness,
                            self.keypoint_classifier_labels[hand_sign_id],
                            self.recognized_word,
                            debug_image.shape[1],
                            debug_image.shape[0]
                        )

                # Convert the image to PhotoImage for Tkinter
                debug_image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
                debug_image = Image.fromarray(debug_image)
                debug_image = ImageTk.PhotoImage(image=debug_image)

                self.video_label.configure(image=debug_image)
                self.video_label.image = debug_image

            self.root.after(10, self.update_frame)

    def update_status(self, status):
        self.status_bar.config(text=status)

def draw_info_text(image, handedness, hand_sign_text, recognized_word, image_width, image_height):
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = "Predicted Text: " + hand_sign_text

    word_text = "Recognized Word: " + recognized_word

    # Set the color to dark blue
    dark_blue = (139, 0, 0)

    # Determine font scale based on image size
    font_scale = min(image_width, image_height) / 1000
    thickness = max(1, int(font_scale * 2))

    # Calculate text size
    info_text_size = cv.getTextSize(info_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    word_text_size = cv.getTextSize(word_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

    # Determine text positions
    text_x = max(10, int((image_width - info_text_size[0]) / 2))
    text_y = max(60, int(image_height * 0.1))
    word_x = max(10, int((image_width - word_text_size[0]) / 2))
    word_y = text_y + info_text_size[1] + 10

    # Display info text
    # cv.putText(image, info_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, dark_blue, thickness, cv.LINE_AA)

    # Display the recognized word below the info text
    cv.putText(image, info_text, (word_x, word_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, dark_blue, thickness, cv.LINE_AA)

    return image

if __name__ == '__main__':
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()
