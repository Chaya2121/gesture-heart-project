from flask import Flask, send_from_directory, jsonify, render_template
import threading, time, cv2, mediapipe as mp

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

app = Flask(__name__, static_folder=".")



gesture_state = {"gesture": "none"}
lock = threading.Lock()

def set_gesture(g):
    with lock:
        gesture_state["gesture"] = g

def get_gesture():
    with lock:
        g = gesture_state["gesture"]
        gesture_state["gesture"] = "none"  # reset after read
        return g

def gesture_thread():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    cap = cv2.VideoCapture(0)
    prev_x, prev_y = None, None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0].landmark[8]  # index fingertip
            x, y = lm.x, lm.y
            if prev_x is not None and prev_y is not None:
                dx, dy = x - prev_x, y - prev_y
                if dx > 0.05: set_gesture("rotate_right")
                elif dx < -0.05: set_gesture("rotate_left")
                elif dy < -0.05: set_gesture("zoom_in")
                elif dy > 0.05: set_gesture("zoom_out")
            prev_x, prev_y = x, y

        cv2.imshow("Gesture Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC closes window
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/gesture")
def get():
    return jsonify({"gesture": get_gesture()})


@app.route("/heart.glb")
def heart_file():
    return send_from_directory(".", "heart.glb")



if __name__ == "__main__":
    threading.Thread(target=gesture_thread, daemon=True).start()
    app.run() 

if __name__ == "__main__":
    print (" Static folder path:", os.path.abspath(app.static_folder))
    app.run(debug=True)


