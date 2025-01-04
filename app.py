from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
import pickle
import pymysql

app = Flask(__name__, template_folder="templates")

# Load the model and other necessary resources
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

labels_dict = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D',
    4: 'E',
    5: 'F',
    6: 'G',
    7: 'H',
    8: 'I',
    9: 'J',
    10: 'K',
    11: 'L',
    12: 'M',
    13: 'N',
    14: 'O',
    15: 'P',
    16: 'Q',
    17: 'R',
    18: 'S',
    19: 'T',
    20: 'U',
    21: 'V',
    22: 'W',
    23: 'X',
    24: 'Y',
    25: 'Z'
}

# Initialize webcam and hand tracking
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create a database connection
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='signlang'
)
cursor = conn.cursor()

# expected_character = None  # Initialize the expected character to None
# character_detected = None  # Initialize the detected character to None
# checking_sign = False  # Flag to check if we are currently checking a sign

# Define the route to display the live webcam feed
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learn')
def learn():
    return render_template('learn.html')

@app.route('/practice')
def practice():
    return render_template('practice.html')

@app.route('/go-back')
def go_back():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

# Define the route to provide the webcam feed
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

            if len(data_aux) == 42:
                data_aux.extend([0.0] * (84 - len(data_aux)))

            x1 = int(min(x_) * W)
            y1 = int(min(y_) * H)
            x2 = int(max(x_) * W)
            y2 = int(max(y_) * H)

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Insert the predicted character into the database
            sql = "INSERT INTO recognition_results (character_predicted) VALUES (%s)"
            try:
                cursor.execute(sql, (predicted_character,))
                conn.commit()  # Commit the changes
            except Exception as e:
                print("Error:", str(e))
                conn.rollback()  # Roll back changes in case of an error

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_predicted_character():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        data_aux = []
        x_ = []
        y_ = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                    x_.append(x)
                    y_.append(y)

            if len(data_aux) == 42:
                data_aux.extend([0.0] * (84 - len(data_aux)))

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            
            return predicted_character
        
# Define a new route to start checking a sign
@app.route('/start_checking', methods=['POST'])
def start_checking_sign():
    global expected_character, checking_sign
    character = request.form.get('character')
    expected_character = character
    checking_sign = True
    return "Start practicing " + character
# Define a new route to check the sign
@app.route('/check_sign')
def check_sign():
    global expected_character

    character = request.args.get('character')
    prediction = get_predicted_character()  # Implement a function to get the predicted character

    if character == prediction:
        return jsonify({'correct': True})
    else:
        return jsonify({'correct': False})

# Define a new route to stop checking a sign
@app.route('/stop_checking')
def stop_checking_sign():
    global checking_sign
    checking_sign = False
    return "Sign checking stopped"

@app.route('/insert_result')
def insert_result():
    character = request.args.get('character')
    result = request.args.get('result')

    # Insert the result into the database
    cursor.execute("INSERT INTO recognition_results (character_predicted) VALUES (%s)", (result,))
    conn.commit()

    return "Result inserted into the database: " + result

# Run the webcam feed and Flask app
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Initialize the camera
    app.run(debug=True, threaded=True)

    # Release resources and close the database connection
    cap.release()
    conn.close()
