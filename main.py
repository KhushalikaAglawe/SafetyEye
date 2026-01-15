import cv2
from ultralytics import YOLO
from flask import Flask, render_template, Response
import sqlite3
from datetime import datetime

app = Flask(__name__)
model = YOLO('models/ppe.pt') # Load our AI brain

# Step 7: Database Setup
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS violations 
                 (id INTEGER PRIMARY KEY, timestamp TEXT, gear_missing TEXT)''')
    conn.commit()
    conn.close()

def generate_frames():
    cap = cv2.VideoCapture(0) # 0 is your default webcam
    while True:
        success, frame = cap.read()
        if not success: break

        # Step 5: Detection Logic
        results = model.predict(frame, conf=0.5)
        
        for r in results:
            annotated_frame = r.plot() # Draws boxes automatically
            
            # Step 6 & 7: Check for violations (simplified)
            # If "No-Helmet" class is detected, log it
            for box in r.boxes:
                class_id = int(box.cls[0])
                label = model.names[class_id]
                if "No-" in label: # Example: "No-Helmet"
                    save_violation(label)

        # Convert image to bytes for the website
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def save_violation(gear):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO violations (timestamp, gear_missing) VALUES (?, ?)", (now, gear))
    conn.commit()
    conn.close()

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    init_db()
    # Change this line to include host='0.0.0.0' so it's easy to access
    app.run(debug=True, host='0.0.0.0', port=5000)