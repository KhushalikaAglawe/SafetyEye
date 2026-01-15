import cv2
import sqlite3
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
from datetime import datetime

app = Flask(__name__)
model = YOLO('yolov8n.pt') 

def get_db():
    conn = sqlite3.connect('safety.db')
    return conn

def init_db():
    conn = get_db()
    conn.execute('CREATE TABLE IF NOT EXISTS logs (id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, msg TEXT, status TEXT)')
    conn.commit()
    conn.close()

def save_log(msg, status):
    try:
        conn = get_db()
        now = datetime.now().strftime("%H:%M:%S")
        conn.execute("INSERT INTO logs (time, msg, status) VALUES (?, ?, ?)", (now, msg, status))
        conn.commit()
        conn.close()
    except: pass

def gen_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success: break
        results = model(frame, conf=0.4, verbose=False)
        person_present = any(int(box.cls[0]) == 0 for box in results[0].boxes)
        gear_present = any(int(box.cls[0]) in [24, 27, 67] for box in results[0].boxes) # Backpack/Tie/Phone
        
        frame_count += 1
        if frame_count % 15 == 0 and person_present:
            if not gear_present:
                save_log("CRITICAL: Personnel missing Safety Gear!", "VIOLATION")
            else:
                save_log("Compliance Verified: Personnel Wearing Gear", "SAFE")

        ret, buffer = cv2.imencode('.jpg', results[0].plot())
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video')
def video(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_logs')
def get_logs():
    conn = get_db()
    logs = conn.execute('SELECT time, msg, status FROM logs ORDER BY id DESC LIMIT 10').fetchall()
    conn.close()
    return jsonify([{"time": l[0], "msg": l[1], "status": l[2]} for l in logs])

@app.route('/simulate')
def simulate():
    save_log("MANUAL ALERT: Safety Check Triggered", "VIOLATION")
    return "OK"

if __name__ == "__main__":
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5001)