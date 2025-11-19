import os, io, time, math, threading, json
from collections import deque, defaultdict
from datetime import datetime
import cv2, numpy as np, requests
from flask import Flask, jsonify, send_from_directory
from ultralytics import YOLO
import pyttsx3
import speech_recognition as sr


mdl_pth = os.getenv("YOLO_MODEL_PATH", "yolov8n-pose.pt")
dsc_wbhk = os.getenv("DISCORD_WEBHOOK_URL", "URL")
en_spc = os.getenv("ENABLE_SPEECH_RECOGNITION", "1") == "1"
flsk_srv = os.getenv("FLASK_SERVER", "0") == "1"
flsk_prt = int(os.getenv("FLASK_PORT", "5000"))
upld_dir = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(upld_dir, exist_ok=True)

cnf_thr = float(os.getenv("CONF_THRESHOLD", "0.5"))
alrt_dur = float(os.getenv("ALERT_DURATION", "4.0"))
agr_thr = float(os.getenv("AGGRESSION_THRESHOLD", "3.5"))
dsc_mnt = os.getenv("DISCORD_MENTION", "@everyone").strip()

app = Flask(__name__, static_folder=upld_dir)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()}), 200

@app.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename):
    return send_from_directory(upld_dir, filename)

def send_discord_embed(title, desc, color, img_bytes=None, src_type="vision"):
    if not dsc_wbhk:
        return False
    
    embed_data = {
        "title": title,
        "description": desc,
        "color": color,
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": f"偵測來源: {src_type.upper()}"},
        "fields": [
            {"name": "偵測類型", "value": "語音" if src_type == "voice" else "鏡頭偵測", "inline": True},
            {"name": "嚴重程度", "value": "緊急" if "緊急" in title else "警告", "inline": True}
        ]
    }
    
    payload = {
        "content": dsc_mnt if dsc_mnt else None,
        "embeds": [embed_data]
    }
    
    files = None
    if img_bytes:
        fn = f"alert_{int(time.time())}.jpg"
        files = {"file": (fn, io.BytesIO(img_bytes), "image/jpeg")}
        embed_data["image"] = {"url": f"attachment://{fn}"}
        payload_json = json.dumps(payload)
        data = {"payload_json": payload_json}
        resp = requests.post(dsc_wbhk, data=data, files=files, timeout=8)
    else:
        resp = requests.post(dsc_wbhk, json=payload, timeout=8)
    
    return resp.status_code < 400

def save_frame_locally(frame):
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    fn = f"frame_{ts}.jpg"
    pth = os.path.join(upld_dir, fn)
    try:
        cv2.imwrite(pth, frame)
        _, enc = cv2.imencode(".jpg", frame)
        return pth, enc.tobytes()
    except Exception as e:
        return None, None

class FightDetectionSystem:
    def __init__(self, model_path=mdl_pth, conf_threshold=cnf_thr,
                 alert_duration=alrt_dur, aggression_threshold=agr_thr):
        if YOLO is None:
            raise RuntimeError("YOLO not loaded")
        
        self.model = YOLO(model_path)
        self.confidence_threshold = conf_threshold
        self.alert_duration = alert_duration
        self.aggression_threshold = aggression_threshold
        
        self.fight_detected = False
        self.alert_start_time = 0.0
        self.total_alerts = 0
        self.session_start_time = time.time()
        self.frame_count = 0
        self.fps_history = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        self.kpts_history = deque(maxlen=10)
        self.centers_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=10)
        self.interaction_history = deque(maxlen=10)
        self.prev_time = None
        self.consecutive_detections = 0
        self.last_alert_time = 0
        
        self.contact_tracker = {} 
        self.person_tracking = {} 
        self.next_person_id = 0
        self.contact_timeout = 1.0 
        self.min_contacts_for_fight = 2 
 
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # 保留欄位，但實際播報時會每次重新建立引擎，避免某些平台只播放一次的問題
        self.tts_engine = None
        if pyttsx3:
            try:
                # 這裡只測試是否可以初始化，不長期持有這個引擎
                test_engine = pyttsx3.init()
                test_engine.setProperty('rate', 180)
                test_engine.setProperty('volume', 1.0)
                print("TTS 初始化成功")
            except Exception as e:
                print(f"TTS 初始化失敗: {e}")
                self.tts_engine = None
        
        self.voice_active = False
        if en_spc and sr:
            self.start_voice_recognition()

    def start_voice_recognition(self):
        def voice_thread():
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = False
            
            try:
                mic = sr.Microphone()
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=2)
                print("Voice recognition active")
                self.voice_active = True
                
                while True:
                    try:
                        with mic as source:
                            audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                        
                        def process_audio():
                            try:
                                txt = recognizer.recognize_google(audio, language="zh-TW")
                                print(f"Heard: {txt}")
                                keywords = ["救命", "help", "停", "不要", "救我", "打架"]
                                if any(kw in txt.lower() for kw in keywords):
                                    with self.frame_lock:
                                        current_frame = self.latest_frame.copy() if self.latest_frame is not None else None
                                    self._on_emergency("voice", txt, current_frame)
                            except sr.UnknownValueError:
                                pass
                            except Exception as e:
                                pass
                        
                        threading.Thread(target=process_audio, daemon=True).start()
                        
                    except sr.WaitTimeoutError:
                        pass
                    except Exception:
                        time.sleep(0.5)
                        
            except Exception as e:
                print(f"Mic error: {e}")
                self.voice_active = False
        
        threading.Thread(target=voice_thread, daemon=True).start()

    @staticmethod
    def _dist(p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def _track_persons(self, centers):
        current_ids = {}
        threshold = 100  
        
        for idx, center in enumerate(centers):
            min_dist = float('inf')
            best_id = None
            
            for pid, prev_center in self.person_tracking.items():
                dist = self._dist(center, prev_center)
                if dist < min_dist and dist < threshold:
                    min_dist = dist
                    best_id = pid

            if best_id is not None:
                current_ids[idx] = best_id
            else:
                current_ids[idx] = self.next_person_id
                self.next_person_id += 1
        
        new_tracking = {}
        for idx, pid in current_ids.items():
            new_tracking[pid] = centers[idx]
        self.person_tracking = new_tracking
        
        return current_ids
    
    def _detect_contact(self, kpts1, kpts2):
        if kpts1.size == 0 or kpts2.size == 0:
            return False
        
        arm_joints = [7, 8, 9, 10]
        body_joints = [5, 6, 11, 12]  
        
        for arm in arm_joints:
            if arm < len(kpts1) and kpts1[arm][2] > 0.5:
                for body in body_joints:
                    if body < len(kpts2) and kpts2[body][2] > 0.5:
                        d = self._dist(kpts1[arm][:2], kpts2[body][:2])
                        if d < 50:
                            return True
   
        for arm in arm_joints:
            if arm < len(kpts2) and kpts2[arm][2] > 0.5:
                for body in body_joints:
                    if body < len(kpts1) and kpts1[body][2] > 0.5:
                        d = self._dist(kpts2[arm][:2], kpts1[body][:2])
                        if d < 50:
                            return True
        
        return False
    
    def _update_contact_tracker(self, person_ids, kpts_list, current_time):
        fight_pairs = []
        to_remove = []
        for pair_key in self.contact_tracker:
            if current_time - self.contact_tracker[pair_key]['last_contact'] > self.contact_timeout:
                to_remove.append(pair_key)
        for key in to_remove:
            del self.contact_tracker[key]
        
        for i in range(len(kpts_list)):
            for j in range(i + 1, len(kpts_list)):
                if self._detect_contact(kpts_list[i], kpts_list[j]):
                    id1, id2 = person_ids[i], person_ids[j]
                    pair_key = tuple(sorted([id1, id2]))
                    
                    if pair_key in self.contact_tracker:
                        tracker = self.contact_tracker[pair_key]
                        time_since_first = current_time - tracker['first_contact']
                        
                        if time_since_first <= self.contact_timeout:
                            tracker['count'] += 1
                            tracker['last_contact'] = current_time
                            if tracker['count'] >= self.min_contacts_for_fight:
                                fight_pairs.append(pair_key)
                        else:
                            tracker['first_contact'] = current_time
                            tracker['last_contact'] = current_time
                            tracker['count'] = 1
                    else:
                        self.contact_tracker[pair_key] = {
                            'first_contact': current_time,
                            'last_contact': current_time,
                            'count': 1
                        }
        
        return len(fight_pairs) > 0, fight_pairs

    def _analyze_pose_interaction(self, kpts1, kpts2=None):
        if kpts1 is None or kpts1.size == 0:
            return 0.0
        if kpts2 is not None and kpts2.size == 0:
            return 0.0
        
        score = 0.0
        
        kpts_list = [kpts1]
        if kpts2 is not None:
            kpts_list.append(kpts2)
            
        for kpts in kpts_list:
            wrists = [9, 10]
            shoulders = [5, 6]
            
            for wrist in wrists:
                if wrist < len(kpts) and kpts[wrist][2] > 0.5:
                    for shoulder in shoulders:
                        if shoulder < len(kpts) and kpts[shoulder][2] > 0.5:
                            if kpts[wrist][1] < kpts[shoulder][1] - 30: 
                                score += 1.5
        return score

    def _compute_aggression_scores(self, kpts_list, centers, dt, person_ids, has_fight_contact):
        scores = [0.0] * len(kpts_list)
        
        if len(kpts_list) < 2:
            return scores
        
        if has_fight_contact:
            for i in range(len(kpts_list)):
                scores[i] += 5.0

        prev_kpts = self.kpts_history[-1] if len(self.kpts_history) > 0 else None
        prev_centers = self.centers_history[-1] if len(self.centers_history) > 0 else None
        
        for idx, kpts in enumerate(kpts_list):
            s = scores[idx]
            s += self._analyze_pose_interaction(kpts)
            if prev_centers and idx < len(prev_centers):
                center_speed = self._dist(centers[idx], prev_centers[idx]) / max(1e-6, dt)
                if center_speed > 300:
                    s += 2.0
                elif center_speed > 150:
                    s += 1.0
            
            scores[idx] = s
        
        return scores

    def detect_frame(self, frame):
        h, w = frame.shape[:2]
        scale = 1.0
        if w > 640:
            scale = 640.0 / w
            frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            frame_resized = frame.copy()
        
        results = self.model(frame_resized, conf=self.confidence_threshold, verbose=False)
        
        fight = False
        persons = 0
        aggression_scores = []
        centers = []
        fight_pairs = []
        
        if len(results) > 0 and getattr(results[0], "keypoints", None) is not None:
            kpts_data = results[0].keypoints.data.cpu().numpy()
            persons = len(kpts_data)
            adjusted = []
            
            for person_kpts in kpts_data:
                k = person_kpts.copy()
                if scale != 1.0:
                    k[:, :2] /= scale
                adjusted.append(k)
                xs = [p[0] for p in k if p[2] > 0]
                ys = [p[1] for p in k if p[2] > 0]
                centers.append((float(np.mean(xs)), float(np.mean(ys))) if xs and ys else (0.0, 0.0))
            
            now = time.time()
            dt = now - (self.prev_time or now)
            self.prev_time = now
            
            if persons >= 2:
                person_ids = self._track_persons(centers)
                has_fight_contact, fight_pairs = self._update_contact_tracker(person_ids, adjusted, now)
                aggression_scores = self._compute_aggression_scores(adjusted, centers, dt, person_ids, has_fight_contact)

                if has_fight_contact:
                    if self.consecutive_detections < 3:
                        self.consecutive_detections += 1
                    if self.consecutive_detections >= 2:
                        fight = True
                else:
                    self.consecutive_detections = max(0, self.consecutive_detections - 1)
            else:
                self.consecutive_detections = 0
            
            self.kpts_history.append(adjusted)
            self.centers_history.append(centers)
        
        return fight, persons, results, scale, aggression_scores, centers

    def draw_keypoints_and_scores(self, frame, results, scale=1.0, aggression_scores=None):
        if results is None or len(results) == 0 or getattr(results[0], "keypoints", None) is None:
            return frame
        
        kpts_data = results[0].keypoints.data.cpu().numpy()
        
        for idx, person_kpts in enumerate(kpts_data):
            adj = person_kpts.copy()
            if adj.size == 0:
                continue
            if scale != 1.0:
                adj[:, :2] /= scale
            
            for i in range(adj.shape[0]):
                x, y, conf = adj[i][0], adj[i][1], adj[i][2]
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
            
            conns = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
            for a, b in conns:
                if a < adj.shape[0] and b < adj.shape[0] and adj[a][2] > 0.5 and adj[b][2] > 0.5:
                    pt1 = (int(adj[a][0]), int(adj[a][1]))
                    pt2 = (int(adj[b][0]), int(adj[b][1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
            
            chest_x = chest_y = None
            if adj.shape[0] > 5 and adj[5][2] > 0.3:
                chest_x = int(adj[5][0])
                chest_y = int(adj[5][1]) - 10
            else:
                xs = [p[0] for p in adj if p[2] > 0]
                ys = [p[1] for p in adj if p[2] > 0]
                if xs and ys:
                    chest_x = int(np.mean(xs))
                    chest_y = int(np.mean(ys)) - 10
            
            if aggression_scores and idx < len(aggression_scores) and chest_x and chest_y:
                score = aggression_scores[idx]
                color = (0, 0, 255) if score >= self.aggression_threshold else (0, 255, 255) if score >= 2.0 else (0, 255, 0)
                cv2.putText(frame, f"Score:{score:.1f}", (chest_x, chest_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw contact tracking info
        info_y = 250
        for pair_key, tracker in self.contact_tracker.items():
            info_text = f"Pair {pair_key}: {tracker['count']} contacts"
            cv2.putText(frame, info_text, (20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            info_y += 20
        
        return frame

    def draw_ui(self, frame, fps, persons):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10,10), (460,220), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        status_color = (0, 0, 255) if self.fight_detected else (0, 255, 0)
        status_text = "FIGHT DETECTED!" if self.fight_detected else "MONITORING"
        cv2.putText(frame, "Fight Detection System", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, f"Status: {status_text}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Persons: {persons}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Total Alerts: {self.total_alerts}", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"Voice: {'ON' if self.voice_active else 'OFF'}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if self.voice_active else (0,0,255), 2)
        runtime = int(time.time() - self.session_start_time)
        cv2.putText(frame, f"Runtime: {runtime}s", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if self.fight_detected:
            cv2.rectangle(frame, (w-180, 10), (w-10, 80), (0,0,255), -1)
            cv2.putText(frame, "ALERT!", (w-140, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        return frame
    
    def _play_tts(self, msg):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)
            engine.say(msg)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"TTS 播放錯誤: {e}")

    def _on_emergency(self, source_type="vision", text=None, frame=None):
        now = time.time()
        
        self.last_alert_time = now
        self.total_alerts += 1
        self.fight_detected = True
        self.alert_start_time = now
        
        if source_type == "voice":
            title = "緊急語音警報"
            desc = f"偵測到求救語音: **{text}**\n請立即提供協助!"
            color = 16711680
        else:
            title = "偵測到打架行為"
            desc = f"視覺系統偵測到重複性肢體衝突\n一秒內發生多次接觸，疑似打鬥行為"
            color = 16744192
        
        def send_task():
            img_bytes = None
            if frame is not None:
                if source_type == "voice":
                    alert_frame = frame.copy()
                    _, enc = cv2.imencode(".jpg", alert_frame)
                else:
                    _, enc = cv2.imencode(".jpg", frame)
                img_bytes = enc.tobytes()
            
            success = send_discord_embed(title, desc, color, img_bytes=img_bytes, src_type=source_type)
            if not success:
                print(f"Discord notification failed")
        
        threading.Thread(target=send_task, daemon=True).start()
        
        if pyttsx3:
            try:
                if source_type == "voice":
                    msg = "偵測到緊急求救，已發送警報!"
                else:
                    msg = "偵測到打架！安全警報已觸發!"
                print(f"TTS 播放：{msg}")
                threading.Thread(target=self._play_tts, args=(msg,), daemon=True).start()
            except Exception as e:
                print(f"TTS 播放錯誤: {e}")

    def process_frame_and_update(self, frame):
        with self.frame_lock:
            self.latest_frame = frame.copy()
        
        fight, persons, results, scale, aggression_scores, centers = self.detect_frame(frame)
        
        now = time.time()
        if fight:
                path, bytes_img = save_frame_locally(frame)
                self._on_emergency(source_type="vision", frame=frame)
        else:
            if self.fight_detected and (now - self.alert_start_time > self.alert_duration):
                self.fight_detected = False
        
        frame = self.draw_keypoints_and_scores(frame, results, scale, aggression_scores)
        
        self.frame_count += 1
        if now - self.last_fps_time >= 1.0:
            fps = self.frame_count / (now - self.last_fps_time)
            self.fps_history.append(fps)
            self.frame_count = 0
            self.last_fps_time = now
        
        avg_fps = float(np.mean(self.fps_history)) if self.fps_history else 0.0
        frame = self.draw_ui(frame, avg_fps, persons)
        return frame

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("Camera error")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                processed = self.process_frame_and_update(frame)
                cv2.imshow("Fight Detection System", processed)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    self.total_alerts = 0
                    self.session_start_time = time.time()
                    self.consecutive_detections = 0
                    self.contact_tracker.clear()
                elif key == ord("t"):
                    self._on_emergency("manual", "Manual test alert", frame)
                    
        except KeyboardInterrupt:
            pass
        finally:
            cap.release()
            cv2.destroyAllWindows()
            runtime = time.time() - self.session_start_time
            avg_fps = float(np.mean(self.fps_history)) if self.fps_history else 0.0
            print(f"Session: {runtime:.1f}s, FPS: {avg_fps:.1f}, Alerts: {self.total_alerts}")

def main():
    if flsk_srv:
        threading.Thread(target=lambda: app.run(host="0.0.0.0", port=flsk_prt), daemon=True).start()
    
    detector = FightDetectionSystem(model_path=mdl_pth,
                                    conf_threshold=cnf_thr,
                                    alert_duration=alrt_dur,
                                    aggression_threshold=agr_thr)
    detector.run(source=0)

if __name__ == "__main__":
    main()
