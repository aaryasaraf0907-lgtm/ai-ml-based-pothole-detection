"""
detect_live.py — PotholeWatch Demo Script (image stored as Base64 in Firestore)
No Firebase Storage / Blaze plan required — fully free tier.

MacBook M1 Pro + Logitech C270 + iPhone GPS2IP + Firebase Firestore

MOCK GPS (indoor testing):
    python detect_live.py --gps mock --camera 0

REAL DEMO (riding scooter):
    python detect_live.py --gps phone --phone-ip <IP from GPS2IP app>

Route Memory / Fleet Consensus:
    python detect_live.py --gps phone --phone-ip <IP> --bus-id BUS-017 --route-id PUNE-R12 --camera-id FRONT-01
"""

import cv2
import argparse
import socket
import time
import threading
import base64
import concurrent.futures
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO

# ════════════════════════════════════════════════════════════════════════════
# 📍 EDIT THIS BEFORE EVERY RIDE — open GPS2IP Lite on iPhone, copy the
#    "iPhone Server IP" shown on screen, paste it here.
# ════════════════════════════════════════════════════════════════════════════
PHONE_IP = "10.213.253.81"
# ════════════════════════════════════════════════════════════════════════════

# ── Route Memory / Fleet metadata ────────────────────────────────────────────
# These identify the simulated onboard unit. They can also be supplied from
# the command line so multiple demo buses can share the same Firestore project.
BUS_ID = "BUS-017"
ROUTE_ID = "PUNE-R12"
CAMERA_ID = "FRONT-01"
DEFECT_TYPE = "pothole"

# Two pothole observations are treated as the same persistent road defect when
# their GPS locations are within this distance.
ROAD_MATCH_RADIUS = 30  # metres

# ── Firebase (Firestore only — no Storage needed) ────────────────────────────
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)
db  = firestore.client()
col = db.collection("detections")
road_defects_col = db.collection("road_defects")
road_defect_counter = db.collection("counters").document("road_defects")
print("✅ Firebase connected (Firestore)")

# ── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "minor", 1: "moderate", 2: "severe"}
COLORS = {
    "minor":    (48,  209, 88),
    "moderate": (0,   149, 255),
    "severe":   (59,  59,  255),
}
RANK = {"minor": 0, "moderate": 1, "severe": 2}

MAX_IMG_DIM = 280     # resize crop so longest side is this many px
JPEG_QUALITY = 60      # keeps base64 size well under Firestore's 1MB doc limit


# ── Route Memory helpers ──────────────────────────────────────────────────────
def haversine_distance(lat1, lon1, lat2, lon2):
    """Return the distance between two GPS coordinates in metres."""
    earth_radius = 6371000

    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = (
        sin(dphi / 2) ** 2
        + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return earth_radius * c


def find_matching_road_defect(lat, lng, defect_type=DEFECT_TYPE):
    """
    Find the closest remembered road defect of the same type within
    ROAD_MATCH_RADIUS metres.

    This is intentionally simple for the prototype. A production version
    can replace this scan with geohash/road-network map matching.
    """
    try:
        nearest_doc = None
        nearest_distance = float("inf")

        for doc in road_defects_col.stream():
            data = doc.to_dict()

            if data.get("defect_type") != defect_type:
                continue

            defect_lat = data.get("lat")
            defect_lng = data.get("lng")

            if defect_lat is None or defect_lng is None:
                continue

            distance = haversine_distance(
                lat,
                lng,
                float(defect_lat),
                float(defect_lng),
            )

            if distance <= ROAD_MATCH_RADIUS and distance < nearest_distance:
                nearest_doc = doc
                nearest_distance = distance

        return nearest_doc, nearest_distance if nearest_doc else None

    except Exception as e:
        print(f"⚠️  Road memory lookup failed: {e}")
        return None, None


def get_next_road_defect_id():
    """Return the next human-readable road-defect ID: RD-001, RD-002, ...

    A Firestore transaction is used so the counter remains safe if more than
    one detection process creates a new road defect at the same time.
    """
    transaction = db.transaction()

    @firestore.transactional
    def increment_counter(transaction):
        snapshot = road_defect_counter.get(transaction=transaction)

        if snapshot.exists:
            current = int(snapshot.to_dict().get("next_id", 1))
        else:
            current = 1

        transaction.set(
            road_defect_counter,
            {"next_id": current + 1},
            merge=True,
        )

        return current

    number = increment_counter(transaction)
    return f"RD-{number:03d}"


def get_consensus_status(unique_bus_count, observation_count):
    """Translate observation history into a simple fleet-consensus label."""
    if unique_bus_count >= 3:
        return "verified"
    if unique_bus_count >= 2:
        return "confirmed"
    if observation_count >= 2:
        return "repeated"
    return "reported"


def update_road_defect(
    severity,
    confidence,
    lat,
    lng,
    bus_id,
    route_id,
    camera_id,
    detection_id,
):
    """
    Merge a new raw observation into an existing remembered road defect,
    or create a new road_defects document.

    Returns:
        road_defect_id, created, observation_count, unique_bus_count,
        consensus_status
    """
    existing_doc, distance = find_matching_road_defect(lat, lng)
    now = firestore.SERVER_TIMESTAMP

    # ────────────────────────────────────────────────────────────────────────
    # CASE 1: Existing road defect
    # ────────────────────────────────────────────────────────────────────────
    if existing_doc:
        ref = existing_doc.reference
        data = existing_doc.to_dict()

        observation_ids = list(data.get("observation_ids", []))

        # Protect against accidentally counting the same raw detection twice.
        if detection_id in observation_ids:
            observation_count = int(data.get("observation_count", 1))
            bus_ids = list(data.get("bus_ids", []))
            unique_bus_count = len(set(bus_ids))
            consensus_status = get_consensus_status(
                unique_bus_count,
                observation_count,
            )
            return (
                existing_doc.id,
                False,
                observation_count,
                unique_bus_count,
                consensus_status,
            )

        observation_count = int(data.get("observation_count", 0)) + 1

        bus_ids = set(data.get("bus_ids", []))
        bus_ids.add(bus_id)
        bus_ids = sorted(bus_ids)

        route_ids = set(data.get("route_ids", []))
        route_ids.add(route_id)
        route_ids = sorted(route_ids)

        old_severity = data.get("severity", "minor")
        current_severity = (
            severity
            if RANK.get(severity, 0) > RANK.get(old_severity, 0)
            else old_severity
        )

        old_conf_sum = float(data.get("confidence_sum", 0.0))
        confidence_sum = old_conf_sum + float(confidence)
        average_confidence = confidence_sum / observation_count

        old_lat = float(data.get("lat", lat))
        old_lng = float(data.get("lng", lng))

        # Running centroid keeps the remembered point stable even when GPS
        # readings vary slightly between passes.
        centre_lat = (
            old_lat * (observation_count - 1) + float(lat)
        ) / observation_count
        centre_lng = (
            old_lng * (observation_count - 1) + float(lng)
        ) / observation_count

        severity_history = list(data.get("severity_history", []))
        severity_history.append(severity)
        severity_history = severity_history[-50:]

        observation_ids.append(detection_id)
        observation_ids = observation_ids[-100:]

        unique_bus_count = len(bus_ids)
        consensus_status = get_consensus_status(
            unique_bus_count,
            observation_count,
        )

        ref.update({
            "severity": current_severity,
            "lat": round(centre_lat, 6),
            "lng": round(centre_lng, 6),
            "observation_count": observation_count,
            "unique_bus_count": unique_bus_count,
            "bus_ids": bus_ids,
            "route_ids": route_ids,
            "unique_route_count": len(route_ids),
            "confidence_sum": confidence_sum,
            "average_confidence": round(average_confidence, 4),
            "last_seen": now,
            "last_bus_id": bus_id,
            "last_route_id": route_id,
            "last_camera_id": camera_id,
            "last_detection_id": detection_id,
            "consensus_status": consensus_status,
            "observation_ids": observation_ids,
            "severity_history": severity_history,
            "status": "unresolved",
        })

        print(
            f"   🧠 Road Memory: MATCHED {existing_doc.id} "
            f"({distance:.1f}m away)"
        )

        return (
            existing_doc.id,
            False,
            observation_count,
            unique_bus_count,
            consensus_status,
        )

    # ────────────────────────────────────────────────────────────────────────
    # CASE 2: New road defect
    # ────────────────────────────────────────────────────────────────────────
    road_defect_id = get_next_road_defect_id()
    road_ref = road_defects_col.document(road_defect_id)

    road_ref.set({
        "defect_type": DEFECT_TYPE,
        "severity": severity,
        "lat": round(float(lat), 6),
        "lng": round(float(lng), 6),

        "observation_count": 1,
        "unique_bus_count": 1,
        "bus_ids": [bus_id],

        "route_ids": [route_id],
        "unique_route_count": 1,

        "confidence_sum": float(confidence),
        "average_confidence": round(float(confidence), 4),

        "first_seen": now,
        "last_seen": now,

        "first_bus_id": bus_id,
        "last_bus_id": bus_id,
        "first_route_id": route_id,
        "last_route_id": route_id,
        "last_camera_id": camera_id,

        "first_detection_id": detection_id,
        "last_detection_id": detection_id,
        "observation_ids": [detection_id],

        "severity_history": [severity],

        "consensus_status": "reported",
        "status": "unresolved",
    })

    print(f"   🧠 Road Memory: NEW {road_defect_id}")

    return road_defect_id, True, 1, 1, "reported"


# ── GPS Reader ───────────────────────────────────────────────────────────────
class GPSReader:
    def __init__(self, mode="mock", phone_ip=None):
        self.mode     = mode
        self.phone_ip = phone_ip
        self.lat      = 18.5204
        self.lng      = 73.8567
        self.valid    = False
        self.accuracy = "~500m (mock)"
        self._lock    = threading.Lock()

        if mode == "phone":
            threading.Thread(target=self._read_phone, daemon=True).start()
        else:
            threading.Thread(target=self._mock_gps, daemon=True).start()

    def _mock_gps(self):
        import random
        print("📡 Mock GPS — simulating Pune coordinates")
        while True:
            with self._lock:
                self.lat      = 18.5204 + random.uniform(-0.008, 0.008)
                self.lng      = 73.8567 + random.uniform(-0.008, 0.008)
                self.valid    = True
                self.accuracy = "~500m (mock)"
            time.sleep(2)

    def _discover_gps2ip(self, port=11123):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            base = ".".join(local_ip.split(".")[:3]) + "."
            print(f"🔍 Scanning {base}0/24 for GPS2IP...")

            def try_ip(ip):
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.5)
                    sock.connect((ip, port))
                    sock.close()
                    return ip
                except:
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=60) as ex:
                results = list(ex.map(try_ip, [f"{base}{i}" for i in range(1, 255)]))
            found = [ip for ip in results if ip]
            return found[0] if found else None
        except Exception as e:
            print(f"⚠️  Discovery error: {e}")
            return None

    def _read_phone(self):
        PORT = 11123
        while True:
            try:
                ip = self.phone_ip or self._discover_gps2ip(PORT)
                if not ip:
                    print("❌ GPS2IP not found — check iPhone hotspot + app")
                    time.sleep(5)
                    continue
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((ip, PORT))
                print(f"✅ GPS2IP connected: {ip}:{PORT}")
                buf = ""
                while True:
                    data = sock.recv(1024).decode("ascii", errors="replace")
                    buf += data
                    lines = buf.split("\n")
                    buf   = lines[-1]
                    for line in lines[:-1]:
                        self._parse_nmea(line.strip())
            except Exception as e:
                print(f"⚠️  GPS disconnected: {e} — reconnecting in 3s...")
                time.sleep(3)

    def _parse_nmea(self, line):
        try:
            if not (line.startswith("$GPRMC") or line.startswith("$GNRMC")):
                return
            parts = line.split(",")
            if len(parts) < 7 or parts[2] != "A":
                return
            lat = float(parts[3][:2]) + float(parts[3][2:]) / 60
            lng = float(parts[5][:3]) + float(parts[5][3:]) / 60
            if parts[4] == "S": lat = -lat
            if parts[6] == "W": lng = -lng
            with self._lock:
                self.lat, self.lng, self.valid = lat, lng, True
                self.accuracy = "3-5m (GPS)"
        except Exception:
            pass

    def get(self):
        with self._lock:
            return self.lat, self.lng, self.valid, self.accuracy


def is_skin_like(frame, box, skin_ratio_thresh=0.35):
    x1, y1, x2, y2 = box
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return False
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    y_ch, cr, cb = cv2.split(ycrcb)
    mask = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135) & (y_ch >= 60)
    skin_ratio = np.count_nonzero(mask) / mask.size
    return skin_ratio >= skin_ratio_thresh


# ── Overlap filter ───────────────────────────────────────────────────────────
def filter_overlapping(boxes, frame, min_dist=80):
    parsed = []
    for box in boxes:
        cls_id   = int(box.cls[0])
        conf     = float(box.conf[0])
        severity = CLASS_NAMES.get(cls_id, "minor")
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if is_skin_like(frame, (x1, y1, x2, y2)):
            continue

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        parsed.append({
            "severity": severity, "conf": conf, "rank": RANK[severity],
            "box": (x1, y1, x2, y2), "centre": (cx, cy),
        })
    parsed.sort(key=lambda d: (d["rank"], d["conf"]), reverse=True)
    kept = []
    for det in parsed:
        cx, cy = det["centre"]
        too_close = any(
            abs(cx - k["centre"][0]) < min_dist and abs(cy - k["centre"][1]) < min_dist
            for k in kept
        )
        if not too_close:
            kept.append(det)
    return kept


def crop_to_base64(crop):
    """Resize + compress crop, return a data URI string ready for <img src=...>"""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    scale = MAX_IMG_DIM / max(h, w)
    if scale < 1:
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)))
    ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ── Firebase upload (Firestore only, image embedded as base64) ───────────────
def upload(
    severity,
    confidence,
    lat,
    lng,
    accuracy,
    image_crop,
    bus_id,
    route_id,
    camera_id,
):
    image_data = crop_to_base64(image_crop)

    try:
        # First write the raw observation exactly as before, with the new
        # vehicle/route metadata added.
        _, detection_ref = col.add({
            "severity":    severity,
            "confidence":  round(float(confidence), 4),
            "lat":         round(lat, 6),
            "lng":         round(lng, 6),
            "accuracy":    accuracy,
            "timestamp":   firestore.SERVER_TIMESTAMP,
            "device":      "scooter-cam-01",
            "bus_id":      bus_id,
            "route_id":    route_id,
            "camera_id":   camera_id,
            "defect_type": DEFECT_TYPE,
            "model":       "YOLOv8s",
            "image_data":  image_data,   # base64 data URI, or None
        })

        # Then update/create the persistent road-defect memory.
        (
            road_defect_id,
            created,
            observation_count,
            bus_count,
            consensus_status,
        ) = update_road_defect(
            severity,
            confidence,
            lat,
            lng,
            bus_id,
            route_id,
            camera_id,
            detection_ref.id,
        )

        # Link the raw observation to the remembered road defect.
        detection_ref.update({
            "road_defect_id": road_defect_id
        })

        tag = "📷" if image_data else "  "
        print(
            f"{tag} 📍 {severity.upper()} ({confidence:.0%}) "
            f"@ {lat:.6f}, {lng:.6f} [{accuracy}]"
        )
        print(f"   🚌 Bus: {bus_id} | Route: {route_id} | Camera: {camera_id}")
        print(f"   🔁 Observations: {observation_count}")
        print(f"   🚌 Unique buses: {bus_count}")
        print(f"   ✅ Fleet consensus: {consensus_status.upper()}")
        print(
            f"   🧠 Road Memory: {road_defect_id} "
            f"({'NEW' if created else 'UPDATED'})"
        )

    except Exception as e:
        print(f"⚠️  Firestore write failed: {e}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PotholeWatch Demo")
    parser.add_argument("--model",    default="runs/pothole_severity_v2/weights/best.pt")
    parser.add_argument("--gps",      default="phone", choices=["mock", "phone"])
    parser.add_argument("--phone-ip", default=PHONE_IP)
    parser.add_argument("--conf",     type=float, default=0.55)
    parser.add_argument("--camera",   type=int,   default=0)
    parser.add_argument("--cooldown", type=float, default=2.0)
    parser.add_argument("--min-dist", type=int,   default=80)
    parser.add_argument("--crop-pad", type=int,   default=25)

    # NEW — fleet identity can be overridden for the demo.
    parser.add_argument("--bus-id",    default=BUS_ID)
    parser.add_argument("--route-id",  default=ROUTE_ID)
    parser.add_argument("--camera-id", default=CAMERA_ID)

    args = parser.parse_args()

    print(f"\n🤖 Loading model: {args.model}")
    model = YOLO(args.model)
    print("✅ Model loaded")

    print(f"\n📡 Starting GPS ({args.gps})...")
    if args.gps == "phone":
        print(f"   Target iPhone IP: {args.phone_ip}")
    gps = GPSReader(mode=args.gps, phone_ip=args.phone_ip)
    if args.gps == "phone":
        time.sleep(2)

    print(f"\n🚌 Fleet Unit: {args.bus_id}")
    print(f"   Route: {args.route_id}")
    print(f"   Camera: {args.camera_id}")

    print(f"\n📷 Opening C270 (camera {args.camera})...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    if not cap.isOpened():
        print(f"❌ Cannot open camera {args.camera}")
        return
    for _ in range(5):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ C270 opened: {w}x{h}")

    last_upload    = {}
    frame_count    = 0
    fps_time       = time.time()
    fps            = 0
    detections     = []
    total_uploaded = 0

    print("\n" + "="*55)
    print("🚀 PotholeWatch DEMO running! (image saved as base64)")
    print("   Q/ESC quit   |   S screenshot")
    print("="*55 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame_count += 1
        if frame_count % 30 == 0:
            fps      = 30 / max(time.time() - fps_time, 0.001)
            fps_time = time.time()

        if frame_count % 3 == 0:
            results = model(frame, conf=args.conf, iou=0.30, agnostic_nms=True,
                            max_det=50, verbose=False)[0]
            lat, lng, gps_valid, accuracy = gps.get()
            detections = filter_overlapping(results.boxes, frame, min_dist=args.min_dist)

            now = time.time()
            for det in detections:
                sev = det["severity"]
                if now - last_upload.get(sev, 0) > args.cooldown:
                    x1, y1, x2, y2 = det["box"]
                    pad = args.crop_pad
                    cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
                    cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
                    crop = frame[cy1:cy2, cx1:cx2].copy()

                    threading.Thread(
                        target=upload,
                        args=(
                            sev,
                            det["conf"],
                            lat,
                            lng,
                            accuracy,
                            crop,
                            args.bus_id,
                            args.route_id,
                            args.camera_id,
                        ),
                        daemon=True,
                    ).start()
                    last_upload[sev] = now
                    total_uploaded  += 1

        for det in detections:
            sev             = det["severity"]
            conf            = det["conf"]
            x1, y1, x2, y2 = det["box"]
            color           = COLORS[sev]
            label           = f"{sev.upper()}  {conf:.0%}"
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), color, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            lw = len(label) * 9 + 10
            cv2.rectangle(frame, (x1, y1-28), (x1+lw, y1), color, -1)
            cv2.putText(frame, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2)
            for px, py in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
                cv2.circle(frame, (px, py), 4, color, -1)

        lat, lng, gps_valid, accuracy = gps.get()
        gps_col = (48, 209, 88) if gps_valid else (59, 59, 255)
        cv2.rectangle(frame, (0, 0), (500, 118), (0, 0, 0), -1)
        cv2.putText(frame, "POTHOLEWATCH  v1.1", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, f"GPS: {lat:.6f}, {lng:.6f}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, gps_col, 1)
        cv2.putText(frame, f"Accuracy: {accuracy}", (12, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.38, gps_col, 1)
        cv2.putText(frame, f"Bus: {args.bus_id}   Route: {args.route_id}", (12, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}   Det: {len(detections)}   Uploaded: {total_uploaded}",
                    (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(frame, ts, (w-240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140,140,140), 1)

        legend = [("MINOR", COLORS["minor"]), ("MODERATE", COLORS["moderate"]), ("SEVERE", COLORS["severe"])]
        x_off = 12
        for name, color in legend:
            cv2.circle(frame, (x_off+6, h-14), 6, color, -1)
            cv2.putText(frame, name, (x_off+18, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
            x_off += 120

        cv2.imshow("PotholeWatch — Live Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        if key == ord('s'):
            fname = f"pothole_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(fname, frame)
            print(f"📸 Saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Demo complete — {total_uploaded} detections uploaded to Firebase.")


if __name__ == "__main__":
    main()
