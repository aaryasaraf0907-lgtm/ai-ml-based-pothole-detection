"""
post_repair_verify.py

TransitEye — Post-Repair Verification

Uses the actual YOLO camera pipeline to verify whether a previously
repaired road defect is still visible.

Verification logic:

    1. Read the persistent road defect from Firestore.
    2. Confirm it is in post-repair monitoring.
    3. Get the current GPS location.
    4. Confirm the camera is at the remembered defect location.
    5. Analyse a sequence of frames.
    6. If a pothole is detected -> verification FAILS.
    7. If all required frames are clear -> record a CLEAR PASS.
    8. Three clear passes from independent buses -> RESOLVED.

IMPORTANT:
    --gps defect is a DEMO mode. It uses the remembered defect's GPS
    coordinates while the camera remains real.

    --gps phone uses real GPS2IP coordinates and should be used when
    physically at the repaired location.
"""

import cv2
import argparse
import socket
import time
import threading
import base64
import concurrent.futures

from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO


# ════════════════════════════════════════════════════════════════════════════
# FIREBASE
# ════════════════════════════════════════════════════════════════════════════

cred = credentials.Certificate(
    "serviceAccountKey.json"
)

firebase_admin.initialize_app(cred)

db = firestore.client()

road_defects_col = db.collection(
    "road_defects"
)

verification_col = db.collection(
    "post_repair_verifications"
)


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

VERIFICATION_RADIUS = 30       # metres
DEFAULT_REQUIRED_FRAMES = 30

CLASS_NAMES = {
    0: "minor",
    1: "moderate",
    2: "severe"
}


# ════════════════════════════════════════════════════════════════════════════
# GPS
# ════════════════════════════════════════════════════════════════════════════

PHONE_IP = "10.213.253.81"


class GPSReader:

    def __init__(
        self,
        mode="defect",
        phone_ip=None,
        defect_lat=None,
        defect_lng=None
    ):

        self.mode = mode

        self.phone_ip = phone_ip

        self.lat = defect_lat or 18.5204

        self.lng = defect_lng or 73.8567

        self.valid = True

        self.accuracy = "Demo defect location"

        self._lock = threading.Lock()

        if mode == "phone":

            threading.Thread(
                target=self._read_phone,
                daemon=True
            ).start()


    def _discover_gps2ip(
        self,
        port=11123
    ):

        try:

            s = socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM
            )

            s.connect(
                ("8.8.8.8", 80)
            )

            local_ip = (
                s.getsockname()[0]
            )

            s.close()

            base = (
                ".".join(
                    local_ip.split(".")[:3]
                )
                + "."
            )


            print(
                f"🔍 Scanning "
                f"{base}0/24 "
                f"for GPS2IP..."
            )


            def try_ip(ip):

                try:

                    sock = socket.socket(
                        socket.AF_INET,
                        socket.SOCK_STREAM
                    )

                    sock.settimeout(
                        0.5
                    )

                    sock.connect(
                        (
                            ip,
                            port
                        )
                    )

                    sock.close()

                    return ip

                except:

                    return None


            with concurrent.futures.ThreadPoolExecutor(
                max_workers=60
            ) as ex:

                results = list(
                    ex.map(

                        try_ip,

                        [
                            f"{base}{i}"
                            for i in range(
                                1,
                                255
                            )
                        ]

                    )
                )


            found = [
                ip
                for ip in results
                if ip
            ]


            return (
                found[0]
                if found
                else None
            )


        except Exception as e:

            print(
                f"⚠️ GPS discovery error: {e}"
            )

            return None


    def _read_phone(self):

        PORT = 11123

        while True:

            try:

                ip = (
                    self.phone_ip
                    or
                    self._discover_gps2ip(
                        PORT
                    )
                )


                if not ip:

                    print(
                        "❌ GPS2IP not found."
                    )

                    time.sleep(5)

                    continue


                sock = socket.socket(
                    socket.AF_INET,
                    socket.SOCK_STREAM
                )

                sock.settimeout(5)

                sock.connect(
                    (
                        ip,
                        PORT
                    )
                )


                print(
                    f"✅ GPS2IP connected: "
                    f"{ip}:{PORT}"
                )


                buf = ""


                while True:

                    data = (
                        sock.recv(1024)
                        .decode(
                            "ascii",
                            errors="replace"
                        )
                    )


                    buf += data

                    lines = buf.split(
                        "\n"
                    )

                    buf = lines[-1]


                    for line in lines[:-1]:

                        self._parse_nmea(
                            line.strip()
                        )


            except Exception as e:

                print(
                    f"⚠️ GPS disconnected: "
                    f"{e}"
                )

                time.sleep(3)


    def _parse_nmea(
        self,
        line
    ):

        try:

            if not (
                line.startswith("$GPRMC")
                or
                line.startswith("$GNRMC")
            ):

                return


            parts = line.split(",")


            if (
                len(parts) < 7
                or
                parts[2] != "A"
            ):

                return


            lat = (
                float(parts[3][:2])
                +
                float(parts[3][2:]) / 60
            )


            lng = (
                float(parts[5][:3])
                +
                float(parts[5][3:]) / 60
            )


            if parts[4] == "S":

                lat = -lat


            if parts[6] == "W":

                lng = -lng


            with self._lock:

                self.lat = lat

                self.lng = lng

                self.valid = True

                self.accuracy = (
                    "3-5m (GPS)"
                )


        except Exception:

            pass


    def get(self):

        with self._lock:

            return (
                self.lat,
                self.lng,
                self.valid,
                self.accuracy
            )


# ════════════════════════════════════════════════════════════════════════════
# DISTANCE
# ════════════════════════════════════════════════════════════════════════════

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371000

    phi1 = radians(lat1)
    phi2 = radians(lat2)

    dphi = radians(
        lat2 - lat1
    )

    dlambda = radians(
        lon2 - lon1
    )

    a = (
        sin(dphi / 2) ** 2
        +
        cos(phi1)
        *
        cos(phi2)
        *
        sin(dlambda / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return earth_radius * c


# ════════════════════════════════════════════════════════════════════════════
# IMAGE
# ════════════════════════════════════════════════════════════════════════════

def frame_to_base64(
    frame,
    quality=60
):

    if (
        frame is None
        or
        frame.size == 0
    ):

        return None


    h, w = frame.shape[:2]

    max_dim = 640

    scale = (
        max_dim
        /
        max(h, w)
    )


    if scale < 1:

        frame = cv2.resize(

            frame,

            (
                int(w * scale),
                int(h * scale)
            )

        )


    ok, buffer = cv2.imencode(

        ".jpg",

        frame,

        [
            cv2.IMWRITE_JPEG_QUALITY,
            quality
        ]

    )


    if not ok:

        return None


    return (
        "data:image/jpeg;base64,"
        +
        base64.b64encode(
            buffer.tobytes()
        ).decode("ascii")
    )


# ════════════════════════════════════════════════════════════════════════════
# VERIFY ROAD LOCATION
# ════════════════════════════════════════════════════════════════════════════

def check_location(
    current_lat,
    current_lng,
    defect_lat,
    defect_lng
):

    distance = haversine_distance(

        current_lat,
        current_lng,

        defect_lat,
        defect_lng

    )

    return (
        distance <= VERIFICATION_RADIUS,
        distance
    )


# ════════════════════════════════════════════════════════════════════════════
# RECORD FAILED VERIFICATION
# ════════════════════════════════════════════════════════════════════════════

def record_failed_verification(
    defect_id,
    bus_id,
    route_id,
    camera_id,
    lat,
    lng,
    detected_count,
    image_data,
):

    verification_col.add({

        "road_defect_id":
            defect_id,

        "bus_id":
            bus_id,

        "route_id":
            route_id,

        "camera_id":
            camera_id,

        "timestamp":
            firestore.SERVER_TIMESTAMP,

        "pothole_detected":
            True,

        "detection_count":
            detected_count,

        "verification_result":
            "failed",

        "lat":
            lat,

        "lng":
            lng,

        "image_data":
            image_data,

    })


# ════════════════════════════════════════════════════════════════════════════
# RECORD SUCCESSFUL VERIFICATION
# ════════════════════════════════════════════════════════════════════════════

def record_successful_verification(

    defect_id,

    bus_id,

    route_id,

    camera_id,

    lat,

    lng,

    image_data,

):

    # Get current defect state
    ref = road_defects_col.document(
        defect_id
    )

    doc = ref.get()

    if not doc.exists:

        raise RuntimeError(
            f"Road defect {defect_id} no longer exists."
        )


    data = doc.to_dict()


    verified_buses = list(
        data.get(
            "verification_bus_ids",
            []
        )
    )


    # Don't count the same bus twice.
    if bus_id not in verified_buses:

        verified_buses.append(
            bus_id
        )


    verification_passes = len(
        verified_buses
    )


    if verification_passes >= 3:

        verification_status = (
            "verified"
        )

        maintenance_status = (
            "resolved"
        )

    else:

        verification_status = (
            "in_progress"
        )

        maintenance_status = (
            "post_repair_monitoring"
        )


    verification_col.add({

        "road_defect_id":
            defect_id,

        "bus_id":
            bus_id,

        "route_id":
            route_id,

        "camera_id":
            camera_id,

        "timestamp":
            firestore.SERVER_TIMESTAMP,

        "pothole_detected":
            False,

        "detection_count":
            0,

        "verification_result":
            "clear",

        "lat":
            lat,

        "lng":
            lng,

        "image_data":
            image_data,

    })


    updates = {

        "verification_passes":
            verification_passes,

        "verification_bus_ids":
            verified_buses,

        "verification_status":
            verification_status,

        "maintenance_status":
            maintenance_status,

        "last_verification_at":
            firestore.SERVER_TIMESTAMP,

        "last_verification_bus":
            bus_id,

    }


    if verification_passes >= 3:

        updates[
            "post_repair_verified_at"
        ] = firestore.SERVER_TIMESTAMP

        updates[
            "resolved_at"
        ] = firestore.SERVER_TIMESTAMP


    ref.update(
        updates
    )


    return (
        verification_passes,
        verification_status,
        maintenance_status
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():

    parser = argparse.ArgumentParser(

        description=(
            "TransitEye "
            "Post-Repair Verification"
        )

    )


    parser.add_argument(
        "--road-defect-id",
        required=True
    )


    parser.add_argument(
        "--model",
        default=(
            "runs/"
            "pothole_severity_v2/"
            "weights/"
            "best.pt"
        )
    )


    parser.add_argument(
        "--gps",
        choices=[
            "defect",
            "phone"
        ],
        default="defect"
    )


    parser.add_argument(
        "--phone-ip",
        default=PHONE_IP
    )


    parser.add_argument(
        "--bus-id",
        required=True
    )


    parser.add_argument(
        "--route-id",
        required=True
    )


    parser.add_argument(
        "--camera-id",
        default="FRONT-01"
    )


    parser.add_argument(
        "--camera",
        type=int,
        default=0
    )


    parser.add_argument(
        "--conf",
        type=float,
        default=0.55
    )


    parser.add_argument(
        "--required-frames",
        type=int,
        default=DEFAULT_REQUIRED_FRAMES
    )


    args = parser.parse_args()


    # ═══════════════════════════════════════════════════════════════════════
    # LOAD DEFECT
    # ═══════════════════════════════════════════════════════════════════════

    defect_ref = (
        road_defects_col
        .document(
            args.road_defect_id
        )
    )


    defect_doc = (
        defect_ref.get()
    )


    if not defect_doc.exists:

        print(
            f"❌ Road defect "
            f"{args.road_defect_id} "
            f"not found."
        )

        return


    defect = (
        defect_doc.to_dict()
    )


    maintenance_status = (
        defect.get(
            "maintenance_status"
        )
    )


    if maintenance_status != (
        "post_repair_monitoring"
    ):

        print(
            "\n❌ This road defect is "
            "not in post-repair monitoring."
        )

        print(
            f"Current status: "
            f"{maintenance_status}"
        )

        print(
            "\nRun:"
        )

        print(
            f"python "
            f"mark_repair_completed.py "
            f"{args.road_defect_id}"
        )

        return


    defect_lat = float(
        defect["lat"]
    )

    defect_lng = float(
        defect["lng"]
    )


    # ═══════════════════════════════════════════════════════════════════════
    # START
    # ═══════════════════════════════════════════════════════════════════════

    print("\n")

    print("=" * 70)

    print(
        "🔍 TRANSITEYE — POST-REPAIR VERIFICATION"
    )

    print("=" * 70)

    print(
        f"\nRoad Defect: "
        f"{args.road_defect_id}"
    )

    print(
        f"Repair status: "
        f"{maintenance_status}"
    )

    print(
        f"Defect GPS: "
        f"{defect_lat:.6f}, "
        f"{defect_lng:.6f}"
    )

    print(
        f"Bus: "
        f"{args.bus_id}"
    )

    print(
        f"Route: "
        f"{args.route_id}"
    )

    print(
        f"Required clear frames: "
        f"{args.required_frames}"
    )


    if args.gps == "defect":

        print(
            "\n📍 GPS mode: DEMO"
        )

        print(
            "   Using the remembered RD location."
        )

        print(
            "   Camera verification remains REAL."
        )

    else:

        print(
            "\n📍 GPS mode: PHONE"
        )

        print(
            f"   iPhone IP: "
            f"{args.phone_ip}"
        )


    # ═══════════════════════════════════════════════════════════════════════
    # GPS
    # ═══════════════════════════════════════════════════════════════════════

    gps = GPSReader(

        mode=args.gps,

        phone_ip=args.phone_ip,

        defect_lat=defect_lat,

        defect_lng=defect_lng

    )


    if args.gps == "phone":

        time.sleep(2)


    # ═══════════════════════════════════════════════════════════════════════
    # CAMERA
    # ═══════════════════════════════════════════════════════════════════════

    print(
        "\n📷 Opening camera..."
    )


    cap = cv2.VideoCapture(
        args.camera,
        cv2.CAP_AVFOUNDATION
    )


    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        30
    )


    if not cap.isOpened():

        print(
            "❌ Cannot open camera."
        )

        return


    for _ in range(5):

        cap.read()


    model = YOLO(
        args.model
    )


    print(
        "✅ Model loaded"
    )


    # ═══════════════════════════════════════════════════════════════════════
    # LOCATION CHECK
    # ═══════════════════════════════════════════════════════════════════════

    lat, lng, gps_valid, accuracy = (
        gps.get()
    )


    location_ok, distance = (
        check_location(

            lat,

            lng,

            defect_lat,

            defect_lng

        )
    )


    print(
        f"\n📍 Current GPS: "
        f"{lat:.6f}, {lng:.6f}"
    )

    print(
        f"📏 Distance from RD location: "
        f"{distance:.1f}m"
    )


    if not location_ok:

        print(
            "\n❌ Verification stopped."
        )

        print(
            f"Camera/GPS is more than "
            f"{VERIFICATION_RADIUS}m "
            f"from {args.road_defect_id}."
        )

        cap.release()

        return


    print(
        "\n✅ Correct road location."
    )


    print(
        "\n🎥 SHOW THE CAMERA THE REPAIRED / CLEAR ROAD."
    )

    print(
        "   Keep the view stable during verification."
    )

    print(
        "   Press Q to cancel."
    )


    # ═══════════════════════════════════════════════════════════════════════
    # VERIFICATION LOOP
    # ═══════════════════════════════════════════════════════════════════════

    analysed_frames = 0

    clear_frames = 0

    pothole_detections = 0

    last_clear_frame = None

    failed = False


    while clear_frames < args.required_frames:

        ret, frame = cap.read()


        if not ret:

            continue


        # Analyse every third frame.
        if analysed_frames % 3 == 0:

            results = model(

                frame,

                conf=args.conf,

                iou=0.30,

                agnostic_nms=True,

                max_det=50,

                verbose=False

            )[0]


            detected = (
                len(
                    results.boxes
                )
                >
                0
            )


            if detected:

                pothole_detections += 1

                failed = True

                # Evidence frame
                failed_image = (
                    frame.copy()
                )

                print(
                    "\n❌ VERIFICATION FAILED"
                )

                print(
                    "   A pothole was detected "
                    "during post-repair monitoring."
                )

                break


            clear_frames += 1

            last_clear_frame = (
                frame.copy()
            )


        # ────────────────────────────────────────────────────────────────
        # CAMERA OVERLAY
        # ────────────────────────────────────────────────────────────────

        display = frame.copy()


        cv2.rectangle(

            display,

            (0, 0),

            (520, 100),

            (0, 0, 0),

            -1

        )


        cv2.putText(

            display,

            "TRANSITEYE — POST-REPAIR CHECK",

            (12, 25),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            display,

            f"Defect: {args.road_defect_id}",

            (12, 48),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            (180, 180, 180),

            1

        )


        cv2.putText(

            display,

            f"CLEAR: "
            f"{clear_frames}/"
            f"{args.required_frames}",

            (12, 72),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.50,

            (48, 209, 88),

            2

        )


        cv2.putText(

            display,

            f"Bus: {args.bus_id}",

            (12, 92),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.38,

            (180, 180, 180),

            1

        )


        cv2.imshow(

            "TransitEye — Post-Repair Verification",

            display

        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key in [
            ord("q"),
            27
        ]:

            print(
                "\n⚠️ Verification cancelled."
            )

            cap.release()

            cv2.destroyAllWindows()

            return


        analysed_frames += 1


    # ═══════════════════════════════════════════════════════════════════════
    # RESULT
    # ═══════════════════════════════════════════════════════════════════════

    if failed:

        image_data = (
            frame_to_base64(
                failed_image
            )
        )


        lat, lng, _, _ = (
            gps.get()
        )


        record_failed_verification(

            args.road_defect_id,

            args.bus_id,

            args.route_id,

            args.camera_id,

            lat,

            lng,

            pothole_detections,

            image_data,

        )


        defect_ref.update({

            "verification_status":
                "failed",

            "last_verification_at":
                firestore.SERVER_TIMESTAMP,

            "last_verification_bus":
                args.bus_id,

        })


        cap.release()

        cv2.destroyAllWindows()

        print(
    f"\n❌ {args.road_defect_id} remains "
    "in post-repair monitoring."
)

        return


    # ═══════════════════════════════════════════════════════════════════════
    # CLEAR PASS
    # ═══════════════════════════════════════════════════════════════════════

    image_data = (
        frame_to_base64(
            last_clear_frame
        )
    )


    lat, lng, _, _ = (
        gps.get()
    )


    (
        verification_passes,
        verification_status,
        maintenance_status
    ) = record_successful_verification(

        args.road_defect_id,

        args.bus_id,

        args.route_id,

        args.camera_id,

        lat,

        lng,

        image_data,

    )


    cap.release()

    cv2.destroyAllWindows()


    print("\n")

    print("=" * 70)

    print(
        "✅ CLEAR POST-REPAIR PASS"
    )

    print("=" * 70)

    print(
        f"Road Defect: "
        f"{args.road_defect_id}"
    )

    print(
        f"Bus: "
        f"{args.bus_id}"
    )

    print(
        f"Clear frames: "
        f"{clear_frames}"
    )

    print(
        f"Verification passes: "
        f"{verification_passes}/3"
    )

    print(
        f"Verification status: "
        f"{verification_status.upper()}"
    )

    print(
        f"Maintenance status: "
        f"{maintenance_status.upper()}"
    )


    if maintenance_status == "resolved":

        print(
            "\n🎉 POST-REPAIR VERIFICATION COMPLETE"
        )

        print(f"{args.road_defect_id} is now RESOLVED.")

    else:

        print(
            "\n✅ One clear verification pass recorded."
        )

        print(
            "More independent clear passes are required."
        )


    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()