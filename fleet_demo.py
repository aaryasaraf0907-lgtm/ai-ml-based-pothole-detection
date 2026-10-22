"""
fleet_demo.py

Controlled demonstration of:
    Route Memory
    Fleet Consensus

This script creates multiple observations of the SAME physical
pothole from different simulated buses.

It uses the Route Memory functions already implemented in detect_live.py.
"""

from datetime import datetime, timezone

from detect_live import (
    col,
    update_road_defect,
    get_consensus_status,
    DEFECT_TYPE,
)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO LOCATION
# ─────────────────────────────────────────────────────────────────────────────
# All three observations are deliberately within a few metres of one another.

BASE_LAT = 18.520000
BASE_LNG = 73.870000


def add_demo_observation(
    bus_id,
    route_id,
    camera_id,
    severity,
    confidence,
    lat,
    lng,
):
    """
    Add one simulated bus observation to Firestore
    and pass it through the same Route Memory logic
    used by the real detector.
    """

    captured_at = datetime.now(timezone.utc)

    # ────────────────────────────────────────────────────────────────────────
    # 1. Create the RAW detection
    # ────────────────────────────────────────────────────────────────────────
    _, detection_ref = col.add({

        "severity":
            severity,

        "confidence":
            float(confidence),

        "lat":
            round(lat, 6),

        "lng":
            round(lng, 6),

        "accuracy":
            "3-5m (demo GPS)",

        "timestamp":
            captured_at,

        "device":
            "fleet-demo",

        "bus_id":
            bus_id,

        "route_id":
            route_id,

        "camera_id":
            camera_id,

        "defect_type":
            DEFECT_TYPE,

        "model":
            "YOLOv8s",

        "image_data":
            None,
    })

    # ────────────────────────────────────────────────────────────────────────
    # 2. Send it through Route Memory
    # ────────────────────────────────────────────────────────────────────────
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

    # Link raw observation to persistent road defect
    detection_ref.update({
        "road_defect_id":
            road_defect_id
    })

    print("\n" + "-" * 65)

    print(
        f"🚌 Bus: {bus_id}"
    )

    print(
        f"🛣️  Route: {route_id}"
    )

    print(
        f"📍 GPS: {lat:.6f}, {lng:.6f}"
    )

    print(
        f"⚠️  Severity: {severity.upper()}"
    )

    print(
        f"🎯 Confidence: {confidence:.0%}"
    )

    print(
        f"🧠 Road Defect: {road_defect_id}"
    )

    print(
        f"🔁 Observations: {observation_count}"
    )

    print(
        f"🚌 Unique Buses: {bus_count}"
    )

    print(
        f"✅ Fleet Consensus: {consensus_status.upper()}"
    )

    print(
        f"📌 Action: "
        f"{'CREATED NEW DEFECT' if created else 'MATCHED EXISTING DEFECT'}"
    )


def main():

    print("\n")
    print("=" * 65)
    print("🚍 TRANSITEYE — FLEET CONSENSUS DEMO")
    print("=" * 65)

    print(
        f"\n📍 Demo location:"
        f"\n   {BASE_LAT:.6f}, {BASE_LNG:.6f}"
    )

    print(
        "\nAll three buses will observe the same physical "
        "road defect.\n"
    )

    # ────────────────────────────────────────────────────────────────────────
    # BUS 1
    # ────────────────────────────────────────────────────────────────────────
    add_demo_observation(

        bus_id="BUS-017",

        route_id="PUNE-R12",

        camera_id="FRONT-01",

        severity="moderate",

        confidence=0.88,

        lat=BASE_LAT,

        lng=BASE_LNG,
    )

    # ────────────────────────────────────────────────────────────────────────
    # BUS 2
    #
    # ~5 metres away from the first observation.
    # ────────────────────────────────────────────────────────────────────────
    add_demo_observation(

        bus_id="BUS-023",

        route_id="PUNE-R08",

        camera_id="FRONT-01",

        severity="moderate",

        confidence=0.91,

        lat=BASE_LAT + 0.000025,

        lng=BASE_LNG + 0.000025,
    )

    # ────────────────────────────────────────────────────────────────────────
    # BUS 3
    #
    # ~8-10 metres away from the original point.
    # Also upgrades the defect from MODERATE → SEVERE.
    # ────────────────────────────────────────────────────────────────────────
    add_demo_observation(

        bus_id="BUS-041",

        route_id="PUNE-R21",

        camera_id="FRONT-02",

        severity="severe",

        confidence=0.95,

        lat=BASE_LAT - 0.000035,

        lng=BASE_LNG + 0.000020,
    )

    print("\n")
    print("=" * 65)
    print("✅ FLEET CONSENSUS DEMO COMPLETE")
    print("=" * 65)

    print(
        "\nExpected result:"
    )

    print(
        "   ONE persistent road defect"
    )

    print(
        "   Observations: 3"
    )

    print(
        "   Unique buses: 3"
    )

    print(
        "   Consensus: VERIFIED"
    )

    print(
        "   Severity: SEVERE"
    )

    print()


if __name__ == "__main__":
    main()