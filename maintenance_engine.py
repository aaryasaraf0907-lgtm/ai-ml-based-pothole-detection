"""
maintenance_engine.py

TransitEye — Proactive Road Maintenance Intelligence

Reads persistent road defects from Firestore and determines:

    1. Whether maintenance is required
    2. Maintenance priority
    3. Severity trend
    4. Persistence
    5. Recommended action
    6. Reason for the maintenance decision

Core principle:

    Every credible defect is flagged for maintenance early.
    Severity, repetition, fleet corroboration and deterioration
    determine how urgently it should be addressed.

This is intentionally rule-based and explainable for the prototype.
"""

from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore


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

print("✅ Firebase connected")


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

SEVERITY_RANK = {
    "minor": 0,
    "moderate": 1,
    "severe": 2
}


# ════════════════════════════════════════════════════════════════════════════
# TIME HELPERS
# ════════════════════════════════════════════════════════════════════════════

def to_datetime(value):

    if value is None:
        return None

    if hasattr(value, "to_datetime"):
        value = value.to_datetime()

    if isinstance(value, datetime):

        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value

    return None


def calculate_persistence_days(
    first_seen,
    last_seen
):

    first = to_datetime(
        first_seen
    )

    last = to_datetime(
        last_seen
    )

    if first is None or last is None:
        return 0

    difference = (
        last - first
    ).total_seconds()

    return max(
        0,
        round(
            difference / 86400,
            1
        )
    )


# ════════════════════════════════════════════════════════════════════════════
# TREND
# ════════════════════════════════════════════════════════════════════════════

def calculate_trend(
    severity_history
):
    """
    Determine whether the defect is:

        stable
        worsening
        improving
    """

    if not severity_history:
        return "stable"

    if len(severity_history) < 2:
        return "stable"

    recent = severity_history[-3:]

    ranks = [
        SEVERITY_RANK.get(
            severity,
            0
        )
        for severity in recent
    ]

    # Example:
    # minor → minor → moderate
    # = worsening

    if len(ranks) >= 3:

        if (
            ranks[-1] > ranks[-2]
            and
            ranks[-2] >= ranks[-3]
        ):
            return "worsening"

        if (
            ranks[-1] < ranks[-2]
            and
            ranks[-2] <= ranks[-3]
        ):
            return "improving"

    # Two observations

    if ranks[-1] > ranks[-2]:
        return "worsening"

    if ranks[-1] < ranks[-2]:
        return "improving"

    return "stable"


# ════════════════════════════════════════════════════════════════════════════
# MAINTENANCE PRIORITY
# ════════════════════════════════════════════════════════════════════════════

def calculate_maintenance_priority(
    severity,
    observation_count,
    unique_bus_count,
    consensus_status,
    trend
):
    """
    Decide how urgently maintenance should occur.

    Important:

        MINOR is still maintenance-required.

    It simply starts at preventive priority.
    """

    # ────────────────────────────────────────────────────────────────────────
    # SEVERE
    # ────────────────────────────────────────────────────────────────────────

    if severity == "severe":

        return (
            "critical",
            "Immediate inspection and repair"
        )


    # ────────────────────────────────────────────────────────────────────────
    # MODERATE
    # ────────────────────────────────────────────────────────────────────────

    if severity == "moderate":

        if trend == "worsening":

            return (
                "urgent",
                "Prioritize field inspection before further deterioration"
            )

        if (
            unique_bus_count >= 2
            or
            observation_count >= 3
            or
            consensus_status in [
                "confirmed",
                "verified"
            ]
        ):

            return (
                "high",
                "Schedule maintenance inspection"
            )

        return (
            "high",
            "Schedule maintenance"
        )


    # ────────────────────────────────────────────────────────────────────────
    # MINOR
    # ────────────────────────────────────────────────────────────────────────

    if severity == "minor":

        # A minor defect that is already worsening
        # should be escalated.

        if trend == "worsening":

            return (
                "high",
                "Inspect before further deterioration"
            )

        # Multiple observations / buses indicate
        # that this is a persistent problem.

        if (
            unique_bus_count >= 2
            or
            observation_count >= 2
            or
            consensus_status in [
                "confirmed",
                "verified"
            ]
        ):

            return (
                "scheduled",
                "Schedule preventive maintenance"
            )

        # First minor observation.

        return (
            "preventive",
            "Flag for preventive maintenance"
        )


    # Safe fallback

    return (
        "preventive",
        "Flag for preventive maintenance"
    )


# ════════════════════════════════════════════════════════════════════════════
# PRIORITY REASON
# ════════════════════════════════════════════════════════════════════════════

def build_priority_reason(
    severity,
    observation_count,
    unique_bus_count,
    consensus_status,
    trend,
    persistence_days
):

    reasons = []

    reasons.append(
        f"{severity} severity"
    )

    if observation_count > 1:

        reasons.append(
            f"{observation_count} observations"
        )

    if unique_bus_count > 1:

        reasons.append(
            f"{unique_bus_count} buses corroborated"
        )

    if consensus_status in [
        "confirmed",
        "verified"
    ]:

        reasons.append(
            f"fleet status {consensus_status}"
        )

    if persistence_days > 0:

        reasons.append(
            f"persistent for {persistence_days} days"
        )

    if trend == "worsening":

        reasons.append(
            "severity is worsening"
        )

    elif trend == "improving":

        reasons.append(
            "severity is improving"
        )

    return ", ".join(
        reasons
    )


# ════════════════════════════════════════════════════════════════════════════
# PROCESS ONE ROAD DEFECT
# ════════════════════════════════════════════════════════════════════════════

def process_defect(doc):

    data = doc.to_dict()

    defect_id = doc.id

    severity = data.get(
        "severity",
        "minor"
    )

    observation_count = int(
        data.get(
            "observation_count",
            1
        )
    )

    unique_bus_count = int(
        data.get(
            "unique_bus_count",
            1
        )
    )

    consensus_status = data.get(
        "consensus_status",
        "reported"
    )

    severity_history = list(
        data.get(
            "severity_history",
            []
        )
    )

    first_seen = data.get(
        "first_seen"
    )

    last_seen = data.get(
        "last_seen"
    )

    persistence_days = (
        calculate_persistence_days(
            first_seen,
            last_seen
        )
    )

    trend = calculate_trend(
        severity_history
    )

    priority, recommended_action = (
        calculate_maintenance_priority(

            severity,

            observation_count,

            unique_bus_count,

            consensus_status,

            trend
        )
    )

    priority_reason = (
        build_priority_reason(

            severity,

            observation_count,

            unique_bus_count,

            consensus_status,

            trend,

            persistence_days
        )
    )


    # ═══════════════════════════════════════════════════════════════════════
    # UPDATE FIRESTORE
    # ═══════════════════════════════════════════════════════════════════════

    doc.reference.update({

        # Maintenance decision
        "maintenance_required":
            True,

        "maintenance_priority":
            priority,

        "recommended_action":
            recommended_action,

        "priority_reason":
            priority_reason,

        # Historical intelligence
        "severity_trend":
            trend,

        "persistence_days":
            persistence_days,

        # Current maintenance state
        "maintenance_status":
            data.get(
                "maintenance_status",
                "pending"
            ),

        "last_maintenance_analysis":
            firestore.SERVER_TIMESTAMP,

        # Remove the old Road Health fields
        # because Road Health is not part of our
        # final core maintenance design.

        "road_health_score":
            firestore.DELETE_FIELD,

        "road_health_status":
            firestore.DELETE_FIELD,
    })


    # ═══════════════════════════════════════════════════════════════════════
    # TERMINAL OUTPUT
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "-" * 70)

    print(
        f"🛣️  ROAD DEFECT: "
        f"{defect_id}"
    )

    print(
        f"⚠️  Severity: "
        f"{severity.upper()}"
    )

    print(
        f"🔁 Observations: "
        f"{observation_count}"
    )

    print(
        f"🚌 Unique buses: "
        f"{unique_bus_count}"
    )

    print(
        f"✅ Fleet consensus: "
        f"{consensus_status.upper()}"
    )

    print(
        f"📆 Persistence: "
        f"{persistence_days} days"
    )

    print(
        f"📈 Trend: "
        f"{trend.upper()}"
    )

    print(
        f"🔧 Maintenance required: "
        f"YES"
    )

    print(
        f"🚦 Maintenance priority: "
        f"{priority.upper()}"
    )

    print(
        f"📌 Recommended action: "
        f"{recommended_action}"
    )

    print(
        f"🧠 Reason: "
        f"{priority_reason}"
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():

    print("\n")
    print("=" * 70)
    print(
        "🚦 TRANSITEYE — PROACTIVE MAINTENANCE ENGINE"
    )
    print("=" * 70)

    docs = list(
        road_defects_col.stream()
    )

    if not docs:

        print(
            "\n⚠️ No road defects found."
        )

        return

    print(
        f"\nFound {len(docs)} road defects."
    )


    for doc in docs:

        try:

            process_defect(
                doc
            )

        except Exception as e:

            print(
                f"\n❌ Failed to process "
                f"{doc.id}: {e}"
            )


    print("\n")
    print("=" * 70)

    print(
        "✅ MAINTENANCE ANALYSIS COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()