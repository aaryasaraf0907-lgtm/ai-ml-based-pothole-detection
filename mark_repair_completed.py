import sys

import firebase_admin
from firebase_admin import credentials, firestore


# Firebase
cred = credentials.Certificate(
    "serviceAccountKey.json"
)

firebase_admin.initialize_app(cred)

db = firestore.client()

road_defects = db.collection(
    "road_defects"
)


if len(sys.argv) < 2:

    print(
        "Usage:\n"
        "python mark_repair_completed.py RD-002"
    )

    raise SystemExit


defect_id = sys.argv[1]

ref = road_defects.document(
    defect_id
)

doc = ref.get()

if not doc.exists:

    print(
        f"❌ Road defect {defect_id} not found."
    )

    raise SystemExit


data = doc.to_dict()


if data.get(
    "maintenance_required"
) is not True:

    print(
        "⚠️ This defect is not currently "
        "marked as requiring maintenance."
    )


ref.update({

    "maintenance_status":
        "post_repair_monitoring",

    "repair_completed_at":
        firestore.SERVER_TIMESTAMP,

    "verification_required":
        True,

    "verification_required_passes":
        3,

    "verification_passes":
        0,

    "verification_bus_ids":
        [],

    "verification_status":
        "pending",

})


print(
    f"\n✅ {defect_id} marked as "
    f"POST-REPAIR MONITORING."
)

print(
    "Verification passes required: 3"
)