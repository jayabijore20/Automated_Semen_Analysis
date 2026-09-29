"""
Diagnostic: checks whether src/analysis/concentration.py has the four
wiring points required for a saved calibration to reach the
concentration calculation. Read-only — changes nothing.

Run from your project root:  python diagnose_calibration_wiring.py
"""
import re, sys
from pathlib import Path

path = Path("src/analysis/concentration.py")
if not path.exists():
    print(f"ERROR: {path} not found. Run this from your project root.")
    sys.exit(1)

src = path.read_text(encoding="utf-8")

checks = [
    ("1. resolve_microns_per_pixel() accepts the parameter",
     r"def resolve_microns_per_pixel\([^)]*saved_calibration_um_per_pixel"),
    ("2. resolve_microns_per_pixel() uses it as top priority",
     r"if saved_calibration_um_per_pixel is not None and saved_calibration_um_per_pixel > 0"),
    ("3. compute_concentration_metrics() accepts the parameter",
     r"def compute_concentration_metrics\((?:[^)]|\n)*saved_calibration_um_per_pixel"),
    ("4. ...and forwards it into resolve_microns_per_pixel()",
     r"resolve_microns_per_pixel\(\s*\n\s*magnification, camera_resolution, manual_microns_per_pixel,\s*\n\s*saved_calibration_um_per_pixel"),
    ("5. from_params() forwards it (PHYSICAL CHAMBER path) <-- your failure",
     r"saved_calibration_um_per_pixel=concentration_params\.get\(\"saved_calibration_um_per_pixel\"\)"),
    ("6. from_params() forwards it (STANDARD SLIDE path)",
     r"concentration_params\.get\(\"saved_calibration_um_per_pixel\"\),"),
]

print(f"Inspecting {path}\n")
missing = []
for label, pattern in checks:
    ok = re.search(pattern, src) is not None
    print(f"  [{'OK ' if ok else 'MISSING'}] {label}")
    if not ok:
        missing.append(label)

print()
if missing:
    print("RESULT: wiring incomplete. Missing:")
    for m in missing:
        print("   -", m)
    print("\nThis is why compute_concentration_metrics_from_params() returns")
    print("'calibration_required' despite a valid saved calibration.")
else:
    print("RESULT: all wiring present. If the test still fails, the cause is")
    print("elsewhere - capture the actual resolve_microns_per_pixel() return value.")