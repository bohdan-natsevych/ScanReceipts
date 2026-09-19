"""Collect the CAM-5 hardware-spike evidence on the target Windows machine."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from scan_receipts.camera import SceneSettler, descriptor_source, enumerate_cameras
from scan_receipts.config import default_settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe cameras, controls, focus settling, and exposure locking."
    )
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=Path("camera_spike_report.json"))
    args = parser.parse_args()
    settings = default_settings()
    settings.focus_lock = True
    settings.exposure_lock = True
    settings.white_balance_lock = True
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "backend": "OpenCV / Media Foundation",
        "devices": [],
        "manual_checks_required": [
            "Identify which entry is the Phone Link connected camera.",
            "Confirm live preview stays sharp after one-shot focus locks.",
            "Confirm hand motion does not visibly shift exposure or white balance.",
        ],
    }
    for descriptor in enumerate_cameras(settings):
        device = {"descriptor": asdict(descriptor), "opened": False, "frames": 0}
        source = descriptor_source(descriptor, settings)
        sharpness = []
        try:
            source.open()
            device["opened"] = True
            device["capabilities"] = [asdict(item) for item in source.capabilities()]
            settler = SceneSettler(source)
            settled_after = None
            started = time.monotonic()
            deadline = started + args.seconds
            while time.monotonic() < deadline:
                packet = source.read()
                if packet is None:
                    continue
                if settled_after is None and settler.observe(packet):
                    settled_after = time.monotonic() - started
                gray = cv2.cvtColor(packet.frame, cv2.COLOR_BGR2GRAY)
                sharpness.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                device["frames"] += 1
            device["sharpness_first"] = sharpness[0] if sharpness else None
            device["sharpness_final"] = sharpness[-1] if sharpness else None
            device["settled_after_seconds"] = settled_after
        except Exception as error:
            device["error"] = str(error)
        finally:
            source.close()
        report["devices"].append(device)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(report['devices'])} camera result(s).")
    return 0 if report["devices"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
