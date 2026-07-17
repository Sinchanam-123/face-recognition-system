"""
Quick webcam sanity check — opens a live preview window in its own process.

Run:  py backend/check_camera.py

If this window is black/gray or frozen, the problem is the camera itself
(privacy shutter, another app using it, or Windows camera permissions) — not
the attendance app. Press Q to close.
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("❌ Could not open the webcam (device 0).")

print("Showing live camera. Press Q in the window to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("… no frame"); continue
    # Overlay simple stats so you can tell live from frozen.
    m, s = frame.mean(), frame.std()
    hint = "looks LIVE" if s > 20 else "BLANK/COVERED?  (std is very low)"
    cv2.putText(frame, f"mean={m:.0f} std={s:.0f}  {hint}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Camera check - press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
