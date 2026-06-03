"""Interactive HSV tuner for LED color thresholds."""

from __future__ import annotations

import cv2
import numpy as np

from bluerov_led.config import ProjectPaths
from bluerov_led.dataset_io import DatasetReader


def run_hsv_tuner(
    dataset_folder,
    frame_index: int = 80,
    display_scale: float = 0.5,
) -> None:
    reader = DatasetReader(dataset_folder)
    frame_paths = reader.list_frame_paths()

    frame_index = min(frame_index, len(frame_paths) - 1)
    frame = cv2.imread(str(frame_paths[frame_index]))

    if frame is None:
        raise RuntimeError("Frame could not be read.")

    frame = cv2.resize(frame, None, fx=display_scale, fy=display_scale)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def nothing(_):
        pass

    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H Min", "HSV Tuner", 20, 179, nothing)
    cv2.createTrackbar("H Max", "HSV Tuner", 95, 179, nothing)
    cv2.createTrackbar("S Min", "HSV Tuner", 80, 255, nothing)
    cv2.createTrackbar("S Max", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V Min", "HSV Tuner", 150, 255, nothing)
    cv2.createTrackbar("V Max", "HSV Tuner", 255, 255, nothing)

    while True:
        h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("Original", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

        if cv2.waitKey(1) == ord("q"):
            break

    print("Selected HSV values:")
    print("LOWER_LED =", [h_min, s_min, v_min])
    print("UPPER_LED =", [h_max, s_max, v_max])
    cv2.destroyAllWindows()


if __name__ == "__main__":
    paths = ProjectPaths()
    run_hsv_tuner(paths.dataset_folder("BackOnly_Test_02"))
