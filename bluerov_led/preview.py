"""Interactive PNG sequence preview."""

from __future__ import annotations

import cv2

from bluerov_led.config import VisionConfig
from bluerov_led.dataset_io import DatasetReader


def preview_dataset(
    dataset_folder,
    config: VisionConfig | None = None,
    display_scale: float = 0.5,
) -> None:
    if config is None:
        config = VisionConfig()

    reader = DatasetReader(dataset_folder)
    frame_paths = reader.list_frame_paths()

    print("Total frame count:", len(frame_paths))
    print("Duration at 60 FPS:", len(frame_paths) / config.fps, "seconds")

    for i, path in enumerate(frame_paths):
        frame = cv2.imread(str(path))

        if frame is None:
            print("Could not read:", path)
            continue

        display = cv2.resize(frame, None, fx=display_scale, fy=display_scale)

        cv2.putText(
            display,
            f"Frame: {i}/{len(frame_paths) - 1}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Unity PNG Sequence Preview", display)
        key = cv2.waitKey(int(1000 / config.fps))

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
