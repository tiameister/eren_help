"""LED tracking offline pipeline orchestrator."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import pandas as pd

from bluerov_led.bit_extractor import BitExtractor
from bluerov_led.config import VisionConfig, ProjectPaths
from bluerov_led.dataset_io import ArtifactWriter, DatasetReader
from bluerov_led.distance_model import DistanceModel
from bluerov_led.geometry import GeometryCalculator
from bluerov_led.packet_builder import ObservationPacketBuilder
from bluerov_led.pair_selector import PairSelector
from bluerov_led.spatio_temporal_matcher import SpatioTemporalMatcher
from bluerov_led.temporal_decoder import TemporalDecoder
from bluerov_led.types import FrameRecord, PipelineResult
from bluerov_led.vision_core import LedCandidateExtractor
from bluerov_led.filtering import SignalSmoother1D, RollingIQRFilter
from bluerov_led.udp_transport import UdpSender


def _float_or_zero(value: float | None) -> float:
    return 0.0 if value is None else float(value)


class BackFacePipeline:
    """Orchestrates extract, decode, filter, calibrate, and packet build."""

    def __init__(
        self,
        config: VisionConfig | None = None,
        paths: ProjectPaths | None = None,
    ) -> None:
        self.config = config or VisionConfig()
        self.paths = paths or ProjectPaths()

        self.decoder = TemporalDecoder(self.config)
        self.packet_builder = ObservationPacketBuilder(self.config)
        self._streaming: StreamingPipeline | None = None

    def _get_streaming(self) -> StreamingPipeline:
        if self._streaming is None:
            self._streaming = StreamingPipeline(
                config=self.config,
                udp_ip=None,
            )
        else:
            self._streaming.apply_config(self.config)
        return self._streaming

    def _draw_preview(
        self,
        frame,
        record: FrameRecord,
        candidates,
        mask_clean,
        tracks=None,
    ) -> None:
        output = frame.copy()

        for idx, c in enumerate(candidates[:8]):
            label = f"A:{int(c.area)}"
            if tracks:
                for t in tracks:
                    if t.candidate is not None and t.candidate.cx == c.cx:
                        label = f"T{t.track_id} {label}"
                        break

            cv2.rectangle(
                output,
                (c.x, c.y),
                (c.x + c.w, c.y + c.h),
                (0, 255, 0),
                2,
            )
            cv2.circle(output, (c.cx, c.cy), 4, (0, 0, 255), -1)
            cv2.putText(
                output,
                label,
                (c.x, c.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

        if record.pair_found:
            led1_x, led1_y = int(record.led1_x), int(record.led1_y)
            led2_x, led2_y = int(record.led2_x), int(record.led2_y)
            image_center_x = record.image_width / 2.0
            image_center_y = record.image_height / 2.0

            cv2.line(output, (led1_x, led1_y), (led2_x, led2_y), (255, 0, 0), 2)
            cv2.putText(
                output,
                f"d_px: {record.pixel_distance:.1f}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.circle(
                output,
                (int(record.mid_x), int(record.mid_y)),
                6,
                (255, 255, 255),
                -1,
            )
            cv2.line(
                output,
                (int(image_center_x), int(image_center_y)),
                (int(record.mid_x), int(record.mid_y)),
                (0, 255, 255),
                2,
            )
            face_label = record.face_id or "?"
            cv2.putText(
                output,
                (
                    f"{face_label} acc:{record.pattern_accuracy:.2f} "
                    f"corr:{record.pair_correlation:.2f}"
                    if record.pair_correlation is not None
                    else face_label
                ),
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            output,
            (
                f"Frame:{record.frame} Bit:{record.bit} "
                f"Pair:{record.pair_found} Cands:{record.candidate_count} "
                f"Tracks:{record.active_track_count}"
            ),
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        scale = self.config.display_scale
        cv2.imshow(
            "LED Pair Tracking",
            cv2.resize(output, None, fx=scale, fy=scale),
        )
        cv2.imshow(
            "LED Mask",
            cv2.resize(mask_clean, None, fx=scale, fy=scale),
        )

    def _dominant_face_from_df(self, df: pd.DataFrame) -> tuple[str, str]:
        if "face_id" not in df.columns:
            return "BACK", self.config.face_patterns["BACK"]

        paired = df[(df["pair_found"] == 1) & df["face_id"].notna()]
        if len(paired) == 0:
            return "BACK", self.config.face_patterns["BACK"]

        face_id = str(paired["face_id"].mode().iloc[0])
        pattern = self.config.face_patterns.get(
            face_id, self.config.face_patterns["BACK"]
        )
        return face_id, pattern

    def extract(self, dataset: str, preview: bool = False) -> Path:
        dataset_folder = self.paths.dataset_folder(dataset)
        output_folder = self.paths.output_folder(dataset)
        ArtifactWriter.ensure_dir(output_folder)

        reader = DatasetReader(dataset_folder)
        frame_paths = reader.list_frame_paths()

        print("Matcher mode:", self.config.matcher_mode)
        print("Total frame count:", len(frame_paths))
        print(
            "Duration at 60 FPS:",
            len(frame_paths) / self.config.fps,
            "seconds",
        )
        print("Frames per bit:", self.config.frames_per_bit)

        streaming = self._get_streaming()
        streaming.reset()

        def preview_callback(frame, record, candidates, mask_clean, tracks):
            self._draw_preview(frame, record, candidates, mask_clean, tracks)
            key = cv2.waitKey(int(1000 / self.config.fps))
            return key != ord("q")

        records, read_fail_count = streaming.process_png_sequence(
            dataset_folder=dataset_folder,
            dataset_name=dataset,
            preview=preview,
            preview_callback=preview_callback if preview else None,
        )

        if preview:
            cv2.destroyAllWindows()

        if read_fail_count > 0:
            print(
                f"Read failures: {read_fail_count} / {len(frame_paths)} "
                "(OFF placeholders inserted)"
            )

        csv_path = self.paths.pair_csv(dataset)
        ArtifactWriter.write_frame_records_csv(csv_path, records)

        df = pd.DataFrame([r.to_csv_row() for r in records])
        print("CSV saved:", csv_path)
        print("Total processed frames:", len(df))

        if "face_id" in df.columns:
            detected = df[df["face_id"].notna()]["face_id"].value_counts()
            print("Detected face counts:")
            print(detected.to_string())

        valid_distances = df["pixel_distance"].dropna()
        if len(valid_distances) > 0:
            print("Mean pixel distance:", valid_distances.mean())
            print("Min pixel distance:", valid_distances.min())
            print("Max pixel distance:", valid_distances.max())
        else:
            print("No valid LED pair distance found.")

        bits = df["bit"].astype(int).tolist()[:120]
        print("First 120 frame bits:")
        print("".join(str(b) for b in bits))

        return csv_path

    def decode_pattern(self, dataset: str):
        csv_path = self.paths.pair_csv(dataset)
        df = ArtifactWriter.read_frame_records_csv(csv_path)
        frame_bits = df["bit"].astype(int).tolist()

        dominant_face, expected_pattern = self._dominant_face_from_df(df)

        summary = self.decoder.decode_dataset(
            dataset, frame_bits, expected_pattern=expected_pattern
        )

        print("Dominant decoded face:", dominant_face)
        self.decoder.print_summary(summary)

        json_path = self.paths.pattern_summary_json(dataset)
        ArtifactWriter.write_pattern_summary(json_path, summary)
        print("\nSummary JSON saved:", json_path)
        return summary

    def filter_distances(self, dataset: str) -> Path:
        csv_path = self.paths.pair_csv(dataset)
        df = ArtifactWriter.read_frame_records_csv(csv_path)

        print("Total rows:", len(df))

        valid = df[
            (df["bit"] == 1)
            & (df["pair_found"] == 1)
            & (df["pixel_distance"].notna())
        ].copy()

        if self.config.matcher_mode == "legacy_largest2":
            valid = valid[valid["candidate_count"] == 2]
        elif "face_id" in valid.columns:
            valid = valid[valid["face_id"].notna()]

        print("Valid ON + pair frames:", len(valid))

        if len(valid) == 0:
            raise RuntimeError("No valid pixel distance data found.")

        print("\nRaw valid distance statistics:")
        print("Mean:", valid["pixel_distance"].mean())
        print("Median:", valid["pixel_distance"].median())
        print("Min:", valid["pixel_distance"].min())
        print("Max:", valid["pixel_distance"].max())
        print("Std:", valid["pixel_distance"].std())

        q1 = valid["pixel_distance"].quantile(0.25)
        q3 = valid["pixel_distance"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered = valid[
            (valid["pixel_distance"] >= lower_bound)
            & (valid["pixel_distance"] <= upper_bound)
        ].copy()

        print("\nIQR filter bounds:")
        print("Lower:", lower_bound)
        print("Upper:", upper_bound)
        print("\nFiltered distance statistics:")
        print("Filtered frame count:", len(filtered))
        print("Mean:", filtered["pixel_distance"].mean())
        print("Median:", filtered["pixel_distance"].median())

        out_path = self.paths.filtered_distance_csv(dataset)
        ArtifactWriter.write_dataframe_csv(out_path, filtered)
        print("\nFiltered CSV saved:", out_path)
        return out_path

    def calibrate(self, points=None) -> DistanceModel:
        model = DistanceModel.fit(points)
        model.print_report()

        cal_folder = self.paths.calibration_folder()
        ArtifactWriter.ensure_dir(cal_folder)

        summary_path = self.paths.distance_model_json()
        ArtifactWriter.write_json(summary_path, model.to_summary_dict())

        eval_path = cal_folder / "distance_model_evaluation.csv"
        model.evaluation_dataframe().to_csv(eval_path, index=False)

        print("\nModel summary saved:", summary_path)
        print("Evaluation CSV saved:", eval_path)
        return model

    def build_packet(
        self,
        dataset: str,
        frame: int | None = None,
    ) -> tuple[dict, Path]:
        pair_csv = self.paths.pair_csv(dataset)
        if not pair_csv.exists():
            raise FileNotFoundError(f"Pair CSV not found: {pair_csv}")

        model_path = self.paths.distance_model_json()
        model_data = ArtifactWriter.read_json(model_path)
        if model_data is None:
            raise FileNotFoundError(f"Distance model summary not found: {model_path}")

        df = ArtifactWriter.read_frame_records_csv(pair_csv)
        pattern_summary = ArtifactWriter.read_pattern_summary(
            self.paths.pattern_summary_json(dataset)
        )
        distance_model = DistanceModel.from_summary_dict(model_data)

        packet = self.packet_builder.build_packet(
            dataset=dataset,
            df=df,
            pattern_summary=pattern_summary,
            distance_model=distance_model,
            target_frame=frame,
        )

        packet_path = self.paths.observation_packet_json(dataset, frame)
        ArtifactWriter.write_json(packet_path, packet)

        print("Dataset:", dataset)
        if frame is None:
            print("Requested frame: None, using first valid observation.")
        else:
            print("Requested frame:", frame)

        print("Observation packet:")
        print(json.dumps(packet, indent=4))
        print("\nPacket saved:", packet_path)

        return packet, packet_path

    def run_all(self, dataset: str, preview: bool = False) -> PipelineResult:
        pair_csv = self.extract(dataset, preview=preview)
        self.decode_pattern(dataset)
        filtered_csv = self.filter_distances(dataset)
        self.calibrate()
        _, packet_json = self.build_packet(dataset)

        return PipelineResult(
            dataset=dataset,
            pair_csv=str(pair_csv),
            pattern_json=str(self.paths.pattern_summary_json(dataset)),
            filtered_csv=str(filtered_csv),
            model_json=str(self.paths.distance_model_json()),
            packet_json=str(packet_json),
        )


class StreamingPipeline:
    """Canonical frame-by-frame processor (offline PNG and live stream)."""

    def __init__(
        self,
        config: VisionConfig | None = None,
        distance_model_dict: dict | None = None,
        udp_ip: str | None = "127.0.0.1",
        udp_port: int | None = 5005,
    ) -> None:
        self.config = config or VisionConfig()

        self.extractor = LedCandidateExtractor(self.config)
        self.bit_extractor = BitExtractor(self.config)
        self.legacy_selector = PairSelector(self.config)
        self.matcher = SpatioTemporalMatcher(self.config)
        self.geometry = GeometryCalculator(self.config)
        self.packet_builder = ObservationPacketBuilder(self.config)
        if distance_model_dict is not None:
            self.distance_model = DistanceModel.from_summary_dict(
                distance_model_dict
            )
        else:
            self.distance_model = DistanceModel.fit()

        self.udp_sender = UdpSender(udp_ip, udp_port) if udp_ip else None

        self.rolling_iqr = RollingIQRFilter(
            self.config.rolling_iqr_window,
            self.config.outlier_distance_rejection_iqr_multiplier,
        )
        self.distance_lpf = SignalSmoother1D(self.config.signal_1d_lpf_alpha)
        self.error_norm_lpf = SignalSmoother1D(self.config.signal_1d_lpf_alpha)
        self.error_x_lpf = SignalSmoother1D(self.config.signal_1d_lpf_alpha)
        self.error_y_lpf = SignalSmoother1D(self.config.signal_1d_lpf_alpha)

        self.last_valid_packet: dict | None = None
        self.invalid_frames_count = 0
        self._sequence_size: tuple[int, int] | None = None
        self._read_fail_count = 0

    def apply_config(self, config: VisionConfig) -> None:
        """Propagate config to all frame-processing submodules."""
        self.config = config
        self.extractor.config = config
        self.bit_extractor.config = config
        self.legacy_selector.config = config
        self.matcher.config = config
        self.matcher.tracker.config = config
        self.matcher.buffers.config = config
        self.matcher.face_decoder.config = config
        self.matcher.face_decoder._decoder.config = config
        self.geometry.config = config
        self.packet_builder.config = config

    def reset(self) -> None:
        self.matcher.reset()
        self.rolling_iqr.reset()
        self.distance_lpf.reset()
        self.error_norm_lpf.reset()
        self.error_x_lpf.reset()
        self.error_y_lpf.reset()
        self.last_valid_packet = None
        self.invalid_frames_count = 0
        self._sequence_size = None
        self._read_fail_count = 0

    def _emit_pid_udp(
        self,
        *,
        valid: bool,
        error_yaw: float,
        error_heave: float,
        distance_surge: float,
    ) -> None:
        if self.udp_sender is None:
            return
        self.udp_sender.send_pid_packet(
            valid=valid,
            error_yaw=error_yaw,
            error_heave=error_heave,
            distance_surge=distance_surge,
        )

    def _emit_pid_lost(self) -> None:
        self._emit_pid_udp(
            valid=False,
            error_yaw=0.0,
            error_heave=0.0,
            distance_surge=0.0,
        )

    def _emit_pid_from_observation(
        self,
        packet: dict | None,
        *,
        is_valid: bool,
        held: bool,
    ) -> None:
        if is_valid and packet is not None:
            self._emit_pid_udp(
                valid=True,
                error_yaw=_float_or_zero(packet.get("error_x")),
                error_heave=_float_or_zero(packet.get("error_y")),
                distance_surge=_float_or_zero(packet.get("estimated_distance")),
            )
        elif held and packet is not None:
            self._emit_pid_udp(
                valid=False,
                error_yaw=_float_or_zero(packet.get("error_x")),
                error_heave=_float_or_zero(packet.get("error_y")),
                distance_surge=_float_or_zero(packet.get("estimated_distance")),
            )
        else:
            self._emit_pid_lost()

    def _set_sequence_size(self, image_width: int, image_height: int) -> None:
        self._sequence_size = (image_width, image_height)

    def _sequence_dimensions(self) -> tuple[int, int]:
        if self._sequence_size is not None:
            return self._sequence_size
        return 0, 0

    def _make_read_failure_record(
        self,
        frame_index: int,
        file_name: str,
    ) -> FrameRecord:
        image_width, image_height = self._sequence_dimensions()
        dt = 1.0 / self.config.fps

        if image_width > 0 and image_height > 0:
            if self.config.matcher_mode != "legacy_largest2":
                self.matcher.update_tracks([], image_width, image_height, dt)

        return FrameRecord(
            frame=frame_index,
            file=file_name,
            candidate_count=0,
            total_area=0.0,
            bit=0,
            pair_found=0,
            led1_x=None,
            led1_y=None,
            led2_x=None,
            led2_y=None,
            pixel_distance=None,
            mid_x=None,
            mid_y=None,
            error_x=None,
            error_y=None,
            ray_x=None,
            ray_y=None,
            ray_z=None,
            camera_vertical_fov_deg=self.config.camera_vertical_fov_deg,
            image_width=image_width,
            image_height=image_height,
            active_track_count=(
                self.matcher.tracker.active_track_count
                if self.config.matcher_mode != "legacy_largest2"
                else 0
            ),
            matcher_mode=self.config.matcher_mode,
            held=0,
            read_ok=0,
        )

    def _build_legacy_frame_record(
        self,
        candidates,
        image_width: int,
        image_height: int,
        frame_index: int,
        file_name: str,
    ) -> FrameRecord:
        total_area = self.bit_extractor.total_area(candidates)
        bit = self.bit_extractor.bit_from_candidates(candidates)

        pair_found = 0
        led1_x = led1_y = led2_x = led2_y = None
        pixel_distance = None
        mid_x = mid_y = error_x = error_y = None
        ray_x = ray_y = ray_z = None
        face_id = None
        pattern = None
        pattern_accuracy = None

        pair = self.legacy_selector.select(candidates)
        if pair is not None:
            c1, c2 = pair
            geom = self.geometry.compute_pair_geometry(
                c1, c2, image_width, image_height
            )
            pair_found = 1
            led1_x = geom.led1_x
            led1_y = geom.led1_y
            led2_x = geom.led2_x
            led2_y = geom.led2_y
            pixel_distance = geom.pixel_distance
            mid_x = geom.mid_x
            mid_y = geom.mid_y
            error_x = geom.error_x
            error_y = geom.error_y
            ray_x = geom.ray_x
            ray_y = geom.ray_y
            ray_z = geom.ray_z

        return FrameRecord(
            frame=frame_index,
            file=file_name,
            candidate_count=len(candidates),
            total_area=total_area,
            bit=bit,
            pair_found=pair_found,
            led1_x=led1_x,
            led1_y=led1_y,
            led2_x=led2_x,
            led2_y=led2_y,
            pixel_distance=pixel_distance,
            mid_x=mid_x,
            mid_y=mid_y,
            error_x=error_x,
            error_y=error_y,
            ray_x=ray_x,
            ray_y=ray_y,
            ray_z=ray_z,
            camera_vertical_fov_deg=self.config.camera_vertical_fov_deg,
            image_width=image_width,
            image_height=image_height,
            face_id=face_id,
            pattern=pattern,
            pattern_accuracy=pattern_accuracy,
            active_track_count=0,
            matcher_mode="legacy_largest2",
            held=0,
            read_ok=1,
        )

    def _spatio_frame_record(
        self,
        *,
        frame_index: int,
        file_name: str,
        candidates,
        image_width: int,
        image_height: int,
        frame_bit: int,
        match,
        geom,
        observation_valid: bool,
        held: bool,
    ) -> FrameRecord:
        active_count = self.matcher.tracker.active_track_count

        if observation_valid and match is not None and geom is not None:
            return FrameRecord(
                frame=frame_index,
                file=file_name,
                candidate_count=len(candidates),
                total_area=self.bit_extractor.total_area(candidates),
                bit=frame_bit,
                pair_found=1,
                led1_x=geom.led1_x,
                led1_y=geom.led1_y,
                led2_x=geom.led2_x,
                led2_y=geom.led2_y,
                pixel_distance=geom.pixel_distance,
                mid_x=geom.mid_x,
                mid_y=geom.mid_y,
                error_x=geom.error_x,
                error_y=geom.error_y,
                ray_x=geom.ray_x,
                ray_y=geom.ray_y,
                ray_z=geom.ray_z,
                camera_vertical_fov_deg=self.config.camera_vertical_fov_deg,
                image_width=image_width,
                image_height=image_height,
                face_id=match.face_id,
                pattern=match.pattern,
                pattern_accuracy=match.pattern_accuracy,
                active_track_count=active_count,
                track_id_1=match.track_id_1,
                track_id_2=match.track_id_2,
                pair_correlation=match.pair_correlation,
                pair_score=match.pair_score,
                geometry_score=match.geometry_score,
                matcher_mode=self.config.matcher_mode,
                held=0,
                read_ok=1,
            )

        return FrameRecord(
            frame=frame_index,
            file=file_name,
            candidate_count=len(candidates),
            total_area=self.bit_extractor.total_area(candidates),
            bit=frame_bit,
            pair_found=0,
            led1_x=None,
            led1_y=None,
            led2_x=None,
            led2_y=None,
            pixel_distance=None,
            mid_x=None,
            mid_y=None,
            error_x=None,
            error_y=None,
            ray_x=None,
            ray_y=None,
            ray_z=None,
            camera_vertical_fov_deg=self.config.camera_vertical_fov_deg,
            image_width=image_width,
            image_height=image_height,
            face_id=None,
            pattern=None,
            pattern_accuracy=None,
            active_track_count=active_count,
            track_id_1=None,
            track_id_2=None,
            pair_correlation=None,
            pair_score=None,
            geometry_score=None,
            matcher_mode=self.config.matcher_mode,
            held=1 if held else 0,
            read_ok=1,
        )

    def process_png_sequence(
        self,
        dataset_folder: Path,
        dataset_name: str,
        preview: bool = False,
        preview_callback: Callable[..., bool] | None = None,
    ) -> tuple[list[FrameRecord], int]:
        reader = DatasetReader(dataset_folder)
        frame_paths = reader.list_frame_paths()
        records: list[FrameRecord] = []
        dt = 1.0 / self.config.fps
        total_frames = len(frame_paths)

        print(f"PROGRESS: 0/{total_frames}", flush=True)

        for frame_index, path in enumerate(frame_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                self._read_fail_count += 1
                record = self._make_read_failure_record(frame_index, path.name)
                self._emit_pid_lost()
                records.append(record)
                continue

            self._set_sequence_size(frame.shape[1], frame.shape[0])
            _packet, candidates, mask_clean, record = self.process_frame(
                frame,
                dataset_name,
                frame_index,
                file_name=path.name,
                dt=dt,
            )
            records.append(record)

            if preview and preview_callback is not None and record.read_ok:
                tracks = (
                    self.matcher.tracker.list_tracks()
                    if self.config.matcher_mode != "legacy_largest2"
                    else None
                )
                if not preview_callback(
                    frame, record, candidates, mask_clean, tracks
                ):
                    break

            print(f"PROGRESS: {frame_index + 1}/{total_frames}", flush=True)

        return records, self._read_fail_count

    def stream_png_sequence(
        self,
        dataset_folder: Path,
        dataset_name: str,
        *,
        realtime_pacing: bool = True,
    ) -> tuple[int, int]:
        """Process and emit PID UDP per frame without accumulating CSV records."""
        reader = DatasetReader(dataset_folder)
        frame_paths = reader.list_frame_paths()
        dt = 1.0 / self.config.fps
        frame_interval = dt if realtime_pacing else 0.0
        frames_processed = 0

        for frame_index, path in enumerate(frame_paths):
            loop_start = time.perf_counter()

            frame = cv2.imread(str(path))
            if frame is None:
                self._read_fail_count += 1
                self._make_read_failure_record(frame_index, path.name)
                self._emit_pid_lost()
            else:
                self._set_sequence_size(frame.shape[1], frame.shape[0])
                self.process_frame(
                    frame,
                    dataset_name,
                    frame_index,
                    file_name=path.name,
                    dt=dt,
                )
                frames_processed += 1

            if frame_interval > 0:
                elapsed = time.perf_counter() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        return frames_processed, self._read_fail_count

    def process_frame(
        self,
        frame_img,
        dataset_name: str,
        frame_index: int,
        file_name: str = "stream",
        dt: float | None = None,
    ):
        image_height, image_width = frame_img.shape[:2]
        dt = dt if dt is not None else (1.0 / self.config.fps)
        self._set_sequence_size(image_width, image_height)

        candidates, mask_clean = self.extractor.extract(frame_img)

        if self.config.matcher_mode == "legacy_largest2":
            record = self._build_legacy_frame_record(
                candidates,
                image_width,
                image_height,
                frame_index,
                file_name,
            )
            return None, candidates, mask_clean, record

        return self._process_frame_spatio(
            frame_img,
            dataset_name,
            frame_index,
            file_name,
            candidates,
            mask_clean,
            image_width,
            image_height,
            dt,
        )

    def _process_frame_spatio(
        self,
        frame_img,
        dataset_name: str,
        frame_index: int,
        file_name: str,
        candidates,
        mask_clean,
        image_width: int,
        image_height: int,
        dt: float,
    ):
        self.matcher.update_tracks(candidates, image_width, image_height, dt)

        match = self.matcher.find_best_pair()

        frame_bit = self.bit_extractor.bit_from_candidates(candidates)
        if match is not None:
            frame_bit = match.fused_bit

        is_valid = False
        packet: dict | None = None
        geom = None

        if match is not None:
            geom = self.geometry.compute_pair_geometry(
                match.led1, match.led2, image_width, image_height
            )
            raw_px_dist = geom.pixel_distance

            iqr_decision = self.rolling_iqr.evaluate(
                raw_px_dist,
                config=self.config,
                pattern_accuracy=match.pattern_accuracy,
                pair_correlation=match.pair_correlation,
                geometry_score=match.geometry_score,
                face_id=match.face_id,
                pattern=match.pattern,
            )
            if iqr_decision.accept:
                self.rolling_iqr.add_valid(raw_px_dist)
                is_valid = True

                est_dist = self.distance_model.estimate(raw_px_dist)
                dist_conf = self.distance_model.confidence(
                    raw_px_dist, match.pattern_accuracy
                )

                smoothed_dist = self.distance_lpf.update(est_dist)
                smoothed_err_x = self.error_x_lpf.update(geom.error_x)
                smoothed_err_y = self.error_y_lpf.update(geom.error_y)
                calc_error_norm = math.sqrt(geom.error_x ** 2 + geom.error_y ** 2)
                smoothed_err_norm = self.error_norm_lpf.update(calc_error_norm)

                packet = {
                    "valid": True,
                    "held": False,
                    "dataset": dataset_name,
                    "frame": frame_index,
                    "face_id": match.face_id,
                    "pattern_accuracy": match.pattern_accuracy,
                    "bit_error_rate": match.bit_error_rate,
                    "track_id_1": match.track_id_1,
                    "track_id_2": match.track_id_2,
                    "led1_x": geom.led1_x,
                    "led1_y": geom.led1_y,
                    "led2_x": geom.led2_x,
                    "led2_y": geom.led2_y,
                    "mid_x": geom.mid_x,
                    "mid_y": geom.mid_y,
                    "error_x": smoothed_err_x,
                    "error_y": smoothed_err_y,
                    "error_norm": smoothed_err_norm,
                    "ray_cam": [geom.ray_x, geom.ray_y, geom.ray_z],
                    "pixel_distance": raw_px_dist,
                    "estimated_distance": smoothed_dist,
                    "distance_confidence": dist_conf,
                    "total_area": match.led1.area + match.led2.area,
                    "bit": match.fused_bit,
                }

                self.last_valid_packet = packet
                self.invalid_frames_count = 0

        held = False
        if not is_valid:
            self.invalid_frames_count += 1
            if (
                self.invalid_frames_count <= self.config.max_hold_frames
                and self.last_valid_packet is not None
            ):
                packet = dict(self.last_valid_packet)
                packet["frame"] = frame_index
                packet["valid"] = False
                packet["held"] = True
                held = True
            else:
                self.distance_lpf.reset()
                self.error_norm_lpf.reset()
                self.error_x_lpf.reset()
                self.error_y_lpf.reset()
                packet = {
                    "valid": False,
                    "held": False,
                    "dataset": dataset_name,
                    "frame": frame_index,
                    "reason": "lost_or_outlier",
                }

        self._emit_pid_from_observation(
            packet,
            is_valid=is_valid,
            held=held,
        )

        record = self._spatio_frame_record(
            frame_index=frame_index,
            file_name=file_name,
            candidates=candidates,
            image_width=image_width,
            image_height=image_height,
            frame_bit=frame_bit,
            match=match,
            geom=geom,
            observation_valid=is_valid,
            held=held,
        )
        return packet, candidates, mask_clean, record

