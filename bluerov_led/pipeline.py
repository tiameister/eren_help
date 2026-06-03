"""LED tracking offline pipeline orchestrator."""

from __future__ import annotations

import json
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


class BackFacePipeline:
    """Orchestrates extract, decode, filter, calibrate, and packet build."""

    def __init__(
        self,
        config: VisionConfig | None = None,
        paths: ProjectPaths | None = None,
    ) -> None:
        self.config = config or VisionConfig()
        self.paths = paths or ProjectPaths()

        self.extractor = LedCandidateExtractor(self.config)
        self.bit_extractor = BitExtractor(self.config)
        self.legacy_selector = PairSelector(self.config)
        self.geometry = GeometryCalculator(self.config)
        self.decoder = TemporalDecoder(self.config)
        self.packet_builder = ObservationPacketBuilder(self.config)
        self.matcher = SpatioTemporalMatcher(self.config)

    def _reset_stateful_matcher(self) -> None:
        self.matcher.reset()

    def _process_frame_legacy(
        self,
        candidates,
        image_width: int,
        image_height: int,
    ) -> tuple:
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

        return (
            total_area,
            bit,
            pair_found,
            led1_x,
            led1_y,
            led2_x,
            led2_y,
            pixel_distance,
            mid_x,
            mid_y,
            error_x,
            error_y,
            ray_x,
            ray_y,
            ray_z,
            face_id,
            pattern,
            pattern_accuracy,
            0,
            None,
            None,
            None,
            None,
            None,
        )

    def _process_frame_spatio_temporal(
        self,
        candidates,
        image_width: int,
        image_height: int,
    ) -> tuple:
        total_area = self.bit_extractor.total_area(candidates)
        self.matcher.update_tracks(candidates, image_width, image_height)

        pair_found = 0
        led1_x = led1_y = led2_x = led2_y = None
        pixel_distance = None
        mid_x = mid_y = error_x = error_y = None
        ray_x = ray_y = ray_z = None
        face_id = None
        pattern = None
        pattern_accuracy = None
        active_track_count = self.matcher.tracker.active_track_count
        track_id_1 = track_id_2 = None
        pair_correlation = pair_score = geometry_score = None

        match = self.matcher.find_best_pair()
        bit = self.bit_extractor.bit_from_candidates(candidates)

        if match is not None:
            geom = self.geometry.compute_pair_geometry(
                match.led1, match.led2, image_width, image_height
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
            face_id = match.face_id
            pattern = match.pattern
            pattern_accuracy = match.pattern_accuracy
            active_track_count = match.active_track_count
            track_id_1 = match.track_id_1
            track_id_2 = match.track_id_2
            pair_correlation = match.pair_correlation
            pair_score = match.pair_score
            geometry_score = match.geometry_score
            bit = match.fused_bit

        return (
            total_area,
            bit,
            pair_found,
            led1_x,
            led1_y,
            led2_x,
            led2_y,
            pixel_distance,
            mid_x,
            mid_y,
            error_x,
            error_y,
            ray_x,
            ray_y,
            ray_z,
            face_id,
            pattern,
            pattern_accuracy,
            active_track_count,
            track_id_1,
            track_id_2,
            pair_correlation,
            pair_score,
            geometry_score,
        )

    def _process_frame(
        self,
        frame,
        frame_index: int,
        file_name: str,
        candidates,
    ) -> FrameRecord:
        image_height, image_width = frame.shape[:2]

        if self.config.matcher_mode == "legacy_largest2":
            fields = self._process_frame_legacy(
                candidates, image_width, image_height
            )
        else:
            fields = self._process_frame_spatio_temporal(
                candidates, image_width, image_height
            )

        (
            total_area,
            bit,
            pair_found,
            led1_x,
            led1_y,
            led2_x,
            led2_y,
            pixel_distance,
            mid_x,
            mid_y,
            error_x,
            error_y,
            ray_x,
            ray_y,
            ray_z,
            face_id,
            pattern,
            pattern_accuracy,
            active_track_count,
            track_id_1,
            track_id_2,
            pair_correlation,
            pair_score,
            geometry_score,
        ) = fields

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
            active_track_count=active_track_count,
            track_id_1=track_id_1,
            track_id_2=track_id_2,
            pair_correlation=pair_correlation,
            pair_score=pair_score,
            geometry_score=geometry_score,
            matcher_mode=self.config.matcher_mode,
        )

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

        self._reset_stateful_matcher()

        print("Matcher mode:", self.config.matcher_mode)
        print("Total frame count:", len(frame_paths))
        print(
            "Duration at 60 FPS:",
            len(frame_paths) / self.config.fps,
            "seconds",
        )
        print("Frames per bit:", self.config.frames_per_bit)

        records: list[FrameRecord] = []

        for frame_index, path in enumerate(frame_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue

            candidates, mask_clean = self.extractor.extract(frame)
            record = self._process_frame(
                frame, frame_index, path.name, candidates
            )
            records.append(record)

            if preview:
                tracks = (
                    self.matcher.tracker.list_tracks()
                    if self.config.matcher_mode != "legacy_largest2"
                    else None
                )
                self._draw_preview(frame, record, candidates, mask_clean, tracks)
                key = cv2.waitKey(int(1000 / self.config.fps))
                if key == ord("q"):
                    break

        if preview:
            cv2.destroyAllWindows()

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
    """True real-time framewise pipeline with Kalman tracking and UDP streaming."""
    def __init__(
        self,
        config: VisionConfig | None = None,
        distance_model_dict: dict | None = None,
        udp_ip: str | None = "127.0.0.1",
        udp_port: int | None = 5005
    ) -> None:
        self.config = config or VisionConfig()
        
        self.extractor = LedCandidateExtractor(self.config)
        self.matcher = SpatioTemporalMatcher(self.config)
        self.geometry = GeometryCalculator(self.config)
        self.packet_builder = ObservationPacketBuilder(self.config)
        self.distance_model = DistanceModel.from_summary_dict(distance_model_dict) if distance_model_dict else DistanceModel.fit()

        self.udp_sender = UdpSender(udp_ip, udp_port) if udp_ip else None
        
        # Filtering States
        self.rolling_iqr = RollingIQRFilter(self.config.rolling_iqr_window, self.config.outlier_distance_rejection_iqr_multiplier)
        self.distance_lpf = SignalSmoother1D(self.config.signal_1d_lpf_alpha)
        self.error_norm_lpf = SignalSmoother1D(self.config.signal_1d_lpf_alpha)
        self.error_x_lpf = SignalSmoother1D(self.config.signal_1d_lpf_alpha)
        self.error_y_lpf = SignalSmoother1D(self.config.signal_1d_lpf_alpha)
        
        # M-Frame Failsafe Holding
        self.last_valid_packet = None
        self.invalid_frames_count = 0
        self.udp_seq = 0

    def process_frame(self, frame_img, dataset_name: str, frame_index: int, dt: float | None = None):
        image_height, image_width = frame_img.shape[:2]
        dt = dt if dt is not None else (1.0 / self.config.fps)
        
        candidates, mask_clean = self.extractor.extract(frame_img)
        self.matcher.update_tracks(candidates, image_width, image_height, dt)
        
        match = self.matcher.find_best_pair()
        
        # Original bit calculation parity
        frame_bit = self.bit_extractor.bit_from_candidates(candidates)
        if match is not None:
            frame_bit = match.fused_bit
        
        is_valid = False
        packet = None
        
        if match is not None:
            geom = self.geometry.compute_pair_geometry(
                match.led1, match.led2, image_width, image_height
            )
            raw_px_dist = geom.pixel_distance
            
            # Online Outlier filtering
            if not self.rolling_iqr.is_outlier(raw_px_dist):
                self.rolling_iqr.add_valid(raw_px_dist)
                is_valid = True
                
                # Estimate distance
                est_dist = self.distance_model.estimate(raw_px_dist)
                dist_conf = self.distance_model.confidence(raw_px_dist, match.pattern_accuracy) if raw_px_dist is not None else 0.0
                
                # Apply 1D low pass filters for control outputs
                smoothed_dist = self.distance_lpf.update(est_dist)
                smoothed_err_x = self.error_x_lpf.update(geom.error_x)
                smoothed_err_y = self.error_y_lpf.update(geom.error_y)
                
                import math
                calc_error_norm = math.sqrt(geom.error_x**2 + geom.error_y**2)
                smoothed_err_norm = self.error_norm_lpf.update(calc_error_norm)
                
                packet = {
                    "valid": True,
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
                    "ray_cam": [geom.ray_x, geom.ray_y, geom.ray_z] if geom.ray_z else None,
                    "pixel_distance": raw_px_dist,
                    "estimated_distance": smoothed_dist,
                    "distance_confidence": dist_conf,
                    "total_area": match.led1.area + match.led2.area,
                    "bit": match.fused_bit,
                }
                
                self.last_valid_packet = packet
                self.invalid_frames_count = 0

        if not is_valid:
            # Hold Last Valid Logic
            self.invalid_frames_count += 1
            if self.invalid_frames_count <= self.config.max_hold_frames and self.last_valid_packet is not None:
                packet = dict(self.last_valid_packet)
                packet["frame"] = frame_index
            else:
                self.distance_lpf.reset()
                self.error_norm_lpf.reset()
                self.error_x_lpf.reset()
                self.error_y_lpf.reset()
                packet = {
                    "valid": False,
                    "dataset": dataset_name,
                    "frame": frame_index,
                    "reason": "lost_or_outlier"
                }
                
        if self.udp_sender is not None:
            self.udp_sender.send_packet(packet, self.udp_seq)
            self.udp_seq += 1
            
        # Also return a FrameRecord equivalent for validation tests
        fake_candidates_count = len(candidates)
        active_count = self.matcher.tracker.active_track_count
        
        record = FrameRecord(
            frame=frame_index,
            file="stream",
            candidate_count=fake_candidates_count,
            total_area=self.bit_extractor.total_area(candidates),
            bit=frame_bit,
            pair_found=1 if packet and packet.get("valid") else 0,
            led1_x=packet.get("led1_x") if packet else None,
            led1_y=packet.get("led1_y") if packet else None,
            led2_x=packet.get("led2_x") if packet else None,
            led2_y=packet.get("led2_y") if packet else None,
            pixel_distance=packet.get("pixel_distance") if packet else None,
            mid_x=packet.get("mid_x") if packet else None,
            mid_y=packet.get("mid_y") if packet else None,
            error_x=packet.get("error_x") if packet else None,
            error_y=packet.get("error_y") if packet else None,
            ray_x=packet.get("ray_cam")[0] if packet and packet.get("ray_cam") else None,
            ray_y=packet.get("ray_cam")[1] if packet and packet.get("ray_cam") else None,
            ray_z=packet.get("ray_cam")[2] if packet and packet.get("ray_cam") else None,
            camera_vertical_fov_deg=self.config.camera_vertical_fov_deg,
            image_width=image_width,
            image_height=image_height,
            face_id=packet.get("face_id") if packet else None,
            pattern=self.config.face_patterns.get(packet.get("face_id"), None) if packet and packet.get("face_id") else None,
            pattern_accuracy=packet.get("pattern_accuracy") if packet else None,
            active_track_count=active_count,
            track_id_1=packet.get("track_id_1") if packet else None,
            track_id_2=packet.get("track_id_2") if packet else None,
            pair_correlation=None, # Online we mainly care about decoding output
            pair_score=None,
            geometry_score=None,
            matcher_mode="streaming"
        )
        return packet, candidates, mask_clean, record

