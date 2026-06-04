#!/usr/bin/env python3
"""
Operator dashboard for the BlueROV LED pipeline (PySide6 wrapper).

Launches existing CLI scripts via QProcess; does not import vision modules.

Usage:
  python tools/gui_dashboard.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_JSON = PROJECT_ROOT / "outputs" / "validation" / "validation_summary.json"
CONSOLE_MAX_BLOCKS = 5000

PROGRESS_RE = re.compile(r"^PROGRESS:\s*(\d+)\s*/\s*(\d+)\s*$")
PHASE_RE = re.compile(r"^PHASE:\s*(\w+)\s*$")
VALIDATION_RESULT_RE = re.compile(
    r"^VALIDATION_RESULT:\s+"
    r"pair_recall=([\d.]+)\s+"
    r"face_acc=([\d.]+)\s+"
    r"temporal=([\d.]+)\s+"
    r"dist_mae=([\d.-]+)\s+"
    r"passed=([01])\s*$"
)

DARK_STYLESHEET = """
QMainWindow, QWidget#centralRoot {
    background-color: #1e1e1e;
    color: #e0e0e0;
}

QWidget#leftPanel, QWidget#rightPanel {
    background-color: #252525;
    border: 1px solid #333333;
    border-radius: 8px;
}

QLabel#panelTitle {
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 2px;
    color: #6a8caf;
    padding: 4px 2px 8px 2px;
}

QLabel#sectionHeader {
    font-size: 10px;
    font-weight: bold;
    letter-spacing: 1.5px;
    color: #666666;
    padding: 2px 0 4px 0;
}

QGroupBox {
    border: none;
    border-radius: 6px;
    margin-top: 14px;
    padding: 8px 4px 12px 4px;
    font-size: 11px;
    font-weight: bold;
    background-color: #222222;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 4px;
    padding: 0 4px;
    color: #9ecbff;
    font-size: 11px;
    font-weight: bold;
}

QComboBox, QLineEdit, QSpinBox {
    background-color: #2d2d2d;
    border: 1px solid #4a4a4a;
    border-radius: 4px;
    padding: 4px 8px;
    color: #e0e0e0;
    min-height: 20px;
    max-height: 28px;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    border: 1px solid #4a4a4a;
    selection-background-color: #3d8bfd;
}

QPushButton {
    background-color: #3d3d3d;
    color: #e0e0e0;
    border: 1px solid #4a4a4a;
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: 600;
    min-height: 18px;
}
QPushButton:hover { background-color: #4a4a4a; border-color: #5a5a5a; }
QPushButton:pressed { background-color: #333333; }
QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666666;
    border-color: #333333;
}

QPushButton#primaryValidation {
    background-color: #3d8bfd;
    border: none;
    color: #ffffff;
    padding: 10px 24px;
    font-size: 12px;
}
QPushButton#primaryValidation:hover { background-color: #5a9dff; }
QPushButton#primaryValidation:pressed { background-color: #2a6fd4; }
QPushButton#primaryValidation:disabled {
    background-color: #2a3a4a;
    color: #666666;
}

QPushButton#streamUdp {
    background-color: #1f6b45;
    border: none;
    color: #ffffff;
}
QPushButton#streamUdp:hover { background-color: #2ea86f; }
QPushButton#streamUdp:pressed { background-color: #185a38; }
QPushButton#streamUdp:disabled {
    background-color: #1a2e24;
    color: #555555;
}

QPushButton#streamReceiver {
    background-color: #2a5a9a;
    border: none;
    color: #ffffff;
}
QPushButton#streamReceiver:hover { background-color: #3d8bfd; }
QPushButton#streamReceiver:pressed { background-color: #1e4a7a; }
QPushButton#streamReceiver:disabled {
    background-color: #1a2838;
    color: #555555;
}

QPushButton#stopButton {
    background-color: #8b2e2e;
    border: none;
    color: #ffffff;
}
QPushButton#stopButton:hover { background-color: #c44; }
QPushButton#stopButton:pressed { background-color: #6b2222; }
QPushButton#stopButton:disabled {
    background-color: #2a1a1a;
    color: #555555;
}

QCheckBox {
    spacing: 6px;
    color: #b0b0b0;
    font-size: 11px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #4a4a4a;
    background-color: #2d2d2d;
}
QCheckBox::indicator:checked {
    background-color: #3d8bfd;
    border-color: #3d8bfd;
}

QProgressBar {
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    text-align: center;
    background-color: #1a1a1a;
    color: #aaaaaa;
    min-height: 20px;
    max-height: 22px;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:0,
        stop:0 #2a6fd4, stop:1 #3d8bfd
    );
    border-radius: 5px;
}

QFrame#metricCard {
    background-color: #2b2b2b;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
}
QLabel#metricCardLabel {
    color: #888888;
    font-size: 10px;
    font-weight: normal;
}
QLabel#metricCardValue {
    color: #f0f0f0;
    font-size: 22px;
    font-weight: bold;
}
QLabel#metricCardValuePass { color: #5cdb7a; }
QLabel#metricCardValueFail { color: #ff6b6b; }

QLabel#liveBadge {
    color: #ff4444;
    font-size: 12px;
    font-weight: bold;
    padding: 4px 10px;
    background-color: #3a1a1a;
    border: 1px solid #ff4444;
    border-radius: 4px;
}
QLabel#statusHint {
    color: #777777;
    font-size: 10px;
}

QPlainTextEdit#activityConsole {
    background-color: #0d0d0d;
    color: #b8b8b8;
    border: 1px solid #333333;
    border-radius: 6px;
    padding: 6px;
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
    selection-background-color: #3d8bfd;
}
"""


def apply_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)


def discover_datasets(project_root: Path) -> list[str]:
    datasets_dir = project_root / "datasets"
    if not datasets_dir.exists():
        return []

    names: list[str] = []
    for folder in sorted(datasets_dir.iterdir()):
        if folder.is_dir() and list(folder.glob("*.png")):
            names.append(folder.name)
    return names


def make_metric_card(title: str) -> tuple[QFrame, QLabel]:
    """Build a telemetry card: small gray label on top, large value below."""
    card = QFrame()
    card.setObjectName("metricCard")
    card.setMinimumHeight(72)

    layout = QVBoxLayout(card)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(4)

    label = QLabel(title)
    label.setObjectName("metricCardLabel")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    value = QLabel("—")
    value.setObjectName("metricCardValue")
    value.setAlignment(Qt.AlignmentFlag.AlignCenter)

    layout.addWidget(label)
    layout.addWidget(value)
    layout.addStretch()

    return card, value


class ProcessConsole:
    """Runs a CLI job via QProcess with optional stdout line hooks."""

    def __init__(
        self,
        tag: str,
        console: QPlainTextEdit,
        project_root: Path,
        on_finished=None,
        on_progress=None,
        on_phase=None,
        on_validation_result=None,
    ) -> None:
        self.tag = tag
        self.console = console
        self.project_root = project_root
        self.on_finished = on_finished
        self.on_progress = on_progress
        self.on_phase = on_phase
        self.on_validation_result = on_validation_result
        self._line_buffer = ""
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        env = self.process.processEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        self.process.setWorkingDirectory(str(project_root))
        self.process.readyReadStandardOutput.connect(self._on_ready_read)
        self.process.finished.connect(self._on_finished)
        self.process.errorOccurred.connect(self._on_error)

    @property
    def is_running(self) -> bool:
        return self.process.state() != QProcess.NotRunning

    def start(self, script_parts: list[str]) -> None:
        if self.is_running:
            self.append_line("Already running.")
            return

        self._line_buffer = ""
        self.process.setProgram(sys.executable)
        self.process.setArguments(script_parts)

        cmd_display = f"{sys.executable} {' '.join(script_parts)}"
        self.append_line(f"--- Running: {cmd_display} ---")
        self.process.start()

    def terminate(self) -> None:
        if self.is_running:
            self.append_line("Stopping...")
            self.process.terminate()

    def kill(self) -> None:
        if self.is_running:
            self.process.kill()

    def append_line(self, text: str) -> None:
        self.console.appendPlainText(f"[{self.tag}] {text}")
        self._trim_console()
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _trim_console(self) -> None:
        doc = self.console.document()
        if doc.blockCount() > CONSOLE_MAX_BLOCKS:
            cursor = self.console.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(
                cursor.MoveOperation.Down,
                cursor.MoveMode.KeepAnchor,
                doc.blockCount() - CONSOLE_MAX_BLOCKS,
            )
            cursor.removeSelectedText()
            cursor.deletePreviousChar()

    def _handle_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return

        progress_match = PROGRESS_RE.match(stripped)
        if progress_match and self.on_progress is not None:
            current = int(progress_match.group(1))
            total = int(progress_match.group(2))
            self.on_progress(current, total)
            return

        phase_match = PHASE_RE.match(stripped)
        if phase_match and self.on_phase is not None:
            self.on_phase(phase_match.group(1))
            return

        result_match = VALIDATION_RESULT_RE.match(stripped)
        if result_match and self.on_validation_result is not None:
            dist_raw = float(result_match.group(4))
            self.on_validation_result(
                {
                    "pair_recall": float(result_match.group(1)),
                    "face_acc": float(result_match.group(2)),
                    "temporal": float(result_match.group(3)),
                    "dist_mae": None if dist_raw < 0 else dist_raw,
                    "passed": result_match.group(5) == "1",
                }
            )
            return

        self.append_line(stripped)

    def _on_ready_read(self) -> None:
        chunk = self.process.readAllStandardOutput().data().decode(
            "utf-8", errors="replace"
        )
        self._line_buffer += chunk
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._handle_line(line.rstrip("\r"))

    def _on_finished(self, exit_code: int, _status) -> None:
        if self._line_buffer.strip():
            self._handle_line(self._line_buffer)
            self._line_buffer = ""
        self.append_line(f"Exited with code {exit_code}")
        if self.on_finished is not None:
            self.on_finished(self.tag, exit_code)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.append_line("Failed to start process.")
            QMessageBox.warning(
                None,
                "Process Error",
                f"Could not start [{self.tag}]. Check Python and script paths.",
            )


class PipelineDashboard(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BlueROV LED — Ground Control Station")
        self.resize(1100, 720)
        self.setMinimumSize(900, 600)

        self._processes: dict[str, ProcessConsole] = {}
        self._dataset_valid = False
        self._live_pulse_on = False

        central = QWidget()
        central.setObjectName("centralRoot")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # ── Left Panel: Control Tower ──────────────────────────────────────
        left_panel = QWidget()
        left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(8)

        left_title = QLabel("CONTROL TOWER")
        left_title.setObjectName("panelTitle")
        left_layout.addWidget(left_title)

        # STEP 1 — Data Source
        step1 = QGroupBox("STEP 1 — Data Source")
        step1_layout = QVBoxLayout(step1)
        step1_layout.setSpacing(8)

        row_a = QHBoxLayout()
        row_a.setSpacing(8)
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._update_enablement)
        row_a.addWidget(self.dataset_combo, stretch=1)

        self.mp4_btn = QPushButton("Select MP4")
        self.mp4_btn.setEnabled(False)
        self.mp4_btn.setToolTip("Coming in a future release")
        row_a.addWidget(self.mp4_btn)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_datasets)
        row_a.addWidget(refresh_btn)
        step1_layout.addLayout(row_a)

        row_b = QHBoxLayout()
        row_b.setSpacing(10)
        row_b.addWidget(QLabel("IP:"))
        self.ip_edit = QLineEdit("127.0.0.1")
        self.ip_edit.setMaximumWidth(120)
        row_b.addWidget(self.ip_edit)
        row_b.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(5005)
        self.port_spin.setMaximumWidth(90)
        row_b.addWidget(self.port_spin)
        self.force_reextract_cb = QCheckBox("Force re-extract")
        row_b.addWidget(self.force_reextract_cb)
        row_b.addStretch()
        step1_layout.addLayout(row_b)

        left_layout.addWidget(step1)

        # STEP 2 — Validation
        step2 = QGroupBox("STEP 2 — Offline Analysis & Validation")
        step2_layout = QVBoxLayout(step2)
        step2_layout.setSpacing(8)

        val_row = QHBoxLayout()
        self.validation_btn = QPushButton("Run Validation Suite")
        self.validation_btn.setObjectName("primaryValidation")
        self.validation_btn.clicked.connect(self._run_validation)
        val_row.addWidget(self.validation_btn)
        val_row.addStretch()
        step2_layout.addLayout(val_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%  ·  %v / %m frames")
        self.progress_bar.setVisible(False)
        step2_layout.addWidget(self.progress_bar)

        self.progress_status = QLabel("Ready.")
        self.progress_status.setObjectName("statusHint")
        step2_layout.addWidget(self.progress_status)

        left_layout.addWidget(step2)

        # STEP 3 — Live Stream
        step3 = QGroupBox("STEP 3 — Live PID Stream")
        step3_layout = QVBoxLayout(step3)
        step3_layout.setSpacing(8)

        live_row = QHBoxLayout()
        self.live_badge = QLabel("● LIVE")
        self.live_badge.setObjectName("liveBadge")
        self.live_badge.setVisible(False)
        live_row.addWidget(self.live_badge)
        live_row.addStretch()
        step3_layout.addLayout(live_row)

        hint = QLabel("Start the PID receiver first, then start the UDP stream.")
        hint.setObjectName("statusHint")
        step3_layout.addWidget(hint)

        stream_row = QHBoxLayout()
        stream_row.setSpacing(8)
        self.receiver_btn = QPushButton("1. Start Receiver")
        self.receiver_btn.setObjectName("streamReceiver")
        self.receiver_btn.clicked.connect(self._toggle_receiver)
        stream_row.addWidget(self.receiver_btn, stretch=1)

        self.stream_btn = QPushButton("2. Start UDP Stream")
        self.stream_btn.setObjectName("streamUdp")
        self.stream_btn.clicked.connect(self._run_stream)
        stream_row.addWidget(self.stream_btn, stretch=1)

        self.stop_btn = QPushButton("Stop All")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_all)
        stream_row.addWidget(self.stop_btn, stretch=1)
        step3_layout.addLayout(stream_row)

        self.no_pacing_cb = QCheckBox("Max speed (--no-pacing)")
        step3_layout.addWidget(self.no_pacing_cb)

        left_layout.addWidget(step3)
        left_layout.addStretch()

        # ── Right Panel: Telemetry & Terminal ──────────────────────────────
        right_panel = QWidget()
        right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(8)

        right_title = QLabel("TELEMETRY & LOGS")
        right_title.setObjectName("panelTitle")
        right_layout.addWidget(right_title)

        telemetry_header = QLabel("LAST VALIDATION")
        telemetry_header.setObjectName("sectionHeader")
        right_layout.addWidget(telemetry_header)

        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(8)

        self.metric_status_card, self.metric_status = make_metric_card("Status")
        self.metric_pair_recall_card, self.metric_pair_recall = make_metric_card("Pair Recall")
        self.metric_face_acc_card, self.metric_face_acc = make_metric_card("Face Accuracy")
        self.metric_temporal_card, self.metric_temporal = make_metric_card("Temporal Decode")
        self.metric_dist_mae_card, self.metric_dist_mae = make_metric_card("Distance MAE")

        cards = [
            self.metric_status_card,
            self.metric_pair_recall_card,
            self.metric_face_acc_card,
            self.metric_temporal_card,
            self.metric_dist_mae_card,
        ]
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        for card, (row, col) in zip(cards, positions):
            metrics_grid.addWidget(card, row, col)

        metrics_grid.setColumnStretch(0, 1)
        metrics_grid.setColumnStretch(1, 1)
        metrics_grid.setColumnStretch(2, 1)
        metrics_grid.setRowStretch(2, 1)
        right_layout.addLayout(metrics_grid)

        log_header = QLabel("ACTIVITY LOG")
        log_header.setObjectName("sectionHeader")
        right_layout.addWidget(log_header)

        self.console = QPlainTextEdit()
        self.console.setObjectName("activityConsole")
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Process output will appear here…")
        mono = QFont("Consolas")
        if not mono.exactMatch():
            mono = QFont("Courier New")
        mono.setStyleHint(QFont.Monospace)
        self.console.setFont(mono)
        right_layout.addWidget(self.console, stretch=1)

        root.addWidget(left_panel, stretch=4)
        root.addWidget(right_panel, stretch=6)

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(500)
        self._live_timer.timeout.connect(self._pulse_live_badge)

        self._init_process_slots()
        self._refresh_datasets()
        self._append_system("Ground control station online.")

    def _init_process_slots(self) -> None:
        self._processes["validation"] = ProcessConsole(
            tag="validation",
            console=self.console,
            project_root=PROJECT_ROOT,
            on_finished=self._on_process_finished,
            on_progress=self._on_extract_progress,
            on_phase=self._on_validation_phase,
            on_validation_result=self._on_validation_result_line,
        )
        self._processes["stream"] = ProcessConsole(
            tag="stream",
            console=self.console,
            project_root=PROJECT_ROOT,
            on_finished=self._on_process_finished,
        )
        self._processes["receiver"] = ProcessConsole(
            tag="receiver",
            console=self.console,
            project_root=PROJECT_ROOT,
            on_finished=self._on_process_finished,
        )

    def _append_system(self, text: str) -> None:
        self.console.appendPlainText(f"[system] {text}")

    def _refresh_datasets(self) -> None:
        previous = self.dataset_combo.currentText()
        names = discover_datasets(PROJECT_ROOT)

        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        if names:
            self.dataset_combo.addItems(names)
            if previous in names:
                self.dataset_combo.setCurrentText(previous)
            self._dataset_valid = True
            self._append_system(f"Found {len(names)} dataset(s).")
        else:
            self.dataset_combo.addItem("(no PNG datasets found)")
            self._dataset_valid = False
            self._append_system("No PNG datasets under datasets/.")
        self.dataset_combo.blockSignals(False)
        self._update_enablement()

    def _selected_dataset(self) -> str | None:
        if not self._dataset_valid:
            return None
        text = self.dataset_combo.currentText().strip()
        if not text or text.startswith("("):
            return None
        return text

    def _update_enablement(self) -> None:
        has_dataset = self._selected_dataset() is not None
        validation_running = self._processes["validation"].is_running
        stream_running = self._processes["stream"].is_running
        receiver_running = self._processes["receiver"].is_running

        self.validation_btn.setEnabled(has_dataset and not validation_running)
        step3_enabled = has_dataset and not validation_running
        self.stream_btn.setEnabled(step3_enabled and not stream_running)
        self.receiver_btn.setEnabled(step3_enabled or receiver_running)
        self.stop_btn.setEnabled(stream_running or receiver_running)

    def _validate_network(self) -> bool:
        if not self.ip_edit.text().strip():
            QMessageBox.warning(self, "Invalid Input", "Target IP cannot be empty.")
            return False
        return True

    def _reset_validation_progress(self) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_status.setText("Starting validation…")

    def _on_extract_progress(self, current: int, total: int) -> None:
        if total <= 0:
            return
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.progress_bar.setFormat("%p%  ·  %v / %m frames")
        self.progress_status.setText(f"Extracting frame {current} of {total}…")

    def _on_validation_phase(self, phase: str) -> None:
        if phase == "metrics":
            self.progress_bar.setRange(0, 0)
            self.progress_status.setText("Computing decode & metrics…")

    def _on_validation_result_line(self, metrics: dict) -> None:
        self._apply_metrics(metrics)

    def _set_metric_value(self, label: QLabel, text: str, *, pass_fail: bool | None = None) -> None:
        label.setText(text)
        if pass_fail is True:
            label.setObjectName("metricCardValuePass")
        elif pass_fail is False:
            label.setObjectName("metricCardValueFail")
        else:
            label.setObjectName("metricCardValue")
        label.style().unpolish(label)
        label.style().polish(label)

    def _apply_metrics(self, metrics: dict) -> None:
        passed = metrics.get("passed", False)
        self._set_metric_value(
            self.metric_status,
            "PASS" if passed else "FAIL",
            pass_fail=passed,
        )
        self._set_metric_value(
            self.metric_pair_recall,
            f"{metrics.get('pair_recall', 0):.3f}",
        )
        self._set_metric_value(
            self.metric_face_acc,
            f"{metrics.get('face_acc', 0):.3f}",
        )
        self._set_metric_value(
            self.metric_temporal,
            f"{metrics.get('temporal', 0):.3f}",
        )

        dist_mae = metrics.get("dist_mae")
        if dist_mae is None:
            self._set_metric_value(self.metric_dist_mae, "n/a")
        else:
            self._set_metric_value(self.metric_dist_mae, f"{dist_mae:.3f}")

    def _load_metrics_from_json(self, dataset: str) -> bool:
        if not VALIDATION_JSON.exists():
            return False
        try:
            with VALIDATION_JSON.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False

        for entry in payload.get("results", []):
            if entry.get("dataset") == dataset:
                self._apply_metrics(
                    {
                        "pair_recall": entry.get("pair_recall_on_frames", 0),
                        "face_acc": entry.get("face_id_accuracy_on_pairs", 0),
                        "temporal": entry.get("temporal_decode_accuracy", 0),
                        "dist_mae": entry.get("distance_mae"),
                        "passed": entry.get("passed", False),
                    }
                )
                return True
        return False

    def _run_validation(self) -> None:
        dataset = self._selected_dataset()
        if dataset is None:
            QMessageBox.warning(self, "No Dataset", "Select a valid dataset first.")
            return

        self._reset_validation_progress()
        self._update_enablement()

        args = [str(PROJECT_ROOT / "run_tests.py"), "--dataset", dataset]
        if self.force_reextract_cb.isChecked():
            args.append("--force-reextract")

        self._processes["validation"].start(args)

    def _run_stream(self) -> None:
        dataset = self._selected_dataset()
        if dataset is None:
            QMessageBox.warning(self, "No Dataset", "Select a valid dataset first.")
            return
        if not self._validate_network():
            return

        args = [
            str(PROJECT_ROOT / "main.py"),
            "stream-udp",
            "--dataset",
            dataset,
            "--ip",
            self.ip_edit.text().strip(),
            "--port",
            str(self.port_spin.value()),
        ]
        if self.no_pacing_cb.isChecked():
            args.append("--no-pacing")

        self._processes["stream"].start(args)
        self._set_stream_live(True)

    def _toggle_receiver(self) -> None:
        receiver = self._processes["receiver"]
        if receiver.is_running:
            receiver.terminate()
            self.receiver_btn.setText("1. Start Receiver")
            self._update_enablement()
            return

        if not self._validate_network():
            return

        args = [
            str(PROJECT_ROOT / "tools" / "eren_pid_receiver.py"),
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port_spin.value()),
        ]
        receiver.start(args)
        self.receiver_btn.setText("Stop Receiver")
        self._update_enablement()

    def _stop_all(self) -> None:
        for proc in self._processes.values():
            proc.terminate()
        self.receiver_btn.setText("1. Start Receiver")
        self._set_stream_live(False)
        self._append_system("Stop requested for all processes.")

    def _set_stream_live(self, active: bool) -> None:
        if active:
            self.live_badge.setVisible(True)
            self._live_pulse_on = True
            self._live_timer.start()
            self.stream_btn.setText("Streaming…")
            self.stream_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        else:
            self._live_timer.stop()
            self.live_badge.setVisible(False)
            self.stream_btn.setText("2. Start UDP Stream")
            self._update_enablement()

    def _pulse_live_badge(self) -> None:
        self._live_pulse_on = not self._live_pulse_on
        color = "#ff4444" if self._live_pulse_on else "#aa2222"
        self.live_badge.setStyleSheet(
            f"color: {color}; font-size: 12px; font-weight: bold; "
            f"padding: 4px 10px; background-color: #3a1a1a; "
            f"border: 1px solid {color}; border-radius: 4px;"
        )

    def _on_process_finished(self, tag: str, exit_code: int) -> None:
        if tag == "validation":
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_status.setText(
                "Validation complete." if exit_code == 0 else "Validation finished with errors."
            )
            dataset = self._selected_dataset()
            if dataset and not self._load_metrics_from_json(dataset):
                self._append_system(
                    "Could not load metrics from validation_summary.json"
                )
            self._update_enablement()

        if tag == "stream":
            self._set_stream_live(False)

        if tag == "receiver":
            self.receiver_btn.setText("1. Start Receiver")
            self._update_enablement()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._live_timer.stop()
        for proc in self._processes.values():
            if proc.is_running:
                proc.kill()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = PipelineDashboard()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
