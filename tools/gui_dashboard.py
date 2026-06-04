#!/usr/bin/env python3
"""
Minimal PySide6 control dashboard for the BlueROV LED pipeline.

Wrapper only — launches existing CLI scripts via QProcess without importing
core vision modules.

Usage:
  python tools/gui_dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
from PySide6.QtGui import QCloseEvent, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONSOLE_MAX_BLOCKS = 5000

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
}
QTabWidget::pane {
    border: 1px solid #3a3a3a;
    background-color: #2d2d2d;
}
QTabBar::tab {
    background-color: #2d2d2d;
    color: #b0b0b0;
    padding: 8px 16px;
    border: 1px solid #3a3a3a;
}
QTabBar::tab:selected {
    background-color: #3a3a3a;
    color: #e0e0e0;
}
QTabBar::tab:disabled {
    color: #555555;
}
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QComboBox, QLineEdit, QSpinBox {
    background-color: #2d2d2d;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    padding: 4px 8px;
    color: #e0e0e0;
}
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled {
    color: #777777;
}
QPushButton {
    background-color: #3d8bfd;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 14px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #5a9dff;
}
QPushButton:pressed {
    background-color: #2a6fd4;
}
QPushButton:disabled {
    background-color: #555555;
    color: #888888;
}
QPushButton#stopButton {
    background-color: #c44;
}
QPushButton#stopButton:hover {
    background-color: #d55;
}
QCheckBox {
    spacing: 6px;
}
QPlainTextEdit {
    background-color: #0d0d0d;
    color: #c0c0c0;
    border: 1px solid #3a3a3a;
    font-family: Consolas, "Courier New", monospace;
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


class ProcessConsole:
    """Runs one CLI job via QProcess and streams output to the shared console."""

    def __init__(
        self,
        tag: str,
        console: QPlainTextEdit,
        project_root: Path,
        on_finished=None,
    ) -> None:
        self.tag = tag
        self.console = console
        self.project_root = project_root
        self.on_finished = on_finished
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
            self.append_line(f"[{self.tag}] Already running.")
            return

        program = sys.executable
        self.process.setProgram(program)
        self.process.setArguments(script_parts)

        cmd_display = f"{program} {' '.join(script_parts)}"
        self.append_line(f"--- [{self.tag}] Running: {cmd_display} ---")
        self.process.start()

    def terminate(self) -> None:
        if self.is_running:
            self.append_line(f"--- [{self.tag}] Stopping... ---")
            self.process.terminate()

    def kill(self) -> None:
        if self.is_running:
            self.process.kill()

    def append_line(self, text: str) -> None:
        self.console.appendPlainText(text)
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

    def _on_ready_read(self) -> None:
        data = self.process.readAllStandardOutput().data().decode(
            "utf-8", errors="replace"
        )
        for line in data.splitlines():
            if line.strip():
                self.append_line(f"[{self.tag}] {line}")

    def _on_finished(self, exit_code: int, _status) -> None:
        self.append_line(f"--- [{self.tag}] Exited code {exit_code} ---")
        if self.on_finished is not None:
            self.on_finished(self.tag, exit_code)

    def _on_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.append_line(f"--- [{self.tag}] Failed to start process ---")
            QMessageBox.warning(
                None,
                "Process Error",
                f"Could not start [{self.tag}]. Check Python and script paths.",
            )


class PipelineDashboard(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BlueROV LED Control Dashboard")
        self.resize(900, 640)
        self.setMinimumSize(720, 480)

        self._processes: dict[str, ProcessConsole] = {}

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        pipeline_tab = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_tab)

        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.setMinimumWidth(280)
        dataset_row.addWidget(self.dataset_combo, stretch=1)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_datasets)
        dataset_row.addWidget(self.refresh_btn)
        pipeline_layout.addLayout(dataset_row)

        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        checks_row = QHBoxLayout()
        self.force_reextract_cb = QCheckBox("Force re-extract (--force-reextract)")
        self.no_pacing_cb = QCheckBox("Max speed stream (--no-pacing)")
        checks_row.addWidget(self.force_reextract_cb)
        checks_row.addWidget(self.no_pacing_cb)
        checks_row.addStretch()
        options_layout.addLayout(checks_row)

        network_row = QHBoxLayout()
        network_row.addWidget(QLabel("Target IP:"))
        self.ip_edit = QLineEdit("127.0.0.1")
        self.ip_edit.setMaximumWidth(160)
        network_row.addWidget(self.ip_edit)
        network_row.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(5005)
        network_row.addWidget(self.port_spin)
        network_row.addStretch()
        options_layout.addLayout(network_row)

        pipeline_layout.addWidget(options_group)

        actions_row = QHBoxLayout()
        self.validation_btn = QPushButton("Run Validation Suite")
        self.validation_btn.clicked.connect(self._run_validation)
        actions_row.addWidget(self.validation_btn)

        self.stream_btn = QPushButton("Start UDP Stream")
        self.stream_btn.clicked.connect(self._run_stream)
        actions_row.addWidget(self.stream_btn)

        self.receiver_btn = QPushButton("Start Eren's PID Receiver")
        self.receiver_btn.clicked.connect(self._toggle_receiver)
        actions_row.addWidget(self.receiver_btn)

        self.stop_btn = QPushButton("Stop Active")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.clicked.connect(self._stop_all)
        actions_row.addWidget(self.stop_btn)

        pipeline_layout.addLayout(actions_row)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Process output will appear here...")
        font = QFont("Consolas")
        if not font.exactMatch():
            font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        self.console.setFont(font)
        pipeline_layout.addWidget(self.console, stretch=1)

        self.tabs.addTab(pipeline_tab, "Pipeline Control")

        mp4_tab = QWidget()
        mp4_layout = QVBoxLayout(mp4_tab)
        mp4_layout.addStretch()
        mp4_label = QLabel("MP4 video processing — not implemented yet.")
        mp4_label.setAlignment(Qt.AlignCenter)
        mp4_label.setStyleSheet("color: #888888; font-size: 14px;")
        mp4_layout.addWidget(mp4_label)
        mp4_btn = QPushButton("Process MP4 Video")
        mp4_btn.setEnabled(False)
        mp4_btn.setMaximumWidth(220)
        mp4_layout.addWidget(mp4_btn, alignment=Qt.AlignCenter)
        mp4_layout.addStretch()
        self.tabs.addTab(mp4_tab, "Process MP4 Video")
        self.tabs.setTabEnabled(1, False)

        self._init_process_slots()
        self._refresh_datasets()
        self.append_system("Dashboard ready. Project root: " + str(PROJECT_ROOT))

    def _init_process_slots(self) -> None:
        for tag in ("validation", "stream", "receiver"):
            self._processes[tag] = ProcessConsole(
                tag=tag,
                console=self.console,
                project_root=PROJECT_ROOT,
                on_finished=self._on_process_finished,
            )

    def append_system(self, text: str) -> None:
        self.console.appendPlainText(f"[system] {text}")
        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _refresh_datasets(self) -> None:
        previous = self.dataset_combo.currentText()
        names = discover_datasets(PROJECT_ROOT)

        self.dataset_combo.clear()
        if names:
            self.dataset_combo.addItems(names)
            if previous in names:
                self.dataset_combo.setCurrentText(previous)
            self._set_actions_enabled(True)
            self.append_system(f"Found {len(names)} dataset(s).")
        else:
            self.dataset_combo.addItem("(no PNG datasets found)")
            self._set_actions_enabled(False)
            self.append_system("No PNG datasets under datasets/.")

    def _set_actions_enabled(self, enabled: bool) -> None:
        self.validation_btn.setEnabled(enabled)
        self.stream_btn.setEnabled(enabled)

    def _selected_dataset(self) -> str | None:
        text = self.dataset_combo.currentText().strip()
        if not text or text.startswith("("):
            return None
        return text

    def _validate_network(self) -> bool:
        ip = self.ip_edit.text().strip()
        if not ip:
            QMessageBox.warning(self, "Invalid Input", "Target IP cannot be empty.")
            return False
        return True

    def _run_validation(self) -> None:
        dataset = self._selected_dataset()
        if dataset is None:
            QMessageBox.warning(self, "No Dataset", "Select a valid dataset first.")
            return

        args = [
            str(PROJECT_ROOT / "run_tests.py"),
            "--dataset",
            dataset,
        ]
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

    def _toggle_receiver(self) -> None:
        receiver = self._processes["receiver"]
        if receiver.is_running:
            receiver.terminate()
            self.receiver_btn.setText("Start Eren's PID Receiver")
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

    def _stop_all(self) -> None:
        for proc in self._processes.values():
            proc.terminate()
        self.receiver_btn.setText("Start Eren's PID Receiver")
        self.append_system("Stop requested for all active processes.")

    def _on_process_finished(self, tag: str, _exit_code: int) -> None:
        if tag == "receiver":
            self.receiver_btn.setText("Start Eren's PID Receiver")

    def closeEvent(self, event: QCloseEvent) -> None:
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
