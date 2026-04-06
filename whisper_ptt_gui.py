#!/usr/bin/env python3
"""
Whisper-PTT GUI: system tray app with recording overlay.
Wraps the console-mode core (whisper_ptt_cuda / whisper_ptt_apple_silicon).
"""

import sys
import os
import time
import subprocess
import collections
import threading
import queue

from PySide6.QtWidgets import (
    QApplication, QSystemTrayIcon, QMenu, QWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, QLineEdit,
    QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit,
    QPushButton, QGroupBox, QFormLayout, QMessageBox,
)
from PySide6.QtGui import QIcon, QPainter, QColor, QPen, QPainterPath, QAction
from PySide6.QtCore import (
    Qt, QTimer, QMetaObject, Slot, Signal, QObject, QPropertyAnimation,
    QEasingCurve, QPoint,
)

# ---------------------------------------------------------------------------
# Platform detection and core import
# ---------------------------------------------------------------------------

_script_dir = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "darwin":
    import whisper_ptt_apple_silicon as core
else:
    import whisper_ptt_cuda as core


# ---------------------------------------------------------------------------
# AudioBridge: thread-safe bridge between core callbacks and Qt signals
# ---------------------------------------------------------------------------

class AudioBridge(QObject):
    """Receives callbacks from core worker threads and re-emits as Qt signals."""
    recording_started = Signal()
    recording_stopped = Signal()
    processing_started = Signal()
    transcription_done = Signal(str)
    spellcheck_started = Signal()
    spellcheck_done = Signal(str, bool)  # text, changed
    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self._audio_levels = collections.deque(maxlen=128)
        self._text_queue = queue.Queue()
        self._error_queue = queue.Queue()
        self._sc_queue = queue.Queue()

    def on_audio_level(self, peak):
        """Called from prebuffer thread. Writes to deque (thread-safe)."""
        self._audio_levels.append(peak)

    def on_event(self, event, data):
        """Called from various core threads. Marshals to Qt main thread."""
        if event == "recording_started":
            QMetaObject.invokeMethod(self, "_emit_recording_started", Qt.QueuedConnection)
        elif event == "recording_stopped":
            QMetaObject.invokeMethod(self, "_emit_recording_stopped", Qt.QueuedConnection)
        elif event == "processing_started":
            QMetaObject.invokeMethod(self, "_emit_processing_started", Qt.QueuedConnection)
        elif event == "transcription_done":
            self._text_queue.put(data.get("text", ""))
            QMetaObject.invokeMethod(self, "_emit_transcription_done", Qt.QueuedConnection)
        elif event == "spellcheck_started":
            QMetaObject.invokeMethod(self, "_emit_spellcheck_started", Qt.QueuedConnection)
        elif event == "spellcheck_done":
            text = data.get("text", "") if data else ""
            changed = data.get("changed", False) if data else False
            self._sc_queue.put((text, changed))
            QMetaObject.invokeMethod(self, "_emit_spellcheck_done", Qt.QueuedConnection)
        elif event == "error":
            self._error_queue.put(data.get("message", "Unknown error"))
            QMetaObject.invokeMethod(self, "_emit_error", Qt.QueuedConnection)

    @Slot()
    def _emit_recording_started(self):
        self.recording_started.emit()

    @Slot()
    def _emit_recording_stopped(self):
        self.recording_stopped.emit()

    @Slot()
    def _emit_processing_started(self):
        self.processing_started.emit()

    @Slot()
    def _emit_transcription_done(self):
        try:
            text = self._text_queue.get_nowait()
        except queue.Empty:
            return
        self.transcription_done.emit(text)

    @Slot()
    def _emit_spellcheck_started(self):
        self.spellcheck_started.emit()

    @Slot()
    def _emit_spellcheck_done(self):
        try:
            text, changed = self._sc_queue.get_nowait()
        except queue.Empty:
            return
        self.spellcheck_done.emit(text, changed)

    @Slot()
    def _emit_error(self):
        try:
            msg = self._error_queue.get_nowait()
        except queue.Empty:
            return
        self.error_occurred.emit(msg)

    def drain_levels(self):
        """Read all pending audio levels. Called from Qt main thread timer."""
        levels = []
        while self._audio_levels:
            try:
                levels.append(self._audio_levels.popleft())
            except IndexError:
                break
        return levels


# ---------------------------------------------------------------------------
# RecordingOverlay: frameless translucent waveform window
# ---------------------------------------------------------------------------

class RecordingOverlay(QWidget):
    """Floating overlay showing audio waveform during recording."""

    WIDTH = 340
    HEIGHT = 70

    def __init__(self, bridge):
        super().__init__()
        self._bridge = bridge
        self._waveform = collections.deque(maxlen=100)
        self._recording_start = 0.0
        self._drag_pos = None

        # Window flags: frameless, always on top, tool window (no taskbar entry)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(self.WIDTH, self.HEIGHT)

        # Default position: center-x, above taskbar
        self._default_pos = None  # computed on first show

        # Waveform update timer (~30fps)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._update_waveform)

        # Fade-in animation
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self._fade_anim.setDuration(150)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Load saved position from env
        self._load_position()

    def _load_position(self):
        """Load saved overlay position from env vars."""
        try:
            x = int(os.environ.get("WHISPER_PTT_OVERLAY_X", ""))
            y = int(os.environ.get("WHISPER_PTT_OVERLAY_Y", ""))
            self._saved_pos = QPoint(x, y)
        except (ValueError, TypeError):
            self._saved_pos = None

    def _compute_default_pos(self):
        """Center horizontally, ~1cm above taskbar."""
        screen = QApplication.primaryScreen()
        if not screen:
            return QPoint(100, 100)
        avail = screen.availableGeometry()
        x = avail.x() + (avail.width() - self.WIDTH) // 2
        # ~38px above bottom of available area (approx 1cm at 96 DPI)
        dpi = screen.logicalDotsPerInch()
        margin = int(dpi * 0.4)  # ~1cm
        y = avail.y() + avail.height() - self.HEIGHT - margin
        return QPoint(x, y)

    def show_overlay(self):
        """Show with fade-in and start waveform updates."""
        self._waveform.clear()
        self._recording_start = time.time()

        if self._saved_pos:
            self.move(self._saved_pos)
        else:
            if not self._default_pos:
                self._default_pos = self._compute_default_pos()
            self.move(self._default_pos)

        self.setWindowOpacity(0.0)
        self.show()
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()
        self._timer.start()

    def hide_overlay(self):
        """Hide and stop waveform updates."""
        self._timer.stop()
        self.hide()
        # Save position
        self._saved_pos = self.pos()
        self._save_position()

    def _save_position(self):
        """Persist overlay position to env (will be saved to .env by settings)."""
        if self._saved_pos:
            os.environ["WHISPER_PTT_OVERLAY_X"] = str(self._saved_pos.x())
            os.environ["WHISPER_PTT_OVERLAY_Y"] = str(self._saved_pos.y())

    def _update_waveform(self):
        """Poll audio levels from bridge and trigger repaint."""
        levels = self._bridge.drain_levels()
        if levels:
            # Normalize to 0.0-1.0 (int16 max = 32768)
            for lv in levels:
                self._waveform.append(min(lv / 32768.0, 1.0))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background: rounded rectangle, semi-transparent dark
        bg = QColor(30, 30, 30, 200)
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, self.WIDTH, self.HEIGHT, 12, 12)

        # Waveform area
        wave_left = 15
        wave_right = self.WIDTH - 15
        wave_top = 10
        wave_bottom = self.HEIGHT - 28
        wave_mid = (wave_top + wave_bottom) / 2
        wave_width = wave_right - wave_left
        wave_height = wave_bottom - wave_top

        if len(self._waveform) > 1:
            # Draw waveform as mirrored curve
            pen = QPen(QColor(100, 200, 255), 2)
            p.setPen(pen)

            path = QPainterPath()
            points = list(self._waveform)
            n = len(points)
            step = wave_width / max(n - 1, 1)

            # Top half
            path.moveTo(wave_left, wave_mid)
            for i, val in enumerate(points):
                x = wave_left + i * step
                y = wave_mid - val * (wave_height / 2) * 1.62
                path.lineTo(x, y)

            # Bottom half (mirror)
            for i in range(n - 1, -1, -1):
                x = wave_left + i * step
                y = wave_mid + points[i] * (wave_height / 2) * 1.62
                path.lineTo(x, y)

            path.closeSubpath()

            # Fill with gradient-like translucent
            p.setPen(Qt.NoPen)
            fill = QColor(100, 200, 255, 60)
            p.setBrush(fill)
            p.drawPath(path)

            # Stroke the top line
            p.setBrush(Qt.NoBrush)
            p.setPen(pen)
            stroke_path = QPainterPath()
            stroke_path.moveTo(wave_left, wave_mid)
            for i, val in enumerate(points):
                x = wave_left + i * step
                y = wave_mid - val * (wave_height / 2) * 1.62
                stroke_path.lineTo(x, y)
            p.drawPath(stroke_path)

            # Bottom stroke
            stroke_bot = QPainterPath()
            stroke_bot.moveTo(wave_left, wave_mid)
            for i, val in enumerate(points):
                x = wave_left + i * step
                y = wave_mid + val * (wave_height / 2) * 1.62
                stroke_bot.lineTo(x, y)
            p.drawPath(stroke_bot)
        else:
            # Flat line when no data
            p.setPen(QPen(QColor(100, 200, 255, 100), 1))
            p.drawLine(int(wave_left), int(wave_mid), int(wave_right), int(wave_mid))

        # Recording text and timer
        elapsed = time.time() - self._recording_start if self._recording_start else 0
        text = f"Recording...  {elapsed:.1f}s"
        p.setPen(QColor(200, 200, 200))
        p.drawText(wave_left, self.HEIGHT - 8, text)

        p.end()

    # --- Drag support ---
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event):
        if self._drag_pos and event.buttons() & Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        self._saved_pos = self.pos()


# ---------------------------------------------------------------------------
# Autostart (Windows registry / macOS launchd)
# ---------------------------------------------------------------------------

_AUTOSTART_REG_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
_AUTOSTART_NAME = "WhisperPTT"


def _get_autostart():
    """Check if autostart is enabled."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _AUTOSTART_REG_KEY, 0, winreg.KEY_READ)
            try:
                winreg.QueryValueEx(key, _AUTOSTART_NAME)
                return True
            except FileNotFoundError:
                return False
            finally:
                winreg.CloseKey(key)
        except Exception:
            return False
    return False


def _set_autostart(enabled):
    """Enable or disable autostart."""
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _AUTOSTART_REG_KEY, 0, winreg.KEY_SET_VALUE)
            try:
                if enabled:
                    # Use pythonw to avoid console window
                    pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
                    if not os.path.isfile(pythonw):
                        pythonw = sys.executable
                    gui_script = os.path.join(_script_dir, "whisper_ptt_gui.py")
                    cmd = f'"{pythonw}" "{gui_script}" --autostart'
                    winreg.SetValueEx(key, _AUTOSTART_NAME, 0, winreg.REG_SZ, cmd)
                else:
                    try:
                        winreg.DeleteValue(key, _AUTOSTART_NAME)
                    except FileNotFoundError:
                        pass
            finally:
                winreg.CloseKey(key)
        except Exception as e:
            print(f"Autostart error: {e}")


# ---------------------------------------------------------------------------
# SettingsDialog
# ---------------------------------------------------------------------------

class SettingsDialog(QDialog):
    """Settings dialog reading/writing .env file."""

    # Settings that require app restart
    RESTART_KEYS = {"WHISPER_MODEL", "SAMPLE_RATE", "CHUNK_SIZE", "HOTKEY", "SPELLCHECK_HOTKEY"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Whisper PTT Settings")
        self.setMinimumWidth(500)
        self._widgets = {}
        self._init_ui()
        self._load_values()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        tabs = QTabWidget()

        # Whisper tab
        whisper_tab = QWidget()
        wf = QFormLayout(whisper_tab)
        self._add_text(wf, "WHISPER_MODEL", "Model:")
        self._add_combo(wf, "WHISPER_LANGUAGE", "Language:", [
            "en", "ru", "de", "fr", "es", "it", "pt", "ja", "ko", "zh",
        ])
        self._add_text(wf, "WHISPER_INITIAL_PROMPT", "Initial prompt:")
        self._add_spin(wf, "WHISPER_NO_REPEAT_NGRAM_SIZE", "No-repeat ngram size:", 0, 10, 1)
        self._add_dspin(wf, "WHISPER_REPETITION_PENALTY", "Repetition penalty:", 1.0, 2.0, 0.05)
        self._add_dspin(wf, "WHISPER_HALLUCINATION_SILENCE_THRESHOLD", "Hallucination silence (sec):", 0.0, 10.0, 0.5)
        tabs.addTab(whisper_tab, "Whisper")

        # Hotkey tab
        hotkey_tab = QWidget()
        hf = QFormLayout(hotkey_tab)
        self._add_text(hf, "HOTKEY", "Hotkey:")
        tabs.addTab(hotkey_tab, "Hotkey")

        # LLM tab
        llm_tab = QWidget()
        lf = QFormLayout(llm_tab)
        self._add_check(lf, "USE_LLM_TRANSFORM", "Enable LLM transform")
        self._add_combo(lf, "LLM_BACKEND", "Backend:", ["ollama", "openai"])
        self._add_text(lf, "LLM_MODEL", "Model:")
        self._add_text(lf, "LLM_URL", "URL:")
        self._add_text(lf, "LLM_API_KEY", "API Key:", password=True)
        self._add_combo(lf, "LLM_REASONING_EFFORT", "Reasoning:", ["none", "low", "medium", "high", "off"])
        tabs.addTab(llm_tab, "LLM")

        # Output tab
        output_tab = QWidget()
        of = QFormLayout(output_tab)
        self._add_check(of, "COPY_TO_CLIPBOARD", "Copy to clipboard")
        self._add_check(of, "PASTE_TO_ACTIVE_WINDOW", "Paste to active window")
        self._add_combo(of, "PASTE_METHOD", "Paste method:", [
            "auto", "ctrl+v", "ctrl+shift+v", "shift+insert",
        ])
        self._add_combo(of, "CLIPBOARD_AFTER_PASTE_POLICY", "Clipboard after paste:", [
            "restore", "clear", "preserve",
        ])
        self._add_text(of, "KEYS_AFTER_PASTE", "Keys after paste:")
        tabs.addTab(output_tab, "Output")

        # Audio tab
        audio_tab = QWidget()
        af = QFormLayout(audio_tab)
        self._add_mic_combo(af)
        self._add_spin(af, "SAMPLE_RATE", "Sample rate:", 8000, 48000, 1000)
        self._add_spin(af, "CHUNK_SIZE", "Chunk size:", 256, 8192, 256)
        self._add_dspin(af, "PREBUFFER_SEC", "Prebuffer (sec):", 0.0, 5.0, 0.1)
        self._add_dspin(af, "PADDING_SEC", "Padding (sec):", 0.0, 2.0, 0.1)
        self._add_spin(af, "MIN_FRAMES", "Min frames:", 1, 50, 1)
        self._add_spin(af, "SILENCE_AMPLITUDE", "Silence threshold:", 0, 10000, 50)
        tabs.addTab(audio_tab, "Audio")

        # Chunking tab
        chunk_tab = QWidget()
        cf = QFormLayout(chunk_tab)
        self._add_dspin(cf, "CHUNK_DURATION_SEC", "Chunk duration (sec):", 0.0, 120.0, 1.0)
        self._add_dspin(cf, "CHUNK_OVERLAP_SEC", "Chunk overlap (sec):", 0.0, 10.0, 0.5)
        tabs.addTab(chunk_tab, "Chunking")

        # SpellCheck tab
        sc_tab = QWidget()
        scf = QFormLayout(sc_tab)
        self._add_check(scf, "SPELLCHECK_ENABLED", "Enable SpellCheck")
        self._add_text(scf, "SPELLCHECK_HOTKEY", "Hotkey:")
        self._add_combo(scf, "SPELLCHECK_LANGUAGE", "Language:", [
            "auto", "ru", "en",
        ])
        self._add_check(scf, "SPELLCHECK_CLEAN_PROFANITY", "Clean profanity")
        tabs.addTab(sc_tab, "SpellCheck")

        # General tab
        general_tab = QWidget()
        gf = QFormLayout(general_tab)
        self._add_check(gf, "SHOW_NOTIFICATIONS", "Show notifications")
        self._add_check(gf, "LOG_ENABLED", "Enable logging")
        self._add_check(gf, "AUTOSTART", "Start with Windows")
        tabs.addTab(general_tab, "General")

        layout.addWidget(tabs)

        # Buttons
        btn_layout = QHBoxLayout()
        self._restart_label = QLabel("")
        self._restart_label.setStyleSheet("color: orange;")
        btn_layout.addWidget(self._restart_label)
        btn_layout.addStretch()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _add_mic_combo(self, form):
        w = QComboBox()
        devices = core.list_audio_devices()
        w.addItem("Default (system)", "default")
        for dev in devices:
            label = dev["name"]
            if dev["is_default"]:
                label += " [system default]"
            w.addItem(label, dev["name"])
        # Select current
        current = core.get_config().get("AUDIO_DEVICE", "default")
        if not current or current.lower() == "default":
            w.setCurrentIndex(0)
        else:
            for i in range(1, w.count()):
                if current.lower() in w.itemData(i).lower():
                    w.setCurrentIndex(i)
                    break
        self._widgets["AUDIO_DEVICE"] = w
        form.addRow("Microphone:", w)

    def _add_text(self, form, key, label, password=False):
        w = QLineEdit()
        if password:
            w.setEchoMode(QLineEdit.Password)
        self._widgets[key] = w
        lbl = label
        if key in self.RESTART_KEYS:
            lbl += " *"
        form.addRow(lbl, w)

    def _add_check(self, form, key, label):
        w = QCheckBox(label)
        self._widgets[key] = w
        form.addRow("", w)

    def _add_combo(self, form, key, label, items):
        w = QComboBox()
        w.addItems(items)
        w.setEditable(True)
        self._widgets[key] = w
        lbl = label
        if key in self.RESTART_KEYS:
            lbl += " *"
        form.addRow(lbl, w)

    def _add_spin(self, form, key, label, min_v, max_v, step):
        w = QSpinBox()
        w.setRange(min_v, max_v)
        w.setSingleStep(step)
        self._widgets[key] = w
        lbl = label
        if key in self.RESTART_KEYS:
            lbl += " *"
        form.addRow(lbl, w)

    def _add_dspin(self, form, key, label, min_v, max_v, step):
        w = QDoubleSpinBox()
        w.setRange(min_v, max_v)
        w.setSingleStep(step)
        w.setDecimals(1)
        self._widgets[key] = w
        lbl = label
        if key in self.RESTART_KEYS:
            lbl += " *"
        form.addRow(lbl, w)

    def _load_values(self):
        cfg = core.get_config()
        # AUTOSTART is not in core config, read from registry
        cfg["AUTOSTART"] = _get_autostart()
        for key, widget in self._widgets.items():
            if key == "AUDIO_DEVICE":
                continue  # already set in _add_mic_combo
            val = cfg.get(key, "")
            if isinstance(widget, QCheckBox):
                widget.setChecked(val is True)
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(val))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                else:
                    widget.setCurrentText(str(val))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(val if val is not None else 0)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(val) if val is not None else "")

    def _get_values(self):
        result = {}
        for key, widget in self._widgets.items():
            if isinstance(widget, QCheckBox):
                result[key] = widget.isChecked()
            elif key == "AUDIO_DEVICE":
                result[key] = widget.currentData() or "default"
            elif isinstance(widget, QComboBox):
                result[key] = widget.currentText()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                result[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                result[key] = widget.text()
        return result

    def _apply(self):
        values = self._get_values()
        old_cfg = core.get_config()

        # Handle autostart separately (not a .env setting)
        autostart = values.pop("AUTOSTART", False)
        _set_autostart(autostart)

        # Write to .env
        env_path = os.path.join(_script_dir, ".env")
        try:
            from dotenv import set_key
            for key, val in values.items():
                env_key = f"WHISPER_PTT_{key}"
                if isinstance(val, bool):
                    str_val = "true" if val else "false"
                else:
                    str_val = str(val)
                set_key(env_path, env_key, str_val)
        except ImportError:
            QMessageBox.warning(self, "Warning",
                "python-dotenv not installed. Cannot save to .env file.")
            return

        # Reload config in core
        changed = core.reload_config()

        # Hot-swap microphone if changed
        if "AUDIO_DEVICE" in changed:
            core.switch_microphone(changed["AUDIO_DEVICE"])

        # Check if restart-requiring settings changed
        restart_needed = set(changed.keys()) & self.RESTART_KEYS
        if restart_needed:
            self._restart_label.setText(
                f"* Restart required for: {', '.join(restart_needed)}")
        else:
            self._restart_label.setText("")

        self.accept()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class WhisperPTTApp:
    """Main application: tray icon, overlay, settings, core lifecycle."""

    def __init__(self):
        self._app = QApplication(sys.argv)
        self._app.setQuitOnLastWindowClosed(False)

        self._bridge = AudioBridge()
        self._overlay = RecordingOverlay(self._bridge)
        self._settings_dialog = None

        # State
        self._state = "loading"  # loading, idle, recording, processing

        # Tray
        self._setup_tray()

        # Connect bridge signals
        self._bridge.recording_started.connect(self._on_recording_started)
        self._bridge.recording_stopped.connect(self._on_recording_stopped)
        self._bridge.processing_started.connect(self._on_processing_started)
        self._bridge.transcription_done.connect(self._on_transcription_done)
        self._bridge.spellcheck_started.connect(self._on_spellcheck_started)
        self._bridge.spellcheck_done.connect(self._on_spellcheck_done)
        self._bridge.error_occurred.connect(self._on_error)

        # Register callbacks with core
        core.set_audio_level_callback(self._bridge.on_audio_level)
        core.set_event_callback(self._bridge.on_event)

    def _setup_tray(self):
        self._icons = {
            "loading": QIcon(os.path.join(_script_dir, "assets", "icon_loading.png")),
            "idle": QIcon(os.path.join(_script_dir, "assets", "icon_idle.png")),
            "recording": QIcon(os.path.join(_script_dir, "assets", "icon_recording.png")),
            "processing": QIcon(os.path.join(_script_dir, "assets", "icon_processing.png")),
        }

        self._tray = QSystemTrayIcon(self._icons["loading"])
        self._tray.setToolTip("Whisper PTT - Loading...")

        menu = QMenu()
        settings_action = QAction("Settings", menu)
        settings_action.triggered.connect(self._show_settings)
        menu.addAction(settings_action)
        log_action = QAction("Open Log", menu)
        log_action.triggered.connect(self._open_log)
        menu.addAction(log_action)
        reregister_action = QAction("Re-register hotkeys", menu)
        reregister_action.triggered.connect(self._reregister_hotkeys)
        menu.addAction(reregister_action)
        menu.addSeparator()
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(self._quit)
        menu.addAction(quit_action)

        self._tray.setContextMenu(menu)
        self._tray.show()

    def _set_state(self, state):
        self._state = state
        icon = self._icons.get(state, self._icons["idle"])
        self._tray.setIcon(icon)
        hotkey = core.HOTKEY.upper()
        tooltips = {
            "loading": "Whisper PTT - Loading model...",
            "idle": f"Whisper PTT - Ready (hold {hotkey})",
            "recording": "Whisper PTT - Recording...",
            "processing": "Whisper PTT - Processing...",
        }
        self._tray.setToolTip(tooltips.get(state, "Whisper PTT"))

    @Slot()
    def _on_recording_started(self):
        self._set_state("recording")
        self._overlay.show_overlay()

    @Slot()
    def _on_recording_stopped(self):
        self._set_state("idle")
        self._overlay.hide_overlay()

    @Slot()
    def _on_processing_started(self):
        self._set_state("processing")

    @Slot(str)
    def _on_transcription_done(self, text):
        self._set_state("idle")
        if text.strip() and core.SHOW_NOTIFICATIONS:
            self._tray.showMessage("Whisper PTT", text[:200], QSystemTrayIcon.MessageIcon.Information, 3000)

    @Slot(str)
    def _on_error(self, message):
        self._set_state("idle")
        self._tray.showMessage(
            "Whisper PTT - Error", message[:300],
            QSystemTrayIcon.MessageIcon.Critical, 5000)

    @Slot()
    def _on_spellcheck_started(self):
        self._set_state("processing")

    @Slot(str, bool)
    def _on_spellcheck_done(self, text, changed):
        self._set_state("idle")
        if core.SHOW_NOTIFICATIONS:
            if changed and text.strip():
                self._tray.showMessage("SpellCheck", text[:200], QSystemTrayIcon.MessageIcon.Information, 3000)
            elif not changed and not text:
                pass  # no text selected — silent

    def _open_log(self):
        log_path = core.get_log_path()
        if not os.path.isfile(log_path):
            self._tray.showMessage(
                "Whisper PTT", "Log file not found. Enable logging in Settings first.",
                QSystemTrayIcon.MessageIcon.Information, 3000)
            return
        # Open with default text editor
        if sys.platform == "win32":
            os.startfile(log_path)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", log_path])
        else:
            subprocess.Popen(["xdg-open", log_path])

    def _reregister_hotkeys(self):
        """Unregister and re-register keyboard hotkeys."""
        try:
            core.unregister_hotkeys()
            core.register_hotkeys()
            parts = [core.HOTKEY.upper()]
            if core.SPELLCHECK_ENABLED:
                parts.append(f"SpellCheck: {core.SPELLCHECK_HOTKEY.upper()}")
            self._tray.showMessage(
                "Whisper PTT", f"Hotkeys re-registered ({', '.join(parts)})",
                QSystemTrayIcon.MessageIcon.Information, 2000)
        except Exception as e:
            self._tray.showMessage(
                "Whisper PTT - Error", f"Failed to re-register hotkeys: {e}",
                QSystemTrayIcon.MessageIcon.Critical, 3000)

    def _show_settings(self):
        if self._settings_dialog and self._settings_dialog.isVisible():
            self._settings_dialog.activateWindow()
            return
        self._settings_dialog = SettingsDialog()
        self._settings_dialog.show()

    def _quit(self):
        print("Shutting down...")
        core.unregister_hotkeys()
        core.shutdown()
        self._tray.hide()
        self._app.quit()

    def run(self):
        """Start the application: load core in background, then enter Qt event loop."""
        self._init_receiver = _CoreInitReceiver(self)

        def _init_core():
            try:
                core.init_whisper()
                core.init_audio()
                core.register_hotkeys()
                QMetaObject.invokeMethod(
                    self._init_receiver, "on_core_ready", Qt.QueuedConnection
                )
            except Exception as e:
                print(f"Core init error: {e}")
                self._bridge._pending_error = str(e)
                QMetaObject.invokeMethod(
                    self._bridge, "_emit_error", Qt.QueuedConnection
                )

        threading.Thread(target=_init_core, daemon=True).start()
        return self._app.exec()


class _CoreInitReceiver(QObject):
    """Helper QObject to receive core-ready signal on main thread."""
    def __init__(self, ptt_app):
        super().__init__()
        self._ptt_app = ptt_app

    @Slot()
    def on_core_ready(self):
        self._ptt_app._set_state("idle")
        hotkey = core.HOTKEY.upper()
        print(f"Ready! Hold {hotkey} to record.")


def _acquire_single_instance():
    """Ensure only one instance is running. Returns mutex handle or exits."""
    if sys.platform == "win32":
        import ctypes
        mutex = ctypes.windll.kernel32.CreateMutexW(None, True, "WhisperPTT_SingleInstance")
        ERROR_ALREADY_EXISTS = 183
        if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
            return None
        return mutex
    return True


def main():
    mutex = _acquire_single_instance()
    if mutex is None:
        print("Another instance is already running.")
        sys.exit(0)

    # Autostart delay: when launched from Windows startup (registry),
    # keyboard hooks registered too early don't receive events because
    # the desktop isn't fully initialized yet. Wait before starting.
    if "--autostart" in sys.argv:
        delay = 5
        for arg in sys.argv:
            if arg.startswith("--autostart-delay="):
                try:
                    delay = int(arg.split("=", 1)[1])
                except ValueError:
                    pass
        if delay > 0:
            time.sleep(delay)

    app = WhisperPTTApp()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
