import sys
import os
import glob
import json
import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QHBoxLayout,
                             QVBoxLayout, QWidget, QGridLayout, QLabel, QFrame,
                             QProgressBar, QFileDialog, QSpinBox, QSlider, QComboBox, QMessageBox)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal

# --- MAC COMPATIBILITY ---
# JACK_CLIENT_NAME is ignored by Core Audio but kept for cross-platform stability
os.environ["JACK_CLIENT_NAME"] = "Lohjinawi-64"

class ClickableLabel(QLabel):
    clicked = pyqtSignal(int)
    def __init__(self, index, text):
        super().__init__(text)
        self.index = index
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("font-size: 10px; color: #a0a0a0; font-weight: bold; padding: 2px;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def mousePressEvent(self, event):
        self.clicked.emit(self.index)

class AudioEngine:
    def __init__(self, channels=64):
        self.channels = channels
        self.samplerate = 48000
        self.tracks = []
        self.levels = np.zeros(channels)
        self.master_gain = 1.0
        self.current_frame = 0
        self.total_frames = 0
        self.is_seeking = False
        self.loop_enabled = False

        self.noise_active = set()
        self.noise_ptr = 0
        self.noise_vol = 0.15

        N = self.samplerate * 2
        X = np.fft.rfft(np.random.randn(N))
        S = 1 / np.sqrt(np.arange(1, len(X)+1))
        pink = np.fft.irfft(X * S)
        self.pink_noise = (pink / np.max(np.abs(pink))).astype(np.float32)

    def add_file(self, path, out_start_index):
        try:
            f = sf.SoundFile(path)
            if out_start_index + f.channels > self.channels:
                return f.channels
            self.tracks.append({'file': f, 'path': path, 'out_idx': out_start_index, 'ch': f.channels, 'active': True})
            self.total_frames = max(self.total_frames, f.frames)
            return f.channels
        except Exception as e:
            return 0

    def seek(self, target_frame):
        self.is_seeking = True
        self.current_frame = target_frame
        for track in self.tracks:
            safe_frame = min(target_frame, track['file'].frames)
            track['file'].seek(safe_frame)
            if safe_frame < track['file'].frames:
                track['active'] = True
        self.is_seeking = False

    def callback(self, outdata, frames, time, status):
        outdata.fill(0)
        if self.is_seeking: return

        for track in self.tracks:
            if not track['active']: continue
            chunk = track['file'].read(frames, dtype='float32')
            if len(chunk) == 0:
                track['active'] = False
                continue

            start = track['out_idx']
            end = min(start + (track['ch'] if len(chunk.shape) > 1 else 1), self.channels)

            if len(chunk.shape) > 1:
                actual_cols = end - start
                outdata[:len(chunk), start:end] += chunk[:, :actual_cols]
            else:
                outdata[:len(chunk), start] += chunk.flatten()

        if self.noise_active:
            frames_left = frames
            out_ptr = 0
            while frames_left > 0:
                chunk_size = min(frames_left, len(self.pink_noise) - self.noise_ptr)
                noise_chunk = self.pink_noise[self.noise_ptr : self.noise_ptr + chunk_size] * self.noise_vol
                for ch in self.noise_active:
                    if ch < self.channels:
                        outdata[out_ptr : out_ptr + chunk_size, ch] += noise_chunk
                self.noise_ptr = (self.noise_ptr + chunk_size) % len(self.pink_noise)
                out_ptr += chunk_size
                frames_left -= chunk_size

        outdata[:] *= self.master_gain
        np.clip(outdata, -1.0, 1.0, outdata)
        self.levels[:self.channels] = np.max(np.abs(outdata[:, :self.channels]), axis=0)
        self.current_frame += frames

class MainWindow(QMainWindow):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.setWindowTitle("Lohjinawi-64 | Mesin Putar Multijalur")
        self.resize(1300, 800)

        self.setStyleSheet("""
            QMainWindow { background-color: #2b2d30; }
            QLabel { color: #dfdfdf; font-family: 'Helvetica Neue', Arial, sans-serif; }
            QFrame#panel { background-color: #36393f; border-radius: 8px; border: 1px solid #45474a; }
            QFrame#bank { background-color: #2a2c2f; border-radius: 6px; border: 1px solid #1e1e1e; }
            QPushButton { background-color: #4b5054; color: #ffffff; border: 1px solid #5c6166; border-radius: 4px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #5c6268; }
            QComboBox, QSpinBox { background-color: #3b3f43; color: white; border: 1px solid #5c6166; border-radius: 4px; padding: 5px; }
            QProgressBar { background-color: #1a1c1e; border: 1px solid #111; border-radius: 2px; }
            QProgressBar::chunk { background-color: #00ff66; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # I/O Panel
        io_panel = QFrame()
        io_panel.setObjectName("panel")
        io_layout = QHBoxLayout(io_panel)
        title_lbl = QLabel("LOHJINAWI-64")
        title_lbl.setStyleSheet("font-size: 18px; font-weight: 900; color: #2ecc71; letter-spacing: 1px;")

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(300)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                self.device_combo.addItem(f"{dev['name']}", i)

        # AUTO-SELECT CORE AUDIO (MAC STANDARD)
        for idx in range(self.device_combo.count()):
            if "CORE AUDIO" in self.device_combo.itemText(idx).upper() or "BUILT-IN" in self.device_combo.itemText(idx).upper():
                self.device_combo.setCurrentIndex(idx)
                break

        io_layout.addWidget(title_lbl)
        io_layout.addStretch()
        io_layout.addWidget(QLabel("Output Device:"))
        io_layout.addWidget(self.device_combo)
        main_layout.addWidget(io_panel)

        # Matrix Panel (8x8 banks)
        matrix_panel = QFrame()
        matrix_panel.setObjectName("panel")
        matrix_layout = QVBoxLayout(matrix_panel)
        grid_container = QWidget()
        grid = QGridLayout(grid_container)
        self.meters = []
        self.meter_labels = []
        for bank in range(8):
            bank_frame = QFrame()
            bank_frame.setObjectName("bank")
            bank_layout = QHBoxLayout(bank_frame)
            for i in range(8):
                ch_idx = bank * 8 + i
                m_cont = QVBoxLayout()
                bar = QProgressBar()
                bar.setOrientation(Qt.Orientation.Vertical)
                bar.setRange(0, 100)
                bar.setTextVisible(False)
                lbl = ClickableLabel(ch_idx, f"{ch_idx+1}")
                lbl.clicked.connect(self.toggle_manual_noise)
                m_cont.addWidget(bar, stretch=1)
                m_cont.addWidget(lbl, stretch=0)
                bank_layout.addLayout(m_cont)
                self.meters.append(bar)
                self.meter_labels.append(lbl)
            grid.addWidget(bank_frame, bank // 4, bank % 4)
        matrix_layout.addWidget(grid_container)
        main_layout.addWidget(matrix_panel)

        # Transport
        transport_panel = QFrame()
        transport_panel.setObjectName("panel")
        transport_layout = QVBoxLayout(transport_panel)

        time_layout = QHBoxLayout()
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("font-family: monospace; font-size: 14px; color: #00ff66; background: #202225; padding: 5px;")
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 1000)
        self.timeline_slider.setEnabled(False)
        self.is_scrubbing_ui = False
        self.timeline_slider.sliderPressed.connect(lambda: setattr(self, 'is_scrubbing_ui', True))
        self.timeline_slider.sliderReleased.connect(self.scrub_finished)
        time_layout.addWidget(self.time_label)
        time_layout.addWidget(self.timeline_slider)
        transport_layout.addLayout(time_layout)

        self.info_label = QLabel("No media loaded.")
        self.info_label.setStyleSheet("color: #888; font-style: italic;")
        transport_layout.addWidget(self.info_label)

        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Add Audio")
        load_btn.clicked.connect(self.load_action)
        folder_btn = QPushButton("Auto-Load Folder")
        folder_btn.clicked.connect(self.load_folder_action)
        self.start_btn = QPushButton("▶ START")
        self.start_btn.clicked.connect(self.start_audio)
        self.stop_btn = QPushButton("⏸ PAUSE")
        self.stop_btn.clicked.connect(self.stop_audio)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(load_btn)
        btn_layout.addWidget(folder_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        transport_layout.addLayout(btn_layout)

        credit_lbl = QLabel("developed by rekambergeraklab - Yogyakarta - Indonesia")
        credit_lbl.setStyleSheet("color: #666; font-size: 10px;")
        credit_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(credit_lbl)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_vis)
        self.timer.start(33)
        self.stream = None
        self.session_file = "panoramix_session.json"
        self.load_settings()

    def scrub_finished(self):
        if self.engine.total_frames > 0:
            self.engine.seek(int((self.timeline_slider.value() / 1000.0) * self.engine.total_frames))
        self.is_scrubbing_ui = False

    def load_action(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Audio")
        if path:
            used = self.engine.add_file(path, 0)
            self.timeline_slider.setEnabled(True)
            self.info_label.setText(f"File: {os.path.basename(path)}")

    def load_folder_action(self):
        folder = QFileDialog.getExistingDirectory(self)
        if folder:
            files = sorted(glob.glob(os.path.join(folder, "*.wav")) + glob.glob(os.path.join(folder, "*.flac")))
            curr = 0
            for p in files:
                if curr >= 64: break
                curr += self.engine.add_file(p, curr)
            self.timeline_slider.setEnabled(True)
            self.info_label.setText(f"Folder: {folder}")

    def start_audio(self):
        try:
            dev_idx = self.device_combo.currentData()
            self.stream = sd.OutputStream(samplerate=48000, channels=64, callback=self.engine.callback, device=dev_idx)
            self.stream.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def stop_audio(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def toggle_manual_noise(self, idx):
        if idx in self.engine.noise_active: self.engine.noise_active.remove(idx)
        else: self.engine.noise_active.add(idx)

    def load_settings(self):
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    for t in data.get('tracks', []): self.engine.add_file(t['path'], t['out_idx'])
            except: pass

    def refresh_vis(self):
        if self.stream:
            for i in range(64): self.meters[i].setValue(int(self.engine.levels[i] * 100))
            if not self.is_scrubbing_ui and self.engine.total_frames > 0:
                prog = self.engine.current_frame / self.engine.total_frames
                self.timeline_slider.setValue(int(prog * 1000))
                self.time_label.setText(f"{int(self.engine.current_frame/48000)//60:02d}:{int(self.engine.current_frame/48000)%60:02d} / Total")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    e = AudioEngine(64)
    w = MainWindow(e)
    w.show()
    sys.exit(app.exec())
