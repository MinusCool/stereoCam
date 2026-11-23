import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image
import json
import datetime
import os
import re
import threading
import time
import subprocess
from collections import deque
from typing import Optional, List, Dict
import torch_directml
from ultralytics import YOLO
try:
    from openpyxl import Workbook, load_workbook
    OPENPYXL_AVAILABLE = True
except Exception:
    OPENPYXL_AVAILABLE = False

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30
YOLO_INPUT_SIZE = 640
VEHICLE_INPUT_SIZE = 960
WHEEL_ROI_INPUT_SIZE = 640
VEHICLE_CONF = 0.20
WHEEL_CONF = 0.08
WHEEL_ROI_PAD_RATIO = 0.15
MAX_EPIPOLAR_DIFF_PX = 20.0
MIN_VALID_DISPARITY_PX = 1.5
MAX_VALID_DISPARITY_PX = 800.0
MIN_VALID_SPARSE_DEPTH_CM = 30.0
MAX_VALID_SPARSE_DEPTH_CM = 3000.0
TRACK_IOU_THRESH = 0.25
TRACK_MIN_HITS = 3
TRACK_MAX_MISS = 6
SMOOTH_ALPHA = 0.35
WHEEL_MATCH_DIST_PX = 24
VEHICLE_PAIR_MAX_Y_DIFF_PX = 70
SUBPIX_TEMPLATE_HALF = 7
SUBPIX_SEARCH_HALF = 40
GUIDE_MIN_DEPTH_CM = 500.0
GUIDE_MAX_DEPTH_CM = 2000.0
EXPOSURE_EV = 0.0
WHITE_BALANCE_SHIFT = 0.0
DIGITAL_GAIN = 1.0
GAMMA_VALUE = 1.0
RUNTIME_CONFIG_PATH = "runtime.json"
EXCEL_LOG_PATH = os.path.join("snapshots", "measurements.xlsx")

cv2.ocl.setUseOpenCL(True)
print("OpenCL available:", cv2.ocl.haveOpenCL())

dml = torch_directml.device()


def load_runtime_config():
    global VEHICLE_CONF, WHEEL_CONF, MAX_EPIPOLAR_DIFF_PX, GUIDE_MIN_DEPTH_CM, GUIDE_MAX_DEPTH_CM
    global EXPOSURE_EV, WHITE_BALANCE_SHIFT, DIGITAL_GAIN, GAMMA_VALUE
    if not os.path.exists(RUNTIME_CONFIG_PATH):
        return
    try:
        with open(RUNTIME_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        VEHICLE_CONF = float(cfg.get("VEHICLE_CONF", VEHICLE_CONF))
        WHEEL_CONF = float(cfg.get("WHEEL_CONF", WHEEL_CONF))
        MAX_EPIPOLAR_DIFF_PX = float(cfg.get("MAX_EPIPOLAR_DIFF_PX", MAX_EPIPOLAR_DIFF_PX))
        GUIDE_MIN_DEPTH_CM = float(cfg.get("GUIDE_MIN_DEPTH_CM", GUIDE_MIN_DEPTH_CM))
        GUIDE_MAX_DEPTH_CM = float(cfg.get("GUIDE_MAX_DEPTH_CM", GUIDE_MAX_DEPTH_CM))
        EXPOSURE_EV = float(cfg.get("EXPOSURE_EV", EXPOSURE_EV))
        WHITE_BALANCE_SHIFT = float(cfg.get("WHITE_BALANCE_SHIFT", WHITE_BALANCE_SHIFT))
        DIGITAL_GAIN = float(cfg.get("DIGITAL_GAIN", DIGITAL_GAIN))
        GAMMA_VALUE = float(cfg.get("GAMMA_VALUE", GAMMA_VALUE))
    except Exception:
        pass


def save_runtime_config():
    cfg = {
        "VEHICLE_CONF": VEHICLE_CONF,
        "WHEEL_CONF": WHEEL_CONF,
        "MAX_EPIPOLAR_DIFF_PX": MAX_EPIPOLAR_DIFF_PX,
        "GUIDE_MIN_DEPTH_CM": GUIDE_MIN_DEPTH_CM,
        "GUIDE_MAX_DEPTH_CM": GUIDE_MAX_DEPTH_CM,
        "EXPOSURE_EV": EXPOSURE_EV,
        "WHITE_BALANCE_SHIFT": WHITE_BALANCE_SHIFT,
        "DIGITAL_GAIN": DIGITAL_GAIN,
        "GAMMA_VALUE": GAMMA_VALUE,
    }
    try:
        with open(RUNTIME_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception:
        pass


def run_ffmpeg_list_dshow_devices() -> str:
    cmd = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except FileNotFoundError:
        return ""
    return p.stderr


load_runtime_config()


def parse_dshow_video_devices(ffmpeg_text: str) -> List[Dict[str, str]]:
    devices = []
    lines = ffmpeg_text.splitlines()
    name_pat = re.compile(r'"\s*(.*?)\s*"\s*\(video\)', re.IGNORECASE)
    alt_pat = re.compile(r'Alternative name\s*"\s*(.*?)\s*"', re.IGNORECASE)

    i = 0
    while i < len(lines):
        m = name_pat.search(lines[i])
        if m:
            name = m.group(1)
            alt = None
            j = i + 1
            while j < len(lines) and j <= i + 5:
                m2 = alt_pat.search(lines[j])
                if m2:
                    alt = m2.group(1)
                    break
                j += 1
            if alt:
                devices.append({"name": name, "alt": alt})
            i = j
        i += 1
    return devices


def resize_with_aspect(bgr: np.ndarray, target_w: int, target_h: int) -> Optional[np.ndarray]:
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    if target_w <= 1 or target_h <= 1:
        return bgr
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def apply_camera_tuning(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame

    out = frame.astype(np.float32)

    # Exposure in EV steps.
    out *= float(2.0 ** EXPOSURE_EV)
    out *= float(DIGITAL_GAIN)

    # White balance shift: + warmer (more R), - cooler (more B).
    wb = float(WHITE_BALANCE_SHIFT)
    r_gain = max(0.6, 1.0 + 0.25 * wb)
    b_gain = max(0.6, 1.0 - 0.25 * wb)
    out[:, :, 2] *= r_gain
    out[:, :, 0] *= b_gain

    out = np.clip(out, 0, 255)
    gamma = float(max(0.5, GAMMA_VALUE))
    if abs(gamma - 1.0) > 1e-3:
        out = (out / 255.0) ** (1.0 / gamma) * 255.0

    return np.clip(out, 0, 255).astype(np.uint8)


class FFmpegCamera:
    def __init__(self, alt_name: str, width: int, height: int, fps: int):
        self.alt_name = alt_name
        self.width = width
        self.height = height
        self.fps = fps

        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.running = False
        self.last_frame: Optional[np.ndarray] = None
        self.last_ts: float = 0.0
        self._thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_tail = deque(maxlen=20)
        self.last_error: str = ""

    def start(self) -> bool:
        if self.running:
            return True

        cmd_profiles = [
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "dshow",
                "-video_size", f"{self.width}x{self.height}",
                "-framerate", str(self.fps),
                "-i", f"video={self.alt_name}",
                "-pix_fmt", "bgr24",
                "-vcodec", "rawvideo",
                "-f", "rawvideo",
                "pipe:1",
            ],
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", "dshow",
                "-video_size", f"{self.width}x{self.height}",
                "-i", f"video={self.alt_name}",
                "-pix_fmt", "bgr24",
                "-vcodec", "rawvideo",
                "-f", "rawvideo",
                "pipe:1",
            ],
        ]

        for cmd in cmd_profiles:
            if self._start_once(cmd):
                return True
            self.stop()
        return False

    def _start_once(self, cmd: List[str]) -> bool:
        self.last_error = ""
        self._stderr_tail.clear()
        self.last_frame = None
        self.last_ts = 0.0

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10 ** 7,
            )
        except FileNotFoundError:
            self.last_error = "FFmpeg tidak ditemukan di PATH."
            return False

        if self.proc.stdout is None:
            self.last_error = "FFmpeg stdout tidak tersedia."
            return False

        self.running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()

        t0 = time.time()
        while time.time() - t0 < 2.0:
            ok, _, _ = self.read()
            if ok:
                return True
            if self.proc is not None and self.proc.poll() is not None:
                break
            time.sleep(0.02)

        err = "\n".join(self._stderr_tail).strip()
        if err:
            self.last_error = err
        else:
            self.last_error = "FFmpeg start timeout: belum ada frame."
        return False

    def stop(self):
        self.running = False
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def _stderr_loop(self):
        if self.proc is None or self.proc.stderr is None:
            return
        while self.running:
            line = self.proc.stderr.readline()
            if not line:
                break
            msg = line.decode("utf-8", "ignore").strip()
            if msg:
                self._stderr_tail.append(msg)

    def _reader_loop(self):
        frame_bytes = self.width * self.height * 3
        stdout = self.proc.stdout

        while self.running and self.proc is not None and stdout is not None:
            raw = stdout.read(frame_bytes)
            if raw is None or len(raw) != frame_bytes:
                time.sleep(0.01)
                break

            frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
            with self.lock:
                self.last_frame = frame.copy()
                self.last_ts = time.time()

        self.running = False

    def read(self):
        with self.lock:
            if self.last_frame is None:
                return False, None, 0.0
            return True, self.last_frame.copy(), self.last_ts


def list_ffmpeg_devices():
    txt = run_ffmpeg_list_dshow_devices()
    devices = parse_dshow_video_devices(txt)
    if not devices:
        return [], {}

    counts = {}
    labels = []
    label_to_alt = {}
    for d in devices:
        base = d["name"]
        counts[base] = counts.get(base, 0) + 1
        idx = counts[base]
        label = f"{base}  (#{idx})"
        labels.append(label)
        label_to_alt[label] = d["alt"]
    return labels, label_to_alt


device_labels, label_to_alt = list_ffmpeg_devices()
selected_left_label = ""
selected_right_label = ""

if len(device_labels) >= 2:
    logi_labels = [lb for lb in device_labels if "logi c270" in lb.lower()]
    if len(logi_labels) >= 2:
        selected_left_label = logi_labels[0]
        selected_right_label = logi_labels[1]
    else:
        selected_left_label = device_labels[0]
        selected_right_label = device_labels[1]
elif len(device_labels) == 1:
    selected_left_label = device_labels[0]
    selected_right_label = device_labels[0]

camL = None
camR = None

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.geometry("1350x800")
app.title("Stereo Depth Measurement GUI (YOLO Only Test)")

calib_ready = False
if os.path.exists("model/stereo_calib_result.npz"):
    data = np.load("model/stereo_calib_result.npz")
    mtxL, distL = data["mtxL"], data["distL"]
    mtxR, distR = data["mtxR"], data["distR"]
    R1, R2 = data["R1"], data["R2"]
    P1, P2 = data["P1"], data["P2"]
    Q = data["Q"]
    fx_rect = float(P1[0, 0])
    baseline_m = float(-P2[0, 3] / fx_rect) if fx_rect != 0 else 0.0
    calib_ready = fx_rect > 0 and baseline_m > 0
    mapLx, mapLy = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (CAPTURE_WIDTH, CAPTURE_HEIGHT), cv2.CV_32FC1)
    mapRx, mapRy = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (CAPTURE_WIDTH, CAPTURE_HEIGHT), cv2.CV_32FC1)

latest_overlay = None
latest_heatmap = None

vehicle_model = YOLO("model/yolov8n.pt")
vehicle_model.model.to(dml)
print("Vehicle model device:", next(vehicle_model.model.parameters()).device)

wheel_model = YOLO("model/best.pt")
wheel_model.model.to(dml)
print("Wheel model device:", next(wheel_model.model.parameters()).device)

detection_result = {"left": [], "right_wheels": []}
latest_measurements = []
left_tracks = []
right_wheel_tracks = []
detection_lock = threading.Lock()
detection_busy = threading.Lock()
fps_counter = {
    "last_time": time.time(),
    "fps": 0.0
}
app_running = True
flip_foh_roh = False

label_cam = ctk.CTkLabel(app, text="Citra Kamera Kiri + YOLO", font=ctk.CTkFont(size=14, weight="bold"))
label_cam.grid(row=0, column=0, padx=10, pady=(10, 0))

label_values = ctk.CTkLabel(app, text="Nilai Deteksi", font=ctk.CTkFont(size=14, weight="bold"))
label_values.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="w")

frame_left = ctk.CTkLabel(app, text="")
frame_left.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

values_box = ctk.CTkTextbox(app, width=360)
values_box.grid(row=1, column=1, padx=(0, 10), pady=10, sticky="nsew")
values_box.insert("1.0", "Belum ada deteksi")
values_box.configure(state="disabled")

app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=0)
app.grid_rowconfigure(1, weight=1)

control_frame = ctk.CTkFrame(app)
control_frame.grid(row=2, column=0, columnspan=2, pady=10, padx=10, sticky="ew")
control_frame.grid_columnconfigure((1, 3, 5), weight=1)

camera_status_label = ctk.CTkLabel(control_frame, text="")
camera_status_label.grid(row=6, column=0, columnspan=10, padx=5, pady=(5, 0), sticky="w")


def stop_cameras():
    global camL, camR
    if camL is not None:
        camL.stop()
    if camR is not None:
        camR.stop()
    camL = None
    camR = None


def initialize_cameras(left_label: str, right_label: str):
    global camL, camR
    stop_cameras()

    if left_label not in label_to_alt or right_label not in label_to_alt:
        camera_status_label.configure(text="Device FFmpeg belum valid.")
        return False

    altL = label_to_alt[left_label]
    altR = label_to_alt[right_label]
    camL = FFmpegCamera(altL, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS)
    camR = FFmpegCamera(altR, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS)

    okL = camL.start()
    okR = camR.start()
    if not okL or not okR:
        errL = camL.last_error if camL is not None else ""
        errR = camR.last_error if camR is not None else ""
        stop_cameras()
        camera_status_label.configure(text=f"Gagal membuka kamera. L: {errL[:140]} | R: {errR[:140]}")
        return False

    if left_label == right_label:
        camera_status_label.configure(text="Warning: Left/Right memilih device yang sama.")
    else:
        camera_status_label.configure(text=f"Running: {left_label} | {right_label}")
    return True


def on_left_cam_select(choice):
    global selected_left_label
    selected_left_label = choice
    initialize_cameras(selected_left_label, selected_right_label)


def on_right_cam_select(choice):
    global selected_right_label
    selected_right_label = choice
    initialize_cameras(selected_left_label, selected_right_label)


camera_select_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
camera_select_frame.grid(row=4, column=0, columnspan=10, pady=(8, 0), sticky="")
camera_select_frame.grid_columnconfigure((0, 1, 2), weight=1)

left_cam_block = ctk.CTkFrame(camera_select_frame)
left_cam_block.grid(row=0, column=0, padx=12, pady=4)
ctk.CTkLabel(left_cam_block, text="Cam Kiri").pack(pady=(6, 2))
left_dropdown = ctk.CTkComboBox(
    left_cam_block,
    values=device_labels if device_labels else ["-"],
    command=on_left_cam_select,
    width=320,
)
left_dropdown.set(selected_left_label if selected_left_label else "-")
left_dropdown.pack(padx=8, pady=(0, 8))

right_cam_block = ctk.CTkFrame(camera_select_frame)
right_cam_block.grid(row=0, column=2, padx=12, pady=4)
ctk.CTkLabel(right_cam_block, text="Cam Kanan").pack(pady=(6, 2))
right_dropdown = ctk.CTkComboBox(
    right_cam_block,
    values=device_labels if device_labels else ["-"],
    command=on_right_cam_select,
    width=320,
)
right_dropdown.set(selected_right_label if selected_right_label else "-")
right_dropdown.pack(padx=8, pady=(0, 8))


def switch_cameras():
    global selected_left_label, selected_right_label
    selected_left_label, selected_right_label = selected_right_label, selected_left_label
    left_dropdown.set(selected_left_label if selected_left_label else "-")
    right_dropdown.set(selected_right_label if selected_right_label else "-")
    initialize_cameras(selected_left_label, selected_right_label)


switch_button = ctk.CTkButton(control_frame, text="Switch Kamera", command=switch_cameras)
switch_button.grid_forget()
switch_button = ctk.CTkButton(camera_select_frame, text="Switch Kamera", command=switch_cameras)
switch_button.grid(row=0, column=1, padx=12, pady=4)

vehicle_conf_label = ctk.CTkLabel(control_frame, text=f"Vehicle Conf: {VEHICLE_CONF:.2f}")
vehicle_conf_label.grid(row=0, column=0, padx=5)


def update_vehicle_conf(val):
    global VEHICLE_CONF
    VEHICLE_CONF = float(val)
    vehicle_conf_label.configure(text=f"Vehicle Conf: {VEHICLE_CONF:.2f}")
    save_runtime_config()


wheel_conf_label = ctk.CTkLabel(control_frame, text=f"Wheel Conf: {WHEEL_CONF:.2f}")
wheel_conf_label.grid(row=0, column=1, padx=5)


def update_wheel_conf(val):
    global WHEEL_CONF
    WHEEL_CONF = float(val)
    wheel_conf_label.configure(text=f"Wheel Conf: {WHEEL_CONF:.2f}")
    save_runtime_config()


epipolar_label = ctk.CTkLabel(control_frame, text=f"Epipolar Tol: {MAX_EPIPOLAR_DIFF_PX:.0f}px")
epipolar_label.grid(row=0, column=2, padx=5)


def update_epipolar_tol(val):
    global MAX_EPIPOLAR_DIFF_PX
    MAX_EPIPOLAR_DIFF_PX = float(val)
    epipolar_label.configure(text=f"Epipolar Tol: {MAX_EPIPOLAR_DIFF_PX:.0f}px")
    save_runtime_config()


guide_min_label = ctk.CTkLabel(control_frame, text=f"Guide Min: {GUIDE_MIN_DEPTH_CM/100:.1f}m")
guide_min_label.grid(row=0, column=4, padx=5)


def update_guide_min(val):
    global GUIDE_MIN_DEPTH_CM
    v = float(val)
    if v >= GUIDE_MAX_DEPTH_CM - 50:
        v = GUIDE_MAX_DEPTH_CM - 50
    GUIDE_MIN_DEPTH_CM = v
    guide_min_label.configure(text=f"Guide Min: {GUIDE_MIN_DEPTH_CM/100:.1f}m")
    save_runtime_config()


guide_max_label = ctk.CTkLabel(control_frame, text=f"Guide Max: {GUIDE_MAX_DEPTH_CM/100:.1f}m")
guide_max_label.grid(row=0, column=5, padx=5)


def update_guide_max(val):
    global GUIDE_MAX_DEPTH_CM
    v = float(val)
    if v <= GUIDE_MIN_DEPTH_CM + 50:
        v = GUIDE_MIN_DEPTH_CM + 50
    GUIDE_MAX_DEPTH_CM = v
    guide_max_label.configure(text=f"Guide Max: {GUIDE_MAX_DEPTH_CM/100:.1f}m")
    save_runtime_config()


exposure_ev_label = ctk.CTkLabel(control_frame, text=f"Exposure EV: {EXPOSURE_EV:.2f}")
exposure_ev_label.grid(row=2, column=0, padx=5)


def update_exposure_ev(val):
    global EXPOSURE_EV
    EXPOSURE_EV = float(val)
    exposure_ev_label.configure(text=f"Exposure EV: {EXPOSURE_EV:.2f}")
    save_runtime_config()


wb_shift_label = ctk.CTkLabel(control_frame, text=f"WB Shift: {WHITE_BALANCE_SHIFT:.2f}")
wb_shift_label.grid(row=2, column=1, padx=5)


def update_wb_shift(val):
    global WHITE_BALANCE_SHIFT
    WHITE_BALANCE_SHIFT = float(val)
    wb_shift_label.configure(text=f"WB Shift: {WHITE_BALANCE_SHIFT:.2f}")
    save_runtime_config()


gain_label = ctk.CTkLabel(control_frame, text=f"Gain: {DIGITAL_GAIN:.2f}")
gain_label.grid(row=2, column=2, padx=5)


def update_digital_gain(val):
    global DIGITAL_GAIN
    DIGITAL_GAIN = float(val)
    gain_label.configure(text=f"Gain: {DIGITAL_GAIN:.2f}")
    save_runtime_config()


gamma_label = ctk.CTkLabel(control_frame, text=f"Gamma: {GAMMA_VALUE:.2f}")
gamma_label.grid(row=2, column=4, padx=5)


def update_gamma(val):
    global GAMMA_VALUE
    GAMMA_VALUE = float(val)
    gamma_label.configure(text=f"Gamma: {GAMMA_VALUE:.2f}")
    save_runtime_config()

def save_snapshot():
    if latest_overlay is None:
        camera_status_label.configure(text="Belum ada snapshot yang bisa disimpan.")
        return

    os.makedirs("snapshots", exist_ok=True)

    # Popup input nama folder dengan customtkinter
    dialog = ctk.CTkInputDialog(
        text="Masukkan nama folder untuk snapshot ini:",
        title="Nama Folder Snapshot"
    )
    folder_name = dialog.get_input()

    # Jika user menekan cancel / close
    if folder_name is None:
        camera_status_label.configure(text="Simpan snapshot dibatalkan.")
        return

    folder_name = folder_name.strip()
    if not folder_name:
        camera_status_label.configure(text="Nama folder tidak boleh kosong.")
        return

    # Sanitasi nama folder agar aman untuk nama direktori
    folder_name = re.sub(r'[\\/:*?"<>|]', "_", folder_name)

    save_dir = os.path.join("snapshots", folder_name)

    # Cegah folder lama tertimpa
    if os.path.exists(save_dir):
        camera_status_label.configure(
            text=f"Folder sudah ada: {save_dir}. Gunakan nama folder lain."
        )
        return

    os.makedirs(save_dir, exist_ok=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    overlay_path = os.path.join(save_dir, f"overlay_{timestamp}.png")
    right_path = os.path.join(save_dir, f"right_{timestamp}.png")
    params_path = os.path.join(save_dir, f"params_{timestamp}.json")

    cv2.imwrite(overlay_path, latest_overlay)

    if latest_heatmap is not None:
        cv2.imwrite(right_path, latest_heatmap)
    else:
        right_path = ""

    params = {
        "VEHICLE_CONF": VEHICLE_CONF,
        "WHEEL_CONF": WHEEL_CONF,
        "MAX_EPIPOLAR_DIFF_PX": MAX_EPIPOLAR_DIFF_PX,
        "GUIDE_MIN_DEPTH_CM": GUIDE_MIN_DEPTH_CM,
        "GUIDE_MAX_DEPTH_CM": GUIDE_MAX_DEPTH_CM,
        "EXPOSURE_EV": EXPOSURE_EV,
        "WHITE_BALANCE_SHIFT": WHITE_BALANCE_SHIFT,
        "DIGITAL_GAIN": DIGITAL_GAIN,
        "GAMMA_VALUE": GAMMA_VALUE,
    }

    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # Simpan nama folder ke Excel, bukan timestamp
    save_snapshot_to_excel(folder_name, overlay_path, right_path, params, latest_measurements)
    camera_status_label.configure(text=f"Snapshot tersimpan di folder: {save_dir}")

def toggle_flip_foh_roh():
    global flip_foh_roh
    flip_foh_roh = not flip_foh_roh
    flip_button.configure(text=f"Flip FOH/ROH: {'ON' if flip_foh_roh else 'OFF'}")

def save_snapshot_to_excel(folder_name, overlay_path, right_path, params, measurements):
    if not OPENPYXL_AVAILABLE:
        camera_status_label.configure(text="openpyxl belum terpasang. Install: pip install openpyxl")
        return

    os.makedirs("snapshots", exist_ok=True)
    headers = [
        "folder_name", "object_idx", "class_name", "z_cm", "oal_cm", "foh_cm",
        "wb_cm", "roh_cm", "oah_cm", "front_depth_cm", "rear_depth_cm",
        "vehicle_conf", "wheel_conf", "max_epipolar_diff_px",
        "guide_min_depth_cm", "guide_max_depth_cm",
        "exposure_ev", "white_balance_shift", "digital_gain", "gamma_value",
        "overlay_image", "right_image"
    ]

    if os.path.exists(EXCEL_LOG_PATH):
        wb = load_workbook(EXCEL_LOG_PATH)
        ws = wb.active
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "measurements"
        ws.append(headers)

    if not measurements:
        ws.append([
            folder_name, -1, "", "", "", "", "", "", "", "", "",
            params["VEHICLE_CONF"], params["WHEEL_CONF"], params["MAX_EPIPOLAR_DIFF_PX"],
            params["GUIDE_MIN_DEPTH_CM"], params["GUIDE_MAX_DEPTH_CM"],
            params["EXPOSURE_EV"], params["WHITE_BALANCE_SHIFT"], params["DIGITAL_GAIN"], params["GAMMA_VALUE"],
            overlay_path, right_path
        ])
    else:
        for idx, m in enumerate(measurements, start=1):
            ws.append([
                folder_name,
                idx,
                m.get("class_name", ""),
                m.get("z_cm", ""),
                m.get("oal_cm", ""),
                m.get("foh_cm", ""),
                m.get("wb_cm", ""),
                m.get("roh_cm", ""),
                m.get("oah_cm", ""),
                m.get("front_depth_cm", ""),
                m.get("rear_depth_cm", ""),
                params["VEHICLE_CONF"],
                params["WHEEL_CONF"],
                params["MAX_EPIPOLAR_DIFF_PX"],
                params["GUIDE_MIN_DEPTH_CM"],
                params["GUIDE_MAX_DEPTH_CM"],
                params["EXPOSURE_EV"],
                params["WHITE_BALANCE_SHIFT"],
                params["DIGITAL_GAIN"],
                params["GAMMA_VALUE"],
                overlay_path,
                right_path,
            ])

    wb.save(EXCEL_LOG_PATH)
    camera_status_label.configure(text=f"Snapshot tersimpan + log Excel: {EXCEL_LOG_PATH}")
    
vehicle_conf_slider = ctk.CTkSlider(control_frame, from_=0.05, to=0.60, number_of_steps=55, command=update_vehicle_conf)
vehicle_conf_slider.set(VEHICLE_CONF)
vehicle_conf_slider.grid(row=1, column=0, padx=10, pady=5)

wheel_conf_slider = ctk.CTkSlider(control_frame, from_=0.03, to=0.40, number_of_steps=37, command=update_wheel_conf)
wheel_conf_slider.set(WHEEL_CONF)
wheel_conf_slider.grid(row=1, column=1, padx=10, pady=5)

epipolar_slider = ctk.CTkSlider(control_frame, from_=5, to=150, number_of_steps=145, command=update_epipolar_tol)
epipolar_slider.set(MAX_EPIPOLAR_DIFF_PX)
epipolar_slider.grid(row=1, column=2, padx=10, pady=5)

guide_min_slider = ctk.CTkSlider(control_frame, from_=200, to=3000, number_of_steps=280, command=update_guide_min)
guide_min_slider.set(GUIDE_MIN_DEPTH_CM)
guide_min_slider.grid(row=1, column=4, padx=10, pady=5)

guide_max_slider = ctk.CTkSlider(control_frame, from_=300, to=4000, number_of_steps=370, command=update_guide_max)
guide_max_slider.set(GUIDE_MAX_DEPTH_CM)
guide_max_slider.grid(row=1, column=5, padx=10, pady=5)

exposure_ev_slider = ctk.CTkSlider(control_frame, from_=-2.0, to=2.0, number_of_steps=80, command=update_exposure_ev)
exposure_ev_slider.set(EXPOSURE_EV)
exposure_ev_slider.grid(row=3, column=0, padx=10, pady=5)

wb_shift_slider = ctk.CTkSlider(control_frame, from_=-1.0, to=1.0, number_of_steps=80, command=update_wb_shift)
wb_shift_slider.set(WHITE_BALANCE_SHIFT)
wb_shift_slider.grid(row=3, column=1, padx=10, pady=5)

gain_slider = ctk.CTkSlider(control_frame, from_=0.5, to=2.5, number_of_steps=80, command=update_digital_gain)
gain_slider.set(DIGITAL_GAIN)
gain_slider.grid(row=3, column=2, padx=10, pady=5)

gamma_slider = ctk.CTkSlider(control_frame, from_=0.6, to=1.8, number_of_steps=60, command=update_gamma)
gamma_slider.set(GAMMA_VALUE)
gamma_slider.grid(row=3, column=4, padx=10, pady=5)

save_button = ctk.CTkButton(control_frame, text="Save Snapshot", command=save_snapshot)
save_button.grid(row=1, column=8, padx=10, pady=10)

flip_button = ctk.CTkButton(
    control_frame,
    text=f"Flip FOH/ROH: {'ON' if flip_foh_roh else 'OFF'}",
    command=toggle_flip_foh_roh,
)
flip_button.grid(row=2, column=8, padx=10, pady=(0, 8))

fps_label = ctk.CTkLabel(control_frame, text="FPS: 0.00")
fps_label.grid(row=0, column=9, padx=5)

if GUIDE_MIN_DEPTH_CM >= GUIDE_MAX_DEPTH_CM - 50:
    GUIDE_MIN_DEPTH_CM = max(200.0, GUIDE_MAX_DEPTH_CM - 200.0)
vehicle_conf_slider.set(VEHICLE_CONF)
wheel_conf_slider.set(WHEEL_CONF)
epipolar_slider.set(MAX_EPIPOLAR_DIFF_PX)
guide_min_slider.set(GUIDE_MIN_DEPTH_CM)
guide_max_slider.set(GUIDE_MAX_DEPTH_CM)
exposure_ev_slider.set(EXPOSURE_EV)
wb_shift_slider.set(WHITE_BALANCE_SHIFT)
gain_slider.set(DIGITAL_GAIN)
gamma_slider.set(GAMMA_VALUE)
save_runtime_config()


def rectify_points(points, mtx, dist, rect_R, rect_P):
    if not points:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    rect = cv2.undistortPoints(arr, mtx, dist, R=rect_R, P=rect_P)
    return rect.reshape(-1, 2)


def match_sparse_depth(left_pt, left_rect_pt, right_pts, right_rect_pts, used_right_idx):
    if right_rect_pts.shape[0] == 0 or not calib_ready:
        return np.nan, None

    lx, ly = float(left_rect_pt[0]), float(left_rect_pt[1])
    best_score = 1e18
    best_idx = None
    best_depth_cm = np.nan

    for idx, rp in enumerate(right_rect_pts):
        if idx in used_right_idx:
            continue
        rx, ry = float(rp[0]), float(rp[1])
        ydiff = abs(ly - ry)
        if ydiff > MAX_EPIPOLAR_DIFF_PX:
            continue
        disp = lx - rx
        if disp < MIN_VALID_DISPARITY_PX or disp > MAX_VALID_DISPARITY_PX:
            continue

        depth_cm = (fx_rect * baseline_m / disp) * 100.0
        if depth_cm < MIN_VALID_SPARSE_DEPTH_CM or depth_cm > MAX_VALID_SPARSE_DEPTH_CM:
            continue

        score = ydiff + 0.01 * abs(disp)
        if score < best_score:
            best_score = score
            best_idx = idx
            best_depth_cm = depth_cm

    return best_depth_cm, best_idx


def pair_vehicles_rectified(vehicles_left, vehicles_right):
    if not calib_ready or not vehicles_left or not vehicles_right:
        return {}
    left_centers = [((vx1 + vx2) // 2, (vy1 + vy2) // 2) for vx1, vy1, vx2, vy2, _ in vehicles_left]
    right_centers = [((vx1 + vx2) // 2, (vy1 + vy2) // 2) for vx1, vy1, vx2, vy2, _ in vehicles_right]
    left_rect = rectify_points(left_centers, mtxL, distL, R1, P1)
    right_rect = rectify_points(right_centers, mtxR, distR, R2, P2)

    pairs = {}
    used_right = set()
    for li, lp in enumerate(left_rect):
        lx, ly = float(lp[0]), float(lp[1])
        best = None
        best_score = 1e18
        for ri, rp in enumerate(right_rect):
            if ri in used_right:
                continue
            rx, ry = float(rp[0]), float(rp[1])
            ydiff = abs(ly - ry)
            disp = lx - rx
            if ydiff > VEHICLE_PAIR_MAX_Y_DIFF_PX:
                continue
            if disp < MIN_VALID_DISPARITY_PX or disp > MAX_VALID_DISPARITY_PX:
                continue
            score = ydiff + 0.01 * abs(disp)
            if score < best_score:
                best_score = score
                best = ri
        if best is not None:
            pairs[li] = best
            used_right.add(best)
    return pairs


def refine_disparity_subpixel(grayL_rect, grayR_rect, left_rect_pt, right_rect_guess):
    xL, yL = int(round(left_rect_pt[0])), int(round(left_rect_pt[1]))
    xRg = int(round(right_rect_guess[0]))
    t = SUBPIX_TEMPLATE_HALF
    s = SUBPIX_SEARCH_HALF
    h, w = grayL_rect.shape[:2]

    if xL - t < 0 or xL + t >= w or yL - t < 0 or yL + t >= h:
        return np.nan

    template = grayL_rect[yL - t:yL + t + 1, xL - t:xL + t + 1]
    xs = max(t, xRg - s)
    xe = min(w - t - 1, xRg + s)
    if xe <= xs:
        return np.nan

    roi = grayR_rect[yL - t:yL + t + 1, xs - t:xe + t + 1]
    if roi.shape[0] != template.shape[0] or roi.shape[1] < template.shape[1]:
        return np.nan

    corr = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED).ravel()
    if corr.size == 0:
        return np.nan
    k = int(np.argmax(corr))
    xR = float(xs + k)

    # quadratic subpixel refinement around best match
    if 1 <= k < corr.size - 1:
        c0, c1, c2 = float(corr[k - 1]), float(corr[k]), float(corr[k + 1])
        denom = (c0 - 2.0 * c1 + c2)
        if abs(denom) > 1e-6:
            xR += 0.5 * (c0 - c2) / denom

    disp = float(left_rect_pt[0]) - xR
    if disp < MIN_VALID_DISPARITY_PX or disp > MAX_VALID_DISPARITY_PX:
        return np.nan
    return disp


def depth_cm_from_disparity(disp):
    if not np.isfinite(disp) or disp <= 0 or not calib_ready:
        return np.nan
    z = (fx_rect * baseline_m / disp) * 100.0
    if z < MIN_VALID_SPARSE_DEPTH_CM or z > MAX_VALID_SPARSE_DEPTH_CM:
        return np.nan
    return float(z)


def get_left_overlap_guide_bbox(img_w, img_h):
    if not calib_ready:
        return None
    zmin_m = GUIDE_MIN_DEPTH_CM / 100.0
    zmax_m = GUIDE_MAX_DEPTH_CM / 100.0
    if zmin_m <= 0 or zmax_m <= 0:
        return None
    dmax = (fx_rect * baseline_m) / zmin_m
    dmin = (fx_rect * baseline_m) / zmax_m
    left_x = int(max(0, min(img_w - 2, np.ceil(dmax))))
    right_x = int(max(left_x + 1, min(img_w - 1, img_w - 1 + np.floor(dmin))))
    return (left_x, 0, right_x, img_h - 1)


def bbox_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def ema_point(old_p, new_p, alpha=SMOOTH_ALPHA):
    ox, oy = old_p
    nx, ny = new_p
    return (
        int(round((1.0 - alpha) * ox + alpha * nx)),
        int(round((1.0 - alpha) * oy + alpha * ny)),
    )


def ema_box(old_b, new_b, alpha=SMOOTH_ALPHA):
    return (
        int(round((1.0 - alpha) * old_b[0] + alpha * new_b[0])),
        int(round((1.0 - alpha) * old_b[1] + alpha * new_b[1])),
        int(round((1.0 - alpha) * old_b[2] + alpha * new_b[2])),
        int(round((1.0 - alpha) * old_b[3] + alpha * new_b[3])),
    )


def stabilize_right_wheels(raw_wheels):
    global right_wheel_tracks
    new_tracks = []
    used_raw = set()

    for tr in right_wheel_tracks:
        best_idx = None
        best_d2 = 1e18
        tx, ty = tr["pt"]
        for i, rp in enumerate(raw_wheels):
            if i in used_raw:
                continue
            rx, ry = rp
            d2 = (tx - rx) * (tx - rx) + (ty - ry) * (ty - ry)
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i

        if best_idx is not None and best_d2 <= (WHEEL_MATCH_DIST_PX * WHEEL_MATCH_DIST_PX):
            used_raw.add(best_idx)
            tr["pt"] = ema_point(tr["pt"], raw_wheels[best_idx])
            tr["hits"] += 1
            tr["miss"] = 0
        else:
            tr["miss"] += 1

        if tr["miss"] <= TRACK_MAX_MISS:
            new_tracks.append(tr)

    for i, rp in enumerate(raw_wheels):
        if i not in used_raw:
            new_tracks.append({"pt": rp, "hits": 1, "miss": 0})

    right_wheel_tracks = new_tracks
    return [tr["pt"] for tr in right_wheel_tracks if tr["hits"] >= TRACK_MIN_HITS]


def stabilize_left_detections(raw_left):
    global left_tracks
    used_raw = set()
    new_tracks = []

    for tr in left_tracks:
        best_idx = None
        best_iou = 0.0
        for i, det in enumerate(raw_left):
            if i in used_raw:
                continue
            x1, y1, x2, y2, vclass, _, _, _, _, _ = det
            if vclass != tr["class_name"]:
                continue
            iou = bbox_iou_xyxy(tr["bbox"], (x1, y1, x2, y2))
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx is not None and best_iou >= TRACK_IOU_THRESH:
            used_raw.add(best_idx)
            x1, y1, x2, y2, vclass, fw, rw, zd, zf, zr = raw_left[best_idx]
            tr["bbox"] = ema_box(tr["bbox"], (x1, y1, x2, y2))
            tr["front_wheel"] = ema_point(tr["front_wheel"], fw)
            tr["rear_wheel"] = ema_point(tr["rear_wheel"], rw)
            if np.isfinite(zd):
                tr["obj_depth_cm"] = zd if not np.isfinite(tr["obj_depth_cm"]) else (1.0 - SMOOTH_ALPHA) * tr["obj_depth_cm"] + SMOOTH_ALPHA * zd
            if np.isfinite(zf):
                tr["front_depth_cm"] = zf
            if np.isfinite(zr):
                tr["rear_depth_cm"] = zr
            tr["hits"] += 1
            tr["miss"] = 0
        else:
            tr["miss"] += 1

        if tr["miss"] <= TRACK_MAX_MISS:
            new_tracks.append(tr)

    for i, det in enumerate(raw_left):
        if i in used_raw:
            continue
        x1, y1, x2, y2, vclass, fw, rw, zd, zf, zr = det
        new_tracks.append(
            {
                "bbox": (x1, y1, x2, y2),
                "class_name": vclass,
                "front_wheel": fw,
                "rear_wheel": rw,
                "obj_depth_cm": zd,
                "front_depth_cm": zf,
                "rear_depth_cm": zr,
                "hits": 1,
                "miss": 0,
            }
        )

    left_tracks = new_tracks
    candidates = [tr for tr in left_tracks if tr["hits"] >= TRACK_MIN_HITS]
    candidates.sort(key=lambda t: (t["hits"], -t["miss"]), reverse=True)
    kept = []
    for tr in candidates:
        tb = tr["bbox"]
        tc = tr["class_name"]
        if any(tc == kt["class_name"] and bbox_iou_xyxy(tb, kt["bbox"]) > 0.50 for kt in kept):
            continue
        kept.append(tr)

    out = []
    for tr in kept:
        x1, y1, x2, y2 = tr["bbox"]
        out.append(
            (
                x1,
                y1,
                x2,
                y2,
                tr["class_name"],
                tr["front_wheel"],
                tr["rear_wheel"],
                float(tr["obj_depth_cm"]) if np.isfinite(tr["obj_depth_cm"]) else np.nan,
                float(tr["front_depth_cm"]) if np.isfinite(tr["front_depth_cm"]) else np.nan,
                float(tr["rear_depth_cm"]) if np.isfinite(tr["rear_depth_cm"]) else np.nan,
            )
        )
    return out


def detect_vehicles_scaled(frame, input_size=VEHICLE_INPUT_SIZE):
    h, w = frame.shape[:2]
    infer_w = input_size
    infer_h = max(32, int(round((h / w) * input_size)))
    infer = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
    sx = w / infer_w
    sy = h / infer_h
    result = vehicle_model.predict(infer, conf=VEHICLE_CONF, verbose=False)[0]
    vehicles = []
    for box in result.boxes:
        label = int(box.cls[0])
        class_name = vehicle_model.names[label]
        if class_name not in ["car", "truck", "bus"]:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
        vehicles.append((x1, y1, x2, y2, class_name))
    dedup = []
    for cand in vehicles:
        cx1, cy1, cx2, cy2, ccls = cand
        duplicated = False
        for ex in dedup:
            ex1, ey1, ex2, ey2, ecls = ex
            if ccls == ecls and bbox_iou_xyxy((cx1, cy1, cx2, cy2), (ex1, ey1, ex2, ey2)) > 0.55:
                duplicated = True
                break
        if not duplicated:
            dedup.append(cand)
    return dedup


def detect_wheels_in_vehicle_rois(frame, vehicles):
    h, w = frame.shape[:2]
    wheels = []
    for vx1, vy1, vx2, vy2, _ in vehicles:
        bw = max(1, vx2 - vx1)
        bh = max(1, vy2 - vy1)
        pad_x = int(bw * WHEEL_ROI_PAD_RATIO)
        pad_y = int(bh * WHEEL_ROI_PAD_RATIO)
        x1 = max(0, vx1 - pad_x)
        y1 = max(0, vy1 - pad_y)
        x2 = min(w, vx2 + pad_x)
        y2 = min(h, vy2 + pad_y)
        if x2 - x1 < 24 or y2 - y1 < 24:
            continue

        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        infer_w = WHEEL_ROI_INPUT_SIZE
        infer_h = max(32, int(round((roi_h / roi_w) * WHEEL_ROI_INPUT_SIZE)))
        roi_infer = cv2.resize(roi, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        sx = roi_w / infer_w
        sy = roi_h / infer_h

        out = wheel_model.predict(roi_infer, conf=WHEEL_CONF, verbose=False)[0]
        for box in out.boxes:
            rx1, ry1, rx2, ry2 = map(float, box.xyxy[0].cpu().numpy())
            rx1, ry1, rx2, ry2 = int(rx1 * sx), int(ry1 * sy), int(rx2 * sx), int(ry2 * sy)
            cx = x1 + (rx1 + rx2) // 2
            cy = y1 + (ry1 + ry2) // 2
            # Wheel biasanya berada di area bawah kendaraan.
            if cy < vy1 + int(0.45 * bh):
                continue
            if 0 <= cx < w and 0 <= cy < h:
                conf = float(box.conf[0].item()) if box.conf is not None else 0.0
                wheels.append((cx, cy, conf))

    # deduplicate close centers from overlapping ROIs
    dedup = []
    for cx, cy, conf in sorted(wheels, key=lambda x: x[2], reverse=True):
        if not any((cx - px) ** 2 + (cy - py) ** 2 < 16 ** 2 for px, py, _ in dedup):
            dedup.append((cx, cy, conf))
    return dedup


def pick_vehicle_wheel_pair(vehicle_box, wheels_with_conf):
    vx1, vy1, vx2, vy2, _ = vehicle_box
    bw = max(1, vx2 - vx1)
    # kandidat roda dalam rentang vehicle
    candidates = [(wx, wy, c) for wx, wy, c in wheels_with_conf if (vx1 - 40) <= wx <= (vx2 + 40)]
    if len(candidates) < 2:
        return None

    # cari pasangan paling masuk akal: jarak X besar + Y relatif sejajar + confidence tinggi.
    best = None
    best_score = -1e9
    for i in range(len(candidates)):
        wx1, wy1, c1 = candidates[i]
        for j in range(i + 1, len(candidates)):
            wx2, wy2, c2 = candidates[j]
            dx = abs(wx2 - wx1)
            if dx < 0.20 * bw:
                continue
            dy = abs(wy2 - wy1)
            if dy > 0.20 * bw:
                continue
            score = dx - 1.2 * dy + 20.0 * (c1 + c2)
            if score > best_score:
                if wx1 <= wx2:
                    front = (wx1, wy1)
                    rear = (wx2, wy2)
                else:
                    front = (wx2, wy2)
                    rear = (wx1, wy1)
                best = (front, rear)
                best_score = score
    return best


def detect_vehicle_and_wheel_async(frame_left, frame_right):
    global detection_result
    if not detection_busy.acquire(blocking=False):
        return
    try:
        vehicles_left = detect_vehicles_scaled(frame_left)
        wheels_left = detect_wheels_in_vehicle_rois(frame_left, vehicles_left)

        vehicles_right = detect_vehicles_scaled(frame_right)
        wheels_right = detect_wheels_in_vehicle_rois(frame_right, vehicles_right)
        vehicle_pairs = pair_vehicles_rectified(vehicles_left, vehicles_right)

        left_detections = []
        wheels_right_pts = [(x, y) for x, y, _ in wheels_right]
        right_rect_pts = rectify_points(wheels_right_pts, mtxR, distR, R2, P2) if calib_ready else np.empty((0, 2), dtype=np.float32)
        wheels_left_pts = [(x, y) for x, y, _ in wheels_left]
        wheels_left_rect = rectify_points(wheels_left_pts, mtxL, distL, R1, P1) if calib_ready else np.empty((0, 2), dtype=np.float32)

        grayL_rect = grayR_rect = None
        if calib_ready:
            grayL_rect = cv2.cvtColor(cv2.remap(frame_left, mapLx, mapLy, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
            grayR_rect = cv2.cvtColor(cv2.remap(frame_right, mapRx, mapRy, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)

        for vx1, vy1, vx2, vy2, vclass in vehicles_left:
            pair = pick_vehicle_wheel_pair((vx1, vy1, vx2, vy2, vclass), wheels_left)
            if pair is None:
                continue
            front_wheel, rear_wheel = pair
            if calib_ready:
                # Restrict right wheel candidates to matched right vehicle ROI when available.
                right_candidates_idx = list(range(len(wheels_right_pts)))
                li = next((i for i, lv in enumerate(vehicles_left) if lv[0] == vx1 and lv[1] == vy1 and lv[2] == vx2 and lv[3] == vy2), None)
                if li is not None and li in vehicle_pairs:
                    rv = vehicles_right[vehicle_pairs[li]]
                    rvx1, rvy1, rvx2, rvy2, _ = rv
                    right_candidates_idx = [
                        i for i, (wx, wy) in enumerate(wheels_right_pts)
                        if (rvx1 - 40) <= wx <= (rvx2 + 40) and (rvy1 - 30) <= wy <= (rvy2 + 30)
                    ] or right_candidates_idx

                # map selected left wheels to rectified points from global list
                front_i = wheels_left_pts.index(front_wheel) if front_wheel in wheels_left_pts else None
                rear_i = wheels_left_pts.index(rear_wheel) if rear_wheel in wheels_left_pts else None
                if front_i is None or rear_i is None or wheels_left_rect.shape[0] == 0:
                    front_depth_cm = np.nan
                    rear_depth_cm = np.nan
                else:
                    left_rect_front = wheels_left_rect[front_i]
                    left_rect_rear = wheels_left_rect[rear_i]
                    used = set()

                    # initial right matches by epipolar proximity
                    def best_right_guess(left_rect_pt):
                        lx, ly = float(left_rect_pt[0]), float(left_rect_pt[1])
                        best_idx = None
                        best_score = 1e18
                        for i in right_candidates_idx:
                            if i in used:
                                continue
                            rx, ry = right_rect_pts[i]
                            ydiff = abs(float(ly - ry))
                            disp = float(lx - rx)
                            if ydiff > MAX_EPIPOLAR_DIFF_PX:
                                continue
                            if disp < MIN_VALID_DISPARITY_PX or disp > MAX_VALID_DISPARITY_PX:
                                continue
                            score = ydiff + 0.01 * abs(disp)
                            if score < best_score:
                                best_score = score
                                best_idx = i
                        return best_idx

                    front_guess = best_right_guess(left_rect_front)
                    if front_guess is not None:
                        used.add(front_guess)
                        disp_f = refine_disparity_subpixel(grayL_rect, grayR_rect, left_rect_front, right_rect_pts[front_guess])
                        front_depth_cm = depth_cm_from_disparity(disp_f)
                    else:
                        front_depth_cm = np.nan

                    rear_guess = best_right_guess(left_rect_rear)
                    if rear_guess is not None:
                        used.add(rear_guess)
                        disp_r = refine_disparity_subpixel(grayL_rect, grayR_rect, left_rect_rear, right_rect_pts[rear_guess])
                        rear_depth_cm = depth_cm_from_disparity(disp_r)
                    else:
                        rear_depth_cm = np.nan

                valid_depths = [d for d in [front_depth_cm, rear_depth_cm] if np.isfinite(d)]
                obj_depth_cm = float(np.median(valid_depths)) if valid_depths else np.nan
            else:
                front_depth_cm = np.nan
                rear_depth_cm = np.nan
                obj_depth_cm = np.nan
            left_detections.append((
                vx1, vy1, vx2, vy2, vclass, front_wheel, rear_wheel,
                obj_depth_cm, front_depth_cm, rear_depth_cm
            ))

        stable_left = stabilize_left_detections(left_detections)
        stable_right = stabilize_right_wheels(wheels_right_pts)

        with detection_lock:
            detection_result = {
                "left": stable_left,
                "right_wheels": stable_right,
            }
    finally:
        detection_busy.release()


def render_to_panel(panel: ctk.CTkLabel, bgr: np.ndarray):
    panel_w = max(2, panel.winfo_width())
    panel_h = max(2, panel.winfo_height())
    fitted = resize_with_aspect(bgr, panel_w, panel_h)
    if fitted is None:
        return
    rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    ctk_img = ctk.CTkImage(light_image=pil, size=pil.size)
    panel.configure(image=ctk_img, text="")
    panel.image = ctk_img


def update():
    global latest_overlay, latest_heatmap, latest_measurements
    if not app_running:
        return

    okL, frameL, _ = camL.read() if camL is not None else (False, None, 0.0)
    okR, frameR, _ = camR.read() if camR is not None else (False, None, 0.0)
    if not okL or not okR or frameL is None or frameR is None:
        frame_left.configure(text="Left Preview (no frames)", image=None)
        frame_left.image = None
        values_box.configure(state="normal")
        values_box.delete("1.0", "end")
        values_box.insert("1.0", "No frames")
        values_box.configure(state="disabled")
        app.after(33, update)
        return

    procL = apply_camera_tuning(frameL)
    procR = apply_camera_tuning(frameR)

    overlay = procL.copy()
    right_preview = procR.copy()

    guide = get_left_overlap_guide_bbox(overlay.shape[1], overlay.shape[0])
    if guide is not None:
        gx1, gy1, gx2, gy2 = guide
        cv2.rectangle(overlay, (gx1, gy1), (gx2, gy2), (255, 140, 0), 2)
        cv2.putText(
            overlay,
            f"Guide overlap R-cam ({int(GUIDE_MIN_DEPTH_CM/100)}-{int(GUIDE_MAX_DEPTH_CM/100)} m)",
            (gx1 + 8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 140, 0),
            2,
        )

    if not detection_busy.locked():
        threading.Thread(
            target=detect_vehicle_and_wheel_async,
            args=(procL.copy(), procR.copy()),
            daemon=True
        ).start()

    with detection_lock:
        left_items = detection_result.get("left", []) if isinstance(detection_result, dict) else []
        right_wheels = detection_result.get("right_wheels", []) if isinstance(detection_result, dict) else []
        measured_rows = []
        info_lines = []
        for x1, y1, x2, y2, class_name, front_wheel, rear_wheel, obj_depth_cm, front_depth_cm, rear_depth_cm in left_items:
            z_center = obj_depth_cm
            if not np.isfinite(z_center) or z_center <= 0:
                continue

            vx1, vy1, vx2, vy2 = x1, y1, x2, y2
            fw_x, fw_y = front_wheel
            rw_x, rw_y = rear_wheel

            f = fx_rect if "fx_rect" in globals() and fx_rect > 0 else 1.0
            fy = float(P1[1, 1]) if "P1" in globals() and P1[1, 1] > 0 else f

            foh = (fw_x - vx1) * z_center / f
            wb = (rw_x - fw_x) * z_center / f
            roh = (vx2 - rw_x) * z_center / f
            oal = (vx2 - vx1) * z_center / f
            oah = (vy2 - vy1) * z_center / fy
            if flip_foh_roh:
                disp_foh = roh
                disp_roh = foh
                foh_label = "R.O.H"
                roh_label = "F.O.H"
            else:
                disp_foh = foh
                disp_roh = roh
                foh_label = "F.O.H"
                roh_label = "R.O.H"

            x_start = vx1
            y_top = vy1
            y_bottom = vy1 + 30

            pixel_oal = (oal * f) / z_center
            x_end_oal = x_start + int(pixel_oal)
            cv2.line(overlay, (x_start, y_top), (x_end_oal, y_top), (255, 0, 0), 2)
            cv2.putText(overlay, f"O.A.L: {oal:.1f} cm", (x_start, y_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            pixel_foh = (disp_foh * f) / z_center
            x_end_foh = x_start + int(pixel_foh)
            cv2.line(overlay, (x_start, y_bottom), (x_end_foh, y_bottom), (0, 255, 0), 2)
            cv2.putText(overlay, f"{foh_label}: {disp_foh:.1f} cm", (x_start, y_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            x_start_wb = x_end_foh
            pixel_wb = (wb * f) / z_center
            x_end_wb = x_start_wb + int(pixel_wb)
            cv2.line(overlay, (x_start_wb, y_bottom + 20), (x_end_wb, y_bottom + 20), (255, 255, 255), 2)
            cv2.putText(overlay, f"W.B: {wb:.1f} cm", (x_start_wb, y_bottom + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            x_start_roh = x_end_wb
            pixel_roh = (disp_roh * f) / z_center
            x_end_roh = x_start_roh + int(pixel_roh)
            cv2.line(overlay, (x_start_roh, y_bottom + 40), (x_end_roh, y_bottom + 40), (0, 165, 255), 2)
            cv2.putText(overlay, f"{roh_label}: {disp_roh:.1f} cm", (x_start_roh, y_bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

            pixel_oah = (oah * fy) / z_center
            y_end_oah = vy1 + int(pixel_oah)
            cv2.line(overlay, (x_start, vy1), (x_start, y_end_oah), (0, 0, 255), 2)
            cv2.putText(overlay, f"O.A.H: {oah:.1f} cm", (x_start - 80, (vy1 + y_end_oah) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(overlay, f"Z: {z_center:.1f} cm", (x_start, max(15, y_top - 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            measured_rows.append(
                {
                    "class_name": class_name,
                    "z_cm": round(float(z_center), 2),
                    "oal_cm": round(float(oal), 2),
                    "foh_cm": round(float(disp_foh), 2),
                    "wb_cm": round(float(wb), 2),
                    "roh_cm": round(float(disp_roh), 2),
                    "oah_cm": round(float(oah), 2),
                    "front_depth_cm": round(float(front_depth_cm), 2) if np.isfinite(front_depth_cm) else "",
                    "rear_depth_cm": round(float(rear_depth_cm), 2) if np.isfinite(rear_depth_cm) else "",
                }
            )
            info_lines.append(
                f"{len(measured_rows)}. {class_name}\n"
                f"   Z={z_center:.1f}  OAL={oal:.1f}  {foh_label}={disp_foh:.1f}\n"
                f"   WB={wb:.1f}  {roh_label}={disp_roh:.1f}  OAH={oah:.1f} (cm)"
            )
        for wx, wy in right_wheels:
            cv2.circle(right_preview, (wx, wy), 5, (0, 255, 255), -1)
        latest_measurements = measured_rows

    latest_overlay = overlay.copy()
    latest_heatmap = right_preview.copy()

    render_to_panel(frame_left, overlay)
    values_box.configure(state="normal")
    values_box.delete("1.0", "end")
    if info_lines:
        values_box.insert("1.0", "\n\n".join(info_lines))
    else:
        values_box.insert("1.0", "Belum ada objek valid dengan depth.")
    values_box.configure(state="disabled")

    current_time = time.time()
    elapsed = current_time - fps_counter["last_time"]
    if elapsed > 0:
        fps_counter["fps"] = 1.0 / elapsed
    fps_counter["last_time"] = current_time
    fps_label.configure(text=f"FPS: {fps_counter['fps']:.2f}")

    app.after(33, update)


def on_close():
    global app_running
    app_running = False
    save_runtime_config()
    stop_cameras()
    cv2.destroyAllWindows()
    app.destroy()


if device_labels:
    initialize_cameras(selected_left_label, selected_right_label)
else:
    camera_status_label.configure(text="FFmpeg device tidak ditemukan.")

app.protocol("WM_DELETE_WINDOW", on_close)
update()
app.mainloop()
