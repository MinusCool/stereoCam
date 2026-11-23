import os
import re
import time
import threading
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30
OUTPUT_DIR = "images"

CHECKERBOARD = (9, 6)
DETECT_MAX_W = 640

AUTO_CAPTURE_COOLDOWN_SEC = 0.8  # cegah capture beruntun
BLUR_MIN_VAR = 80.0              # minimum var(Laplacian) agar dianggap tidak blur
STABLE_MOTION_PX = 2.0           # pergeseran center corners max antar frame (px)
STABLE_FRAMES_REQUIRED = 4       # jumlah frame stabil sebelum auto-freeze


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_ffmpeg_list_dshow_devices() -> str:
    cmd = ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return p.stderr


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


def resize_with_aspect(bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
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


def downscale_for_detect(bgr: np.ndarray, max_w: int = DETECT_MAX_W):
    h, w = bgr.shape[:2]
    if w <= max_w:
        return bgr
    scale = max_w / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


@dataclass
class FramePacket:
    frame: np.ndarray = None      # full-res
    preview: np.ndarray = None    # preview frame
    ok: bool = False
    ts: float = 0.0
    chess_ok: bool = False


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

    def start(self) -> bool:
        if self.running:
            return True

        cmd = [
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
        ]

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**7
            )
        except FileNotFoundError:
            return False

        if self.proc.stdout is None:
            return False

        self.running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self.running = False
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

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


class StereoCaptureApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Stereo Capture (FFmpeg DirectShow Device Path)")
        self.root.geometry("1200x650")

        ensure_dir(OUTPUT_DIR)

        self.lock = threading.Lock()
        self.running = False

        self.camL: Optional[FFmpegCamera] = None
        self.camR: Optional[FFmpegCamera] = None

        self.packetL = FramePacket()
        self.packetR = FramePacket()

        self.detect_var = tk.BooleanVar(value=True)   # default ON, karena kamu mau auto
        self.require_chess_for_save_var = tk.BooleanVar(value=False)

        self.status = tk.StringVar(value="Ready.")
        self.pair_id = self._get_next_pair_id()

        self.devices = []
        self.device_labels = []
        self.label_to_alt = {}

        # pending + freeze state
        self.pending_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None  # full-res L,R
        self.pending_preview: Optional[Tuple[np.ndarray, np.ndarray]] = None  # preview L,R (untuk display freeze)
        self.pending_ts: float = 0.0
        self.freeze_preview: bool = False

        self._auto_last_capture_ts: float = 0.0
        self._auto_prev_both_ok: bool = False
        self._stable_counter: int = 0
        self._prev_centerL: Optional[np.ndarray] = None
        self._prev_centerR: Optional[np.ndarray] = None

        self._build_ui()
        self.refresh_devices()

        self.root.after(33, self._ui_update_loop)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _get_next_pair_id(self) -> int:
        max_id = -1
        for fn in os.listdir(OUTPUT_DIR):
            if fn.startswith("chessboard-L") and fn.endswith(".png"):
                try:
                    num = int(fn.replace("chessboard-L", "").replace(".png", ""))
                    max_id = max(max_id, num)
                except:
                    pass
        return max_id + 1

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(top, text="Refresh (FFmpeg)", command=self.refresh_devices).pack(side=tk.LEFT, padx=6)

        ttk.Label(top, text="Left:").pack(side=tk.LEFT, padx=(16, 4))
        self.left_var = tk.StringVar(value="")
        self.left_combo = ttk.Combobox(top, textvariable=self.left_var, width=45, state="readonly")
        self.left_combo.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(top, text="Right:").pack(side=tk.LEFT, padx=(0, 4))
        self.right_var = tk.StringVar(value="")
        self.right_combo = ttk.Combobox(top, textvariable=self.right_var, width=45, state="readonly")
        self.right_combo.pack(side=tk.LEFT, padx=(0, 12))

        self.start_btn = ttk.Button(top, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=6)

        self.stop_btn = ttk.Button(top, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=6)

        self.save_btn = ttk.Button(top, text="Save Pair", command=self.save_pair, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=6)

        self.cancel_btn = ttk.Button(top, text="Cancel Save", command=self.cancel_save, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=6)

        options = ttk.Frame(self.root, padding=(8, 0))
        options.pack(side=tk.TOP, fill=tk.X)

        ttk.Checkbutton(
            options,
            text="Detect Chessboard (Auto Freeze)",
            variable=self.detect_var
        ).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Checkbutton(
            options,
            text="Require chessboard to save (manual mode)",
            variable=self.require_chess_for_save_var
        ).pack(side=tk.LEFT)

        ttk.Label(self.root, textvariable=self.status, padding=(8, 0)).pack(side=tk.TOP, fill=tk.X)

        mid = ttk.Frame(self.root, padding=8)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)
        mid.rowconfigure(0, weight=1)

        self.left_panel = ttk.Label(mid, text="Left Preview", anchor="center")
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.right_panel = ttk.Label(mid, text="Right Preview", anchor="center")
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        self._tkimgL = None
        self._tkimgR = None

    def refresh_devices(self):
        if self.running:
            messagebox.showinfo("Running", "Stop dulu sebelum refresh.")
            return

        try:
            txt = run_ffmpeg_list_dshow_devices()
            devices = parse_dshow_video_devices(txt)
        except FileNotFoundError:
            messagebox.showerror("FFmpeg not found", "FFmpeg tidak ditemukan di PATH.")
            return

        if not devices:
            self.status.set("Tidak ada device video ditemukan dari FFmpeg.")
            return

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

        self.devices = devices
        self.device_labels = labels
        self.label_to_alt = label_to_alt

        self.left_combo["values"] = labels
        self.right_combo["values"] = labels

        logi_labels = [lb for lb in labels if "logi c270" in lb.lower()]
        if len(logi_labels) >= 2:
            self.left_var.set(logi_labels[0])
            self.right_var.set(logi_labels[1])
        else:
            self.left_var.set(labels[0])
            self.right_var.set(labels[1] if len(labels) > 1 else labels[0])

        self.status.set("Devices loaded from FFmpeg. Tips: pilih 2x Logi C270 (#1 & #2).")

    def start(self):
        if self.running:
            return

        left_label = self.left_var.get()
        right_label = self.right_var.get()
        if left_label not in self.label_to_alt or right_label not in self.label_to_alt:
            messagebox.showwarning("Select camera", "Pilih kamera Left dan Right dulu.")
            return

        altL = self.label_to_alt[left_label]
        altR = self.label_to_alt[right_label]

        self.camL = FFmpegCamera(altL, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS)
        self.camR = FFmpegCamera(altR, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS)

        ok1 = self.camL.start()
        ok2 = self.camR.start()
        if not ok1 or not ok2:
            self._stop_cams()
            messagebox.showerror(
                "Open failed",
                "Gagal membuka salah satu kamera via FFmpeg.\n"
                "Pastikan ffmpeg ada di PATH dan kamera tidak dipakai aplikasi lain."
            )
            return

        # reset state
        self.pending_pair = None
        self.pending_preview = None
        self.pending_ts = 0.0
        self.freeze_preview = False
        self._auto_last_capture_ts = 0.0
        self._auto_prev_both_ok = False
        self._stable_counter = 0
        self._prev_centerL = None
        self._prev_centerR = None

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._update_buttons()

        threading.Thread(target=self._capture_loop, daemon=True).start()
        self.status.set(f"Running 720p. Pair ID: {self.pair_id}")

    def stop(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self._stop_cams()

        self.pending_pair = None
        self.pending_preview = None
        self.pending_ts = 0.0
        self.freeze_preview = False

        self._stable_counter = 0
        self._prev_centerL = None
        self._prev_centerR = None

        self._update_buttons()
        self.status.set("Stopped.")

    def _stop_cams(self):
        try:
            if self.camL is not None:
                self.camL.stop()
        except Exception:
            pass
        try:
            if self.camR is not None:
                self.camR.stop()
        except Exception:
            pass
        self.camL = None
        self.camR = None

    def _detect_chessboard_fast(self, bgr: np.ndarray):
        small = downscale_for_detect(bgr, DETECT_MAX_W)
        overlay = small.copy()
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        center = None
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(overlay, CHECKERBOARD, corners2, found)
            center = corners2.reshape(-1, 2).mean(axis=0)
        return found, overlay, center

    def _blur_score(self, bgr: np.ndarray) -> float:
        small = downscale_for_detect(bgr, DETECT_MAX_W)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _update_buttons(self):
        # Save/Cancel aktif jika freeze/pending ada
        if self.pending_pair is not None:
            self.save_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.NORMAL)
        else:
            self.cancel_btn.config(state=tk.DISABLED)
            # Manual mode masih boleh save current frame
            if self.running and self.require_chess_for_save_var.get():
                self.save_btn.config(state=tk.NORMAL)
            else:
                self.save_btn.config(state=tk.DISABLED)

    def _freeze_with_pending(self, frameL: np.ndarray, frameR: np.ndarray, prevL: np.ndarray, prevR: np.ndarray):
        # jangan overwrite pending
        if self.pending_pair is not None:
            return
        self.pending_pair = (frameL.copy(), frameR.copy())
        self.pending_preview = (prevL.copy(), prevR.copy())
        self.pending_ts = time.time()
        self.freeze_preview = True
        self._update_buttons()
        self.status.set("Detected! Preview frozen. Click 'Save Pair' or 'Cancel Save'.")

    def _capture_loop(self):
        last = time.time()
        fps = 0.0
        stale_warn = 0.0

        while self.running:
            okL, frameL, tsL = self.camL.read() if self.camL else (False, None, 0.0)
            okR, frameR, tsR = self.camR.read() if self.camR else (False, None, 0.0)
            now = time.time()

            if not okL or not okR or frameL is None or frameR is None:
                if now - stale_warn > 1.0:
                    self.status.set("Menunggu frame... (pastikan 2 kamera tidak dipakai aplikasi lain)")
                    stale_warn = now
                time.sleep(0.01)
                continue

            # kalau freeze, jangan update packet preview lagi (biar tetap frozen)
            if self.freeze_preview:
                time.sleep(0.01)
                continue

            dt = now - last
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last = now

            chessL = chessR = False
            centerL = centerR = None
            prevL = frameL
            prevR = frameR
            blurL = blurR = 0.0
            sharp_ok = False
            stable_ok = False

            if self.detect_var.get() or self.require_chess_for_save_var.get():
                chessL, prevL, centerL = self._detect_chessboard_fast(frameL)
                chessR, prevR, centerR = self._detect_chessboard_fast(frameR)

            both = chessL and chessR

            if both:
                blurL = self._blur_score(frameL)
                blurR = self._blur_score(frameR)
                sharp_ok = (blurL >= BLUR_MIN_VAR) and (blurR >= BLUR_MIN_VAR)

                if self._prev_centerL is not None and self._prev_centerR is not None and centerL is not None and centerR is not None:
                    motionL = np.linalg.norm(centerL - self._prev_centerL)
                    motionR = np.linalg.norm(centerR - self._prev_centerR)
                    stable_ok = (motionL <= STABLE_MOTION_PX) and (motionR <= STABLE_MOTION_PX)
                else:
                    stable_ok = False

                self._prev_centerL = centerL
                self._prev_centerR = centerR

                if sharp_ok and stable_ok:
                    self._stable_counter += 1
                else:
                    self._stable_counter = 0
            else:
                self._stable_counter = 0
                self._prev_centerL = None
                self._prev_centerR = None

            ready_for_freeze = both and sharp_ok and (self._stable_counter >= STABLE_FRAMES_REQUIRED)

            # Freeze trigger: rising edge + cooldown
            if self.detect_var.get():
                rising_edge = ready_for_freeze and (not self._auto_prev_both_ok)
                self._auto_prev_both_ok = ready_for_freeze

                if rising_edge and (now - self._auto_last_capture_ts) >= AUTO_CAPTURE_COOLDOWN_SEC:
                    self._auto_last_capture_ts = now
                    self._freeze_with_pending(frameL, frameR, prevL, prevR)

            with self.lock:
                self.packetL = FramePacket(frame=frameL, preview=prevL, ok=True, ts=tsL, chess_ok=chessL)
                self.packetR = FramePacket(frame=frameR, preview=prevR, ok=True, ts=tsR, chess_ok=chessR)

            self.status.set(
                f"Running | FPS~{fps:.1f} | Pair ID: {self.pair_id} | "
                f"L:{chessL} R:{chessR} | sharp L:{blurL:.0f} R:{blurR:.0f} | "
                f"stable:{self._stable_counter}/{STABLE_FRAMES_REQUIRED} | "
                f"age L:{now-tsL:.2f}s R:{now-tsR:.2f}s"
            )

            time.sleep(0.001)

    def _ui_update_loop(self):
        # jika freeze: tampilkan pending_preview (beku)
        if self.freeze_preview and self.pending_preview is not None:
            prevL, prevR = self.pending_preview
            showL_src = prevL
            showR_src = prevR
        else:
            with self.lock:
                pL = self.packetL
                pR = self.packetR
            showL_src = pL.preview if (pL.ok and pL.preview is not None) else None
            showR_src = pR.preview if (pR.ok and pR.preview is not None) else None

        wL = max(2, self.left_panel.winfo_width())
        hL = max(2, self.left_panel.winfo_height())
        wR = max(2, self.right_panel.winfo_width())
        hR = max(2, self.right_panel.winfo_height())

        if showL_src is not None:
            showL = resize_with_aspect(showL_src, wL, hL)
            self._tkimgL = self._bgr_to_tk(showL)
            self.left_panel.configure(image=self._tkimgL, text="")
        else:
            self.left_panel.configure(text="Left Preview (no frames)", image="")

        if showR_src is not None:
            showR = resize_with_aspect(showR_src, wR, hR)
            self._tkimgR = self._bgr_to_tk(showR)
            self.right_panel.configure(image=self._tkimgR, text="")
        else:
            self.right_panel.configure(text="Right Preview (no frames)", image="")

        self._update_buttons()
        self.root.after(33, self._ui_update_loop)

    def _bgr_to_tk(self, bgr: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def save_pair(self):
        # Case 1: save pending (freeze)
        if self.pending_pair is not None:
            L, R = self.pending_pair
            left_path = os.path.join(OUTPUT_DIR, f"chessboard-L{self.pair_id}.png")
            right_path = os.path.join(OUTPUT_DIR, f"chessboard-R{self.pair_id}.png")
            cv2.imwrite(left_path, L)
            cv2.imwrite(right_path, R)

            self.pending_pair = None
            self.pending_preview = None
            self.pending_ts = 0.0
            self.freeze_preview = False
            self.pair_id += 1
            self._update_buttons()
            self.status.set(f"Saved pair {self.pair_id - 1} (720p). Preview resumed.")
            return

        # Case 2: manual save (optional)
        with self.lock:
            pL = self.packetL
            pR = self.packetR

        if not (pL.ok and pR.ok and pL.frame is not None and pR.frame is not None):
            messagebox.showwarning("No frame", "Frame belum tersedia dari kedua kamera.")
            return

        if self.require_chess_for_save_var.get() and not (pL.chess_ok and pR.chess_ok):
            messagebox.showwarning("Chessboard not detected", "Chessboard harus terdeteksi di kedua kamera untuk save.")
            return

        left_path = os.path.join(OUTPUT_DIR, f"chessboard-L{self.pair_id}.png")
        right_path = os.path.join(OUTPUT_DIR, f"chessboard-R{self.pair_id}.png")
        cv2.imwrite(left_path, pL.frame)
        cv2.imwrite(right_path, pR.frame)
        self.pair_id += 1
        self.status.set(f"Saved pair {self.pair_id - 1} (720p)")

    def cancel_save(self):
        # buang pending dan resume live preview
        if self.pending_pair is None:
            return
        self.pending_pair = None
        self.pending_preview = None
        self.pending_ts = 0.0
        self.freeze_preview = False
        self._update_buttons()
        self.status.set("Cancelled. Preview resumed.")

    def on_close(self):
        self.running = False
        self._stop_cams()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    StereoCaptureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
