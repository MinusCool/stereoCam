"""Stereo C270 simulator and launcher; existing project files stay untouched.

Run this file to select a photo and launch main.py, or use --help for CLI options.
The photo is a texture on a plane in the ORIGINAL left-camera coordinate system
(X right, Y down, Z forward, metres). A single photo cannot supply true 3D shape.
Both raw camera views include their calibrated intrinsics and lens distortion.

The process-local FFmpeg bridge adds two selectable cameras to the unmodified
application. These endpoints are not registered as Windows camera devices.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "simulator_images"
CALIBRATION_PATH = ROOT / "model" / "stereo_calib_result.npz"
WIDTH, HEIGHT, FPS = 1280, 720, 30
CAMERA_NAMES = {
    "left": "Logi C270 Simulator Left",
    "right": "Logi C270 Simulator Right",
}
ENDPOINTS = {side: f"stereo-sim://{side}" for side in CAMERA_NAMES}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class Scene:
    image: Path
    distance_m: float = 8.0
    plane_width_m: float = 6.0
    yaw_deg: float = 0.0
    center_x_m: float = 0.0
    center_y_m: float = 0.0

    def validate(self):
        numbers = (self.distance_m, self.plane_width_m, self.yaw_deg,
                   self.center_x_m, self.center_y_m)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError("Semua parameter harus berupa angka yang finite.")
        if self.distance_m <= 0 or self.plane_width_m <= 0:
            raise ValueError("Jarak dan lebar bidang harus lebih besar dari nol.")
        if abs(self.yaw_deg) >= 80:
            raise ValueError("Sudut bidang harus di antara -80 dan 80 derajat.")
        if not self.image.is_file():
            raise ValueError(f"Gambar tidak ditemukan: {self.image}")

    def plane(self, image_shape):
        angle = math.radians(self.yaw_deg)
        u_axis = np.array([math.cos(angle), 0.0, -math.sin(angle)])
        v_axis = np.array([0.0, 1.0, 0.0])
        normal = np.cross(u_axis, v_axis)
        center = np.array([self.center_x_m, self.center_y_m, self.distance_m])
        plane_height = self.plane_width_m * image_shape[0] / image_shape[1]
        return center, u_axis, v_axis, normal, plane_height


class StereoRenderer:
    """Inverse ray/plane intersection, producing distorted ORIGINAL camera frames."""

    def __init__(self, calibration_path=CALIBRATION_PATH):
        self.calibration_path = Path(calibration_path).resolve()
        required = ("mtxL", "distL", "mtxR", "distR", "R", "T",
                    "R1", "R2", "P1", "P2")
        with np.load(self.calibration_path, allow_pickle=False) as data:
            missing = set(required) - set(data.files)
            if missing:
                raise ValueError(f"Kalibrasi tidak lengkap: {sorted(missing)}")
            self.calib = {key: data[key].astype(np.float64) for key in required}
        for key, value in self.calib.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nilai kalibrasi {key} tidak valid.")
        for key in ("mtxL", "mtxR", "R", "R1", "R2"):
            if self.calib[key].shape != (3, 3):
                raise ValueError(f"Bentuk matriks {key} harus 3x3.")
        for key in ("P1", "P2"):
            if self.calib[key].shape != (3, 4):
                raise ValueError(f"Bentuk matriks {key} harus 3x4.")
        if self.calib["T"].size != 3:
            raise ValueError("Vektor T harus berisi tiga nilai.")
        self.calib["T"] = self.calib["T"].reshape(3)
        self.baseline_m = float(np.linalg.norm(self.calib["T"]))
        if self.baseline_m <= 0:
            raise ValueError("Baseline kalibrasi harus lebih besar dari nol.")
        for key in ("mtxL", "mtxR"):
            if self.calib[key][0, 0] <= 0 or self.calib[key][1, 1] <= 0:
                raise ValueError(f"Focal length {key} harus positif.")
        # The original NPZ has no imageSize field. capture.py and main.py both
        # explicitly use 1280x720, so keep exactly that resolution here as well.
        yy, xx = np.indices((HEIGHT, WIDTH), dtype=np.float64)
        pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
        self.rays = {}
        self.centers = {"left": np.zeros(3),
                        "right": -self.calib["R"].T @ self.calib["T"]}
        for side, suffix in (("left", "L"), ("right", "R")):
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-10)
            if hasattr(cv2, "undistortPointsIter"):  # OpenCV 4.x
                xy = cv2.undistortPointsIter(
                    pixels, self.calib[f"mtx{suffix}"], self.calib[f"dist{suffix}"],
                    None, None, criteria,
                )
            else:  # OpenCV 5.x folds the iterative overload into this function.
                xy = cv2.undistortPoints(
                    pixels, self.calib[f"mtx{suffix}"], self.calib[f"dist{suffix}"],
                    criteria=criteria,
                )
            xy = xy.reshape(-1, 2)
            rays = np.column_stack((xy, np.ones(len(xy))))
            # Calibration uses X_right = R @ X_left + T. Transform right rays
            # back into the left coordinate system (row-vector convention).
            self.rays[side] = rays if side == "left" else rays @ self.calib["R"]

    def render(self, scene: Scene):
        scene.validate()
        # imdecode/fromfile handles Windows Unicode paths as well.
        photo = cv2.imdecode(np.fromfile(scene.image, dtype=np.uint8), cv2.IMREAD_COLOR)
        if photo is None:
            raise ValueError(f"File tidak dapat dibaca sebagai gambar: {scene.image}")
        center, u_axis, v_axis, normal, plane_height = scene.plane(photo.shape)
        corners = np.array([
            center + su * u_axis * scene.plane_width_m / 2 + sv * v_axis * plane_height / 2
            for su in (-1, 1) for sv in (-1, 1)
        ])
        right_corners = corners @ self.calib["R"].T + self.calib["T"]
        if min(corners[:, 2].min(), right_corners[:, 2].min()) <= 0.05:
            raise ValueError("Bidang melewati/belakang kamera. Tambahkan jarak atau kurangi sudut.")
        frames = {}
        coverage = {}
        for side in CAMERA_NAMES:
            camera_center = self.centers[side]
            rays = self.rays[side]
            denom = rays @ normal
            scale = np.full(len(rays), np.nan)
            np.divide((center - camera_center) @ normal, denom, out=scale,
                      where=np.abs(denom) > 1e-12)
            points = camera_center + rays * scale[:, None] - center
            map_x = ((points @ u_axis) / scene.plane_width_m + 0.5) * (photo.shape[1] - 1)
            map_y = ((points @ v_axis) / plane_height + 0.5) * (photo.shape[0] - 1)
            valid = ((scale > 0) & np.isfinite(map_x) & np.isfinite(map_y)
                     & (map_x >= 0) & (map_x <= photo.shape[1] - 1)
                     & (map_y >= 0) & (map_y <= photo.shape[0] - 1))
            map_x[~valid] = -1
            map_y[~valid] = -1
            frames[side] = cv2.remap(
                photo, map_x.reshape(HEIGHT, WIDTH).astype(np.float32),
                map_y.reshape(HEIGHT, WIDTH).astype(np.float32), cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(32, 32, 32),
            )
            coverage[side] = round(float(valid.mean()), 4)
            if not valid.any():
                raise ValueError(f"Foto tidak terlihat di kamera {side}; periksa posisi bidang.")
        metadata = {
            "image": str(scene.image.resolve()),
            "calibration": str(self.calibration_path),
            "resolution": [WIDTH, HEIGHT], "fps": FPS,
            "baseline_m": self.baseline_m,
            "right_camera_center_in_left_m": self.centers["right"].tolist(),
            "plane_center_in_left_m": center.tolist(),
            "plane_width_m": scene.plane_width_m,
            "plane_height_m": plane_height, "yaw_deg": scene.yaw_deg,
            "visible_frame_fraction": coverage,
            "model": "Single photo on a plane; not recovered 3D vehicle geometry.",
            "distance_definition": "Plane center Z in original left-camera coordinates, metres.",
        }
        return frames, metadata

    def rectified_pair(self, frames):
        views = []
        for side, suffix, index in (("left", "L", 1), ("right", "R", 2)):
            map_x, map_y = cv2.initUndistortRectifyMap(
                self.calib[f"mtx{suffix}"], self.calib[f"dist{suffix}"],
                self.calib[f"R{index}"], self.calib[f"P{index}"],
                (WIDTH, HEIGHT), cv2.CV_32FC1,
            )
            views.append(cv2.remap(frames[side], map_x, map_y, cv2.INTER_LINEAR))
        result = np.hstack(views)
        for y in range(40, HEIGHT, 80):
            cv2.line(result, (0, y), (2 * WIDTH - 1, y), (0, 255, 0), 1)
        return result


# A separate lightweight process provides real, paced stdout pipes to the exact
# FFmpegCamera implementation in main.py. It imports only the standard library,
# so camera startup does not have to wait for numpy/OpenCV/Torch imports.
STREAM_WORKER = r'''
import os
from pathlib import Path
import sys
import time
frame = Path(sys.argv[1]).read_bytes()
expected = int(sys.argv[2]) * int(sys.argv[3]) * 3
fps = float(sys.argv[4])
if len(frame) != expected or fps <= 0:
    print("Invalid simulator frame or FPS", file=sys.stderr)
    sys.exit(2)
period = 1.0 / fps
deadline = time.perf_counter()
try:
    while True:
        sys.stdout.buffer.write(frame)
        sys.stdout.buffer.flush()
        deadline += period
        now = time.perf_counter()
        if deadline < now - period:
            deadline = now
        time.sleep(max(0.0, deadline - now))
except (BrokenPipeError, OSError):
    os._exit(0)
'''


def _is_dshow_command(args):
    if not isinstance(args, (list, tuple)) or not args:
        return False
    name = str(args[0]).replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name in ("ffmpeg", "ffmpeg.exe") and any(
        str(args[i]) == "-f" and str(args[i + 1]) == "dshow"
        for i in range(len(args) - 1)
    )


def _argument(args, flag, default=None):
    try:
        return str(args[args.index(flag) + 1])
    except (ValueError, IndexError):
        return default


@contextmanager
def virtual_cameras(frames, include_real_cameras=False):
    """Intercept only FFmpeg DirectShow enumeration and our two video endpoints."""
    original_run = subprocess.run
    original_popen = subprocess.Popen
    processes = []
    with tempfile.TemporaryDirectory(prefix="stereo_c270_") as folder:
        frame_paths = {}
        for side in CAMERA_NAMES:
            frame = frames[side]
            if frame.shape != (HEIGHT, WIDTH, 3) or frame.dtype != np.uint8:
                raise ValueError("Frame simulator harus uint8 BGR 1280x720.")
            path = Path(folder) / f"{side}.bgr"
            path.write_bytes(frame.tobytes())
            frame_paths[ENDPOINTS[side]] = path

        class SimulatorPopen(original_popen):
            def __init__(self, args, *positional, **kwargs):
                simulated = False
                if _is_dshow_command(args):
                    input_arg = _argument(args, "-i", "")
                    endpoint = input_arg.removeprefix("video=")
                    if endpoint in frame_paths:
                        if _argument(args, "-video_size", "1280x720") != "1280x720":
                            raise ValueError("Simulator menggunakan resolusi kalibrasi 1280x720.")
                        rate = _argument(args, "-framerate", str(FPS))
                        if not math.isfinite(float(rate)) or not 0 < float(rate) <= 120:
                            raise ValueError("FPS simulator harus di antara 0 dan 120.")
                        args = [sys.executable, "-u", "-c", STREAM_WORKER,
                                str(frame_paths[endpoint]), str(WIDTH), str(HEIGHT), rate]
                        simulated = True
                        if os.name == "nt":
                            kwargs["creationflags"] = (
                                kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
                            )
                super().__init__(args, *positional, **kwargs)
                if simulated:
                    processes.append(self)

        def simulator_run(*positional, **kwargs):
            args = positional[0] if positional else kwargs.get("args")
            if not _is_dshow_command(args) or _argument(args, "-list_devices") != "true":
                return original_run(*positional, **kwargs)
            listing = ""
            for side, name in CAMERA_NAMES.items():
                listing += (f'[dshow @ simulator] "{name}" (video)\n'
                            f'[dshow @ simulator]   Alternative name "{ENDPOINTS[side]}"\n')
            if include_real_cameras:
                try:
                    real = original_run(
                        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        text=True, encoding="utf-8", errors="replace", timeout=10,
                    )
                    listing += real.stderr or ""
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    pass
            is_text = bool(kwargs.get("text") or kwargs.get("universal_newlines")
                           or kwargs.get("encoding") or kwargs.get("errors"))
            result = subprocess.CompletedProcess(
                args, 1, "" if is_text else b"",
                listing if is_text else listing.encode("utf-8"),
            )
            if kwargs.get("check"):
                result.check_returncode()
            return result

        subprocess.Popen = SimulatorPopen
        subprocess.run = simulator_run
        try:
            yield
        finally:
            subprocess.run = original_run
            subprocess.Popen = original_popen
            # Also reap streams left open by an exception/failed GUI startup.
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()


def export_pair(folder, frames, metadata, renderer):
    folder = Path(folder).resolve()
    names = ("left.png", "right.png", "rectified_pair.jpg", "simulation.json")
    if any((folder / name).exists() for name in names):
        raise ValueError(f"File ekspor sudah ada di {folder}; pilih folder baru.")
    folder.mkdir(parents=True, exist_ok=True)
    pictures = {"left.png": frames["left"], "right.png": frames["right"],
                "rectified_pair.jpg": renderer.rectified_pair(frames)}
    for name, frame in pictures.items():
        ok, encoded = cv2.imencode(Path(name).suffix, frame)
        if not ok:
            raise ValueError(f"Gagal mengenkode {name}.")
        encoded.tofile(folder / name)
    (folder / "simulation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )


def show_preview(frames, metadata):
    title = "Stereo C270 Simulator - Left | Right (Esc untuk tutup)"
    view = np.hstack([frames["left"], frames["right"]])
    for index, side in enumerate(CAMERA_NAMES):
        cv2.putText(view, CAMERA_NAMES[side], (20 + index * WIDTH, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(view, f"Z bidang: {metadata['plane_center_in_left_m'][2]:.2f} m | "
                f"Baseline: {metadata['baseline_m'] * 100:.2f} cm",
                (20, HEIGHT - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 360)
    try:
        while True:
            cv2.imshow(title, view)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()


def launch_application(frames, include_real_cameras=False, application="main.py"):
    target = ROOT / application
    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    try:
        os.chdir(ROOT)
        sys.argv = [str(target)]
        with virtual_cameras(frames, include_real_cameras):
            namespace = runpy.run_path(str(target), run_name="__main__")
            # main.py keeps the NPZ open at module scope. Release that input
            # handle when its GUI exits, without changing the original source.
            calibration_data = namespace.get("data")
            if hasattr(calibration_data, "close"):
                calibration_data.close()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


def setup_window(args):
    """Return a Scene after the user selects a photo and clicks Run."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Simulator Stereo Logitech C270")
    root.resizable(False, False)
    panel = ttk.Frame(root, padding=18)
    panel.grid(sticky="nsew")
    candidates = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    initial_image = str(args.image or (candidates[0] if candidates else ""))
    image_var = tk.StringVar(value=initial_image)
    fields = {}
    result = []
    renderer = None

    def browse():
        path = filedialog.askopenfilename(
            parent=root, initialdir=IMAGE_DIR, title="Pilih gambar truk",
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Semua file", "*.*")],
        )
        if path:
            image_var.set(path)

    ttk.Label(panel, text="Foto sumber").grid(row=0, column=0, sticky="w")
    ttk.Entry(panel, textvariable=image_var, width=58).grid(row=1, column=0, columnspan=2, pady=(4, 12))
    ttk.Button(panel, text="Pilih gambar...", command=browse).grid(row=1, column=2, padx=(8, 0), pady=(4, 12))
    for row, (key, label, value) in enumerate((
        ("distance_m", "Jarak pusat bidang Z (meter)", args.distance),
        ("plane_width_m", "Lebar SELURUH foto (meter)", args.plane_width),
        ("yaw_deg", "Sudut bidang / yaw (derajat)", args.yaw),
        ("center_x_m", "Geser pusat X (meter; + kanan)", args.center_x),
        ("center_y_m", "Geser pusat Y (meter; + bawah)", args.center_y),
    ), start=2):
        ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=4)
        fields[key] = tk.StringVar(value=str(value))
        ttk.Entry(panel, textvariable=fields[key], width=16).grid(row=row, column=1, sticky="w", padx=12)
    ttk.Label(panel, text="1280 x 720 | 30 FPS | geometri dari kalibrasi asli\n"
              "Foto dimodelkan sebagai bidang datar; lebar foto bukan ukuran pasti truk.\n"
              "Kamera tersedia di aplikasi yang dijalankan melalui simulator ini.",
              foreground="#555555", justify="left").grid(
                  row=7, column=0, columnspan=3, sticky="w", pady=14)

    def read_scene():
        scene = Scene(Path(image_var.get()).expanduser().resolve(),
                      **{key: float(var.get()) for key, var in fields.items()})
        scene.validate()
        return scene

    def render_scene():
        nonlocal renderer
        scene = read_scene()
        if renderer is None:
            renderer = StereoRenderer()
        return renderer.render(scene)

    def preview():
        try:
            frames, metadata = render_scene()
            show_preview(frames, metadata)
        except Exception as exc:
            messagebox.showerror("Simulator", str(exc), parent=root)

    def start():
        try:
            scene = read_scene()
            result.append(scene)
            root.destroy()
        except (ValueError, OSError) as exc:
            messagebox.showerror("Parameter belum valid", str(exc), parent=root)

    ttk.Button(panel, text="Pratinjau dua kamera", command=preview).grid(row=8, column=0, sticky="w")
    ttk.Button(panel, text="Jalankan main.py", command=start).grid(row=8, column=1, columnspan=2, sticky="e")
    root.mainloop()
    return result[0] if result else None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Dua kamera C270 simulasi dari foto dan kalibrasi asli; file final tidak diedit.",
        epilog="Tanpa --image, jendela pemilih gambar dibuka. Kamera simulator hanya tersedia "
               "saat aplikasi dijalankan melalui launcher ini, bukan perangkat Windows global.",
    )
    parser.add_argument("--image", type=Path, help="Foto sumber (JPG/PNG/etc.)")
    parser.add_argument("--distance", type=float, default=8.0, help="Z pusat bidang dalam meter (default: 8)")
    parser.add_argument("--plane-width", type=float, default=6.0, help="Lebar fisik seluruh foto, meter (default: 6)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Sudut bidang dalam derajat (default: 0)")
    parser.add_argument("--center-x", type=float, default=0.0, help="Posisi pusat X, meter (+ kanan)")
    parser.add_argument("--center-y", type=float, default=0.0, help="Posisi pusat Y, meter (+ bawah)")
    parser.add_argument("--preview", action="store_true", help="Hanya tampilkan pasangan frame, tanpa main.py")
    parser.add_argument("--export", type=Path, metavar="FOLDER", help="Ekspor pasangan frame dan metadata ke folder baru")
    parser.add_argument("--no-launch", action="store_true", help="Tidak menjalankan main.py (gunakan bersama --export)")
    parser.add_argument("--include-real-cameras", action="store_true", help="Sertakan kamera Windows dari FFmpeg asli")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    IMAGE_DIR.mkdir(exist_ok=True)
    if args.no_launch and not (args.export or args.preview):
        raise ValueError("--no-launch memerlukan --export atau --preview.")
    if args.image is None:
        scene = setup_window(args)
        if scene is None:
            return 0
    else:
        scene = Scene(args.image.expanduser().resolve(), args.distance, args.plane_width,
                      args.yaw, args.center_x, args.center_y)
    scene.validate()
    renderer = StereoRenderer()
    frames, metadata = renderer.render(scene)
    print(f"Baseline kalibrasi: {renderer.baseline_m * 100:.3f} cm | "
          f"Z pusat bidang: {scene.distance_m:.2f} m | {WIDTH}x{HEIGHT} @ {FPS} FPS")
    print("Model foto: bidang datar. Ukuran/jarak berasal dari parameter simulasi.")
    if args.export:
        export_pair(args.export, frames, metadata, renderer)
        print(f"Pasangan frame diekspor ke: {args.export.resolve()}")
    if args.preview:
        show_preview(frames, metadata)
    elif not args.no_launch:
        print("Membuka main.py dengan kamera Logi C270 Simulator Left/Right...")
        launch_application(frames, args.include_real_cameras)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
    except (ValueError, OSError) as exc:
        print(f"Simulator: {exc}", file=sys.stderr)
        raise SystemExit(1)
