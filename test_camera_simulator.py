"""Geometry and real pipe integration tests, without opening/mutating main.py."""

import ast
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import List, Dict, Optional
import unittest

import cv2
import numpy as np

import camera_simulator as sim


def original_functions(*names):
    """Load the actual requested definitions, excluding GUI/module side effects."""
    tree = ast.parse((sim.ROOT / "main.py").read_text(encoding="utf-8-sig"))
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if len(selected) != len(names):
        raise AssertionError("Requested main.py definitions were not found")
    namespace = {
        "cv2": cv2, "np": np, "subprocess": subprocess, "time": time,
        "threading": threading, "deque": deque, "Optional": Optional,
        "List": List, "Dict": Dict, "re": __import__("re"),
        "SUBPIX_TEMPLATE_HALF": 7, "SUBPIX_SEARCH_HALF": 40,
        "MIN_VALID_DISPARITY_PX": 1.5, "MAX_VALID_DISPARITY_PX": 800.0,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), "main.py (selected definitions)", "exec"), namespace)
    return namespace


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="stereo_simulator_test_")
        cls.folder = Path(cls.temp.name)
        cls.photo_path = cls.folder / "test_texture.png"
        rng = np.random.default_rng(12345)
        cls.photo = rng.integers(0, 256, (320, 640, 3), dtype=np.uint8)
        cls.photo = cv2.GaussianBlur(cls.photo, (3, 3), 0.7)
        cls.markers = [(150, 110, (20, 20, 240)), (490, 220, (240, 220, 20))]
        for x, y, color in cls.markers:
            cv2.circle(cls.photo, (x, y), 7, color, -1)
        cv2.imencode(".png", cls.photo)[1].tofile(cls.photo_path)
        cls.renderer = sim.StereoRenderer()
        cls.scene = sim.Scene(cls.photo_path, distance_m=8.0, plane_width_m=4.0)
        cls.frames, cls.metadata = cls.renderer.render(cls.scene)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def project(self, points):
        calib = self.renderer.calib
        left = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                 calib["mtxL"], calib["distL"])[0]
        right = cv2.projectPoints(points, cv2.Rodrigues(calib["R"])[0], calib["T"],
                                  calib["mtxR"], calib["distR"])[0]
        return left, right

    def rectify(self, points, side):
        c = self.renderer.calib
        suffix, i = ("L", 1) if side == "left" else ("R", 2)
        return cv2.undistortPoints(points, c[f"mtx{suffix}"], c[f"dist{suffix}"],
                                  R=c[f"R{i}"], P=c[f"P{i}"]).reshape(-1, 2)

    def test_calibration_and_raw_frame_contract(self):
        self.assertAlmostEqual(self.renderer.baseline_m, 0.44817316857779477, places=10)
        for side in sim.CAMERA_NAMES:
            self.assertEqual(self.frames[side].shape, (720, 1280, 3))
            self.assertEqual(self.frames[side].dtype, np.uint8)
        self.assertFalse(np.array_equal(self.frames["left"], self.frames["right"]))

    def test_camera_center_satisfies_extrinsics(self):
        c = self.renderer.calib
        np.testing.assert_allclose(c["R"] @ self.renderer.centers["right"] + c["T"], 0, atol=1e-12)

    def test_rendered_landmarks_follow_original_camera_projection(self):
        # Check inverse rendering against forward projection, for both R/T and
        # lens models, including an oblique plane and unequal wheel depths.
        for yaw in (0, 22):
            scene = sim.Scene(self.photo_path, 8.0, 4.0, yaw)
            frames, _ = self.renderer.render(scene)
            center, u, v, _, height = scene.plane(self.photo.shape)
            points = np.array([
                center + (x / 639 - 0.5) * 4.0 * u + (y / 319 - 0.5) * height * v
                for x, y, _ in self.markers
            ])
            projections = self.project(points)
            for side, pixels in zip(sim.CAMERA_NAMES, projections):
                for pixel, (_, _, color) in zip(pixels.reshape(-1, 2), self.markers):
                    x, y = np.rint(pixel).astype(int)
                    np.testing.assert_allclose(frames[side][y, x], color, atol=2)

    def test_epipolar_alignment_and_depth_against_3d_truth(self):
        points = np.array([[x, y, z] for z in (5.0, 8.0, 15.0)
                           for y in (-0.5, 0.5) for x in (-1.0, 0.0, 1.0)])
        left, right = [self.rectify(p, s) for p, s in zip(self.project(points), sim.CAMERA_NAMES)]
        np.testing.assert_allclose(left[:, 1], right[:, 1], atol=1e-4)
        c = self.renderer.calib
        z_from_disparity = -c["P2"][0, 3] / (left[:, 0] - right[:, 0])
        expected_z = (points @ c["R1"].T)[:, 2]
        np.testing.assert_allclose(z_from_disparity, expected_z, atol=1e-4)

    def test_actual_main_subpixel_matching_recovers_simulated_distance(self):
        c = self.renderer.calib
        gray = {}
        for side, suffix, index in (("left", "L", 1), ("right", "R", 2)):
            maps = cv2.initUndistortRectifyMap(c[f"mtx{suffix}"], c[f"dist{suffix}"],
                                              c[f"R{index}"], c[f"P{index}"],
                                              (sim.WIDTH, sim.HEIGHT), cv2.CV_32FC1)
            gray[side] = cv2.cvtColor(cv2.remap(self.frames[side], *maps, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
        # A textured point avoids the uniform center of the colored landmarks.
        points = np.array([[0.3, -0.3, 8.0]])
        left, right = [self.rectify(p, s)[0] for p, s in zip(self.project(points), sim.CAMERA_NAMES)]
        fn = original_functions("refine_disparity_subpixel")["refine_disparity_subpixel"]
        disparity = fn(gray["left"], gray["right"], left, right)
        recovered_z = -c["P2"][0, 3] / disparity
        truth = (c["R1"] @ points[0])[2]
        self.assertTrue(np.isfinite(recovered_z))
        self.assertLess(abs(recovered_z - truth), 0.15)

    def test_baseline_disparity_changes_with_distance(self):
        disparities = []
        for distance in (5.0, 10.0):
            points = np.array([[0.0, 0.0, distance]])
            lp, rp = [self.rectify(p, s)[0] for p, s in zip(self.project(points), sim.CAMERA_NAMES)]
            disparities.append(lp[0] - rp[0])
        self.assertAlmostEqual(disparities[0] / disparities[1], 2.0, places=5)

    def test_invalid_scenes_rejected(self):
        for kwargs in ({"distance_m": 0}, {"plane_width_m": -1},
                       {"yaw_deg": 89}, {"distance_m": float("nan")}):
            with self.assertRaises(ValueError):
                self.renderer.render(sim.Scene(self.photo_path, **kwargs))
        with self.assertRaises(ValueError):
            self.renderer.render(sim.Scene(self.photo_path, 0.2, 6, 75))
        with self.assertRaises(ValueError):
            self.renderer.render(sim.Scene(self.folder / "missing.jpg"))

    def test_export_writes_frames_and_refuses_to_overwrite(self):
        folder = self.folder / "export"
        sim.export_pair(folder, self.frames, self.metadata, self.renderer)
        data = json.loads((folder / "simulation.json").read_text())
        self.assertEqual(data["plane_center_in_left_m"][2], 8)
        actual = cv2.imdecode(np.fromfile(folder / "right.png", np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(actual, self.frames["right"])
        self.assertTrue((folder / "rectified_pair.jpg").exists())
        with self.assertRaises(ValueError):
            sim.export_pair(folder, self.frames, self.metadata, self.renderer)

    def test_original_main_lists_and_opens_both_cameras_and_can_restart(self):
        definitions = original_functions("FFmpegCamera", "parse_dshow_video_devices",
                                         "run_ffmpeg_list_dshow_devices", "list_ffmpeg_devices")
        native_run, native_popen = subprocess.run, subprocess.Popen
        cameras = []
        with sim.virtual_cameras(self.frames):
            labels, mapping = definitions["list_ffmpeg_devices"]()
            self.assertEqual(len(labels), 2)
            self.assertIn("Left", labels[0])
            self.assertIn("Right", labels[1])
            try:
                for label, side in zip(labels, sim.CAMERA_NAMES):
                    camera = definitions["FFmpegCamera"](mapping[label], sim.WIDTH, sim.HEIGHT, sim.FPS)
                    cameras.append(camera)
                    self.assertTrue(camera.start(), camera.last_error)
                    ok, frame, timestamp = camera.read()
                    self.assertTrue(ok)
                    self.assertLess(abs(time.time() - timestamp), 1.0)
                    np.testing.assert_array_equal(frame, self.frames[side])
                # Running both streams should continue to deliver fresh frames.
                old = [camera.read()[2] for camera in cameras]
                time.sleep(0.15)
                self.assertTrue(all(camera.read()[2] > previous for camera, previous in zip(cameras, old)))
                cameras[0].stop()
                cameras[0]._thread.join(timeout=1)
                cameras[0]._stderr_thread.join(timeout=1)
                self.assertTrue(cameras[0].start(), cameras[0].last_error)
            finally:
                for camera in cameras:
                    camera.stop()
                    if camera._thread:
                        camera._thread.join(timeout=1)
                    if camera._stderr_thread:
                        camera._stderr_thread.join(timeout=1)
        self.assertIs(subprocess.run, native_run)
        self.assertIs(subprocess.Popen, native_popen)

    def test_bridge_restores_process_functions_after_exception(self):
        native_run, native_popen = subprocess.run, subprocess.Popen
        process = None
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with sim.virtual_cameras(self.frames):
                process = subprocess.Popen(
                    ["ffmpeg", "-f", "dshow", "-i", "video=stereo-sim://left"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                self.assertEqual(len(process.stdout.read(sim.WIDTH * sim.HEIGHT * 3)), sim.WIDTH * sim.HEIGHT * 3)
                raise RuntimeError("intentional")
        self.assertIs(subprocess.run, native_run)
        self.assertIs(subprocess.Popen, native_popen)
        self.assertIsNotNone(process.poll())

    def test_unrelated_subprocess_commands_passthrough(self):
        with sim.virtual_cameras(self.frames):
            result = subprocess.run([sys.executable, "-c", "print('passthrough')"],
                                    text=True, capture_output=True, check=True)
        self.assertEqual(result.stdout.strip(), "passthrough")

if __name__ == "__main__":
    unittest.main(verbosity=2)
