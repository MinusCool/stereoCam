import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (9, 6)  # jumlah inner corners per baris dan kolom
SQUARE_SIZE = 0.04
IMAGE_DIR = "images"
SAVE_RESULT = True

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpointsL = []
imgpointsR = []

images_left = sorted(glob.glob(os.path.join(IMAGE_DIR, 'chessboard-L*.png')))
images_right = sorted(glob.glob(os.path.join(IMAGE_DIR, 'chessboard-R*.png')))

assert len(images_left) == len(images_right), "Jumlah gambar kiri dan kanan tidak sama"

for fnameL, fnameR in zip(images_left, images_right):
    imgL = cv2.imread(fnameL)
    imgR = cv2.imread(fnameR)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)

    if retL and retR:
        objpoints.append(objp)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

retL, mtxL, distL, _, _ = cv2.calibrateCamera(objpoints, imgpointsL, grayL.shape[::-1], None, None)
retR, mtxR, distR, _, _ = cv2.calibrateCamera(objpoints, imgpointsR, grayR.shape[::-1], None, None)

flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_RATIONAL_MODEL
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 500, 1e-6)

ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    mtxL, distL, mtxR, distR,
    grayL.shape[::-1], criteria=criteria_stereo, flags=flags)

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, grayL.shape[::-1], R, T, alpha=0.2)

if SAVE_RESULT:
    np.savez("stereo_calib_result.npz",
             mtxL=mtxL, distL=distL,
             mtxR=mtxR, distR=distR,
             R=R, T=T, E=E, F=F, Q=Q,
             R1=R1, R2=R2, P1=P1, P2=P2)

    print("\n📁 Kalibrasi disimpan di file 'stereo_calib_result.npz'")

print("\n=== HASIL KALIBRASI STEREO ===")
print("R (Rotasi):\n", R)
print("T (Translasi dalam meter):\n", T)
print("Matriks Kamera Kiri:\n", mtxL)
print("Distorsi Kiri:\n", distL.ravel())
print("Matriks Kamera Kanan:\n", mtxR)
print("Distorsi Kanan:\n", distR.ravel())
print("Q Matrix:\n", Q)

errorL = 0
errorR = 0
for i in range(len(objpoints)):
    imgpointsL_reproj, _ = cv2.projectPoints(objpoints[i], R, T, mtxL, distL)
    imgpointsR_reproj, _ = cv2.projectPoints(objpoints[i], R, T, mtxR, distR)
    errorL += cv2.norm(imgpointsL[i], imgpointsL_reproj, cv2.NORM_L2) / len(imgpointsL[i])
    errorR += cv2.norm(imgpointsR[i], imgpointsR_reproj, cv2.NORM_L2) / len(imgpointsR[i])

print(f"Error kiri: {errorL / len(objpoints)}")
print(f"Error kanan: {errorR / len(objpoints)}")
