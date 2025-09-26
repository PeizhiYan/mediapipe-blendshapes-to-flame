#
# Peizhi Yan
# 2025. Copyright
#

import os
import numpy as np
import cv2

class MP_2_FLAME():
    """
    Convert Mediapipe 52 blendshape scores to FLAME's coefficients 
    """
    def __init__(self, mappings_path):
        self.bs2exp = np.load(os.path.join(mappings_path, 'bs2exp.npy'))
        self.bs2jaw = np.load(os.path.join(mappings_path, 'bs2jaw.npy'))
        self.bs2eye = np.load(os.path.join(mappings_path, 'bs2eye.npy'))

    def convert(self, blendshape_scores : np.array):
        # blendshape_scores: [N, 52]

        # Calculate expression, pose, and eye_pose using the mappings
        exp = blendshape_scores @ self.bs2exp 
        jaw = blendshape_scores @ self.bs2jaw 
        eye_pose = blendshape_scores @ self.bs2eye

        return exp, jaw, eye_pose
    

def compute_head_pose_from_mp_landmarks_3d(face_landmarks : np.array, img_h : int, img_w : int):
    """
    Compute head pose based on facial landmarks detected by MediaPipe FaceMesh.

    This code is based on: https://medium.com/@jaykumaran2217/real-time-head-pose-estimation-facemesh-with-mediapipe-and-opencv-a-comprehensive-guide-b63a2f40b7c6

    Authors: Peizhi Yan, Haoyu Wang

    Parameters:
        face_landmarks : The normalized 3D facial landmarks detected by MediaPipe. Numpy array with shape [478, 3]
        img_h, img_w : width and height of original image

    Returns:
        rotation_vec (np.ndarray): Rotation vector indicating the orientation of the head. [3, 1]
        translation_vec (np.ndarray): Translation vector indicating the position of the head. [3, 1]
    """

    # Prepare 2D and 3D landmark arrays
    face_2d = []
    face_3d = []

    # Select specific landmarks
    for idx, lm in enumerate(face_landmarks):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            x, y = int(lm[0]), int(lm[1])
            face_2d.append([x, y])
            face_3d.append([x, y, lm[2]])

    # Get 2D and 3D coordinates
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Set camera parameters
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    distortion_matrix = np.zeros((4, 1), dtype=np.float64)

    # Pose estimation
    success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

    return rotation_vec, translation_vec



