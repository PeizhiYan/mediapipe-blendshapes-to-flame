import os
import numpy as np

class MP_2_FLAME():
    """
    Convert Mediapipe 52 blendshape scores to FLAME's coefficients 
    """
    def __init__(self, mappings_path):
        self.bs2exp = np.load(os.path.join(mappings_path, 'bs2exp.npy'))
        self.bs2pose = np.load(os.path.join(mappings_path, 'bs2pose.npy'))
        self.bs2eye = np.load(os.path.join(mappings_path, 'bs2eye.npy'))

    def convert(self, blendshape_scores : np.array):
        # blendshape_scores: [N, 52]

        # Calculate expression, pose, and eye_pose using the mappings
        exp = blendshape_scores @ self.bs2exp 
        pose = blendshape_scores @ self.bs2pose 
        pose[0, :3] = 0  # we do not support head rotation yet
        eye_pose = blendshape_scores @ self.bs2eye

        return exp, pose, eye_pose
    

