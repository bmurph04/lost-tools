import numpy as np
import cv2

from src.utils import unity_pose_to_cv

class StereoRectifier:
    """
    Rectifies Quest 3 passthrough stereo pairs so that corresponding points
    share an image row, which is what FoundationStereo's horizontal cost
    volume requires.

    The two cameras are rigidly mounted, so the maps are built once.
    """

    def __init__(self, left_focal_length, left_optical_center, right_focal_length, right_optical_center, rel_camera_rot, rel_camera_trans, image_width, image_height):
        
        size = (image_width, image_height)

        R_L, t_L = unity_pose_to_cv(left['pos'], left['rot'])
        R_R, t_R = unity_pose_to_cv(right['pos'], right['rot'])

        # Transform taking a point from the left camera frame into the right
        R = R_R.T @ R_L
        T = R_R.T @ (t_L - t_R)

        K_L = np.array([[left['fx'], 0, left['cx']],
                        [0, left['fy'], left['cy']],
                        [0, 0, 1]], dtype=np.float64)
        K_R = np.array([[right['fx'], 0, right['cx']],
                        [0, right['fy'], right['cy']],
                        [0, 0, 1]], dtype=np.float64)
        D = np.zeros(5)   # passthrough images are delivered rectilinear

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_L, D, K_R, D, size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        self._map_left = cv2.initUndistortRectifyMap(K_L, D, R1, P1, size, cv2.CV_16SC2)
        self._map_right = cv2.initUndistortRectifyMap(K_R, D, R2, P2, size, cv2.CV_16SC2)

        self._R1 = R1
        self.Q = Q
        self.focal_length = (float(P1[0, 0]), float(P1[1, 1]))
        self.optical_center = (float(P1[0, 2]), float(P1[1, 2]))
        self.baseline = float(-P2[0, 3] / P2[0, 0])

    def rectify_pair(self, left_chw, right_chw):
        """(3, H, W) uint8 in -> rectified (3, H, W) uint8 out."""
        left_hwc = np.ascontiguousarray(left_chw.transpose(1, 2, 0))
        right_hwc = np.ascontiguousarray(right_chw.transpose(1, 2, 0))
        left_rect = cv2.remap(left_hwc, *self._map_left, interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_hwc, *self._map_right, interpolation=cv2.INTER_LINEAR)
        return left_rect.transpose(2, 0, 1), right_rect.transpose(2, 0, 1)

    def rectified_left_pose(self, metadata):
        """
        World pose of the *rectified* left camera in OpenCV convention.
        Rectification rotates the left camera by R1 about its own centre,
        so the translation is unchanged and the rotation gains R1 transposed.
        """
        R_L, t_L = unity_pose_to_cv(metadata['leftCamera']['pos'],
                                    metadata['leftCamera']['rot'])
        return R_L @ self._R1.T, t_L
