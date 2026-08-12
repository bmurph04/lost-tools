import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import depth_pro
from depth_pro.depth_pro import DepthPro
from external.unidepth.unidepth.models.unidepthv2 import UniDepthV2
from external.fast_foundationstereo.core.foundation_stereo import FastFoundationStereo

class DepthEstimator:

    def __init__(self, device, model):
        self.device = device
        self.model = model


    def process_frame(self, frame, transform=None, output=None):
        """
        Given a frame, process a frame using the initialized depth estimator model.

        Returns the depth and focal length.
        """

        def depthpro_process_frame(frame_str, transform):
            """
            Process a frame using DepthPro depth estimator model.

            Returns depth and focal length.
            """
            image, _, f_px = depth_pro.load_rgb(frame_str)
            image = transform(image).to(self.device)

            prediction = self.model.infer(image, f_px=f_px)
            depth = prediction["depth"] # depth in [m]
            focallength_px = prediction["focallength_px"] # focal length in [px]
            # FIXME: Return all intrinsics [fx, fy, cx, cy] properly
            return depth, focallength_px, None

        def unidepth_process_frame(frame):
            """
            Process a frame using UniDepth estimator model.

            Returns depth and focal length.
            """
            predictions = self.model.infer(frame)

            depth_batched = predictions['depth']
            depth = depth_batched[0, 0, :, :]

            intrinsics = predictions['intrinsics']
            camera_coords = intrinsics[0, 0, 2].item(), intrinsics[0, 1, 2].item()
            focal_length = intrinsics[0, 0, 0].item(), intrinsics[0, 1, 1].item()

            return depth, focal_length, camera_coords

        def ffstereo_process_frame(left_frame, right_frame):
            """
            Process a frame using Fast Foundation Stereo depth estimator model.

            Returns depth.
            """

            left_frame = left_frame.unsqueeze(0)
            right_frame = right_frame.unsqueeze(0)

            with torch.inference_mode():
                disp = self.model.forward(left_frame, right_frame, test_mode=True)

            disp = None # TODO: unpad

        if isinstance(frame, tuple):
            left_frame, right_frame = frame
            if isinstance(self.model, FastFoundationStereo):
                depth = ffstereo_process_frame(left_frame, right_frame)

            return depth

        else:
            if isinstance(self.model, DepthPro):
                assert isinstance(frame, str), "For DepthPro, frame must be passed in as str for process_frame"
                depth, focal_length, camera_coords = depthpro_process_frame(frame, transform)

            if isinstance(self.model, UniDepthV2):
                depth, focal_length, camera_coords = unidepth_process_frame(frame)

            return depth, focal_length, camera_coords
        
    def visualize(self, depth, output):
        inverse_depth = 1 / depth

        # Ensure inverse_depth is a CPU NumPy array for clipping & math
        if torch.is_tensor(inverse_depth):
            inverse_depth = inverse_depth.detach().cpu().numpy()

        # Visualize inverse depth clipped to [0.1m; 250m] range
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())

        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu + 1e-8
        )
        inverse_depth_normalized = np.clip(inverse_depth_normalized, 0.0, 1.0)

        # Apply Matplotlib "turbo" colormap
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
            np.uint8
        )

        # Convert RGB to BGR for OpenCV saving
        color_mapped_bgr = cv2.cvtColor(color_depth, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output, color_mapped_bgr)
