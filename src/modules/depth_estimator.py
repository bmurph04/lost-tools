import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import depth_pro
from depth_pro.depth_pro import DepthPro
from unidepth.models import UniDepthV2

class DepthEstimator:

    def __init__(self, device, model):
        self.device = device
        self.model = model


    def process_frame(self, frame_str, transform=None, output=None):
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

            return depth, focallength_px

        def unidepth_process_frame(frame_str):
            """
            Process a frame using UniDepth estimator model.

            Returns depth and focal length.
            """
            image = torch.from_numpy(np.array(Image.open(frame_str))).permute(2, 0, 1)
            predictions = self.model.infer(image)

            depth_batched = predictions['depth']
            depth = depth_batched[0, 0, :, :]

            intrinsics = predictions['intrinsics']
            focal_length = intrinsics[0, 0, 0].item(), intrinsics[0, 1, 1].item()

            return depth, focal_length


        if isinstance(self.model, DepthPro):
            depth, focal_length = depthpro_process_frame(frame_str, transform)

        if isinstance(self.model, UniDepthV2):
            depth, focal_length = unidepth_process_frame(frame_str)

        # DepthPro output saving logic
        if output:
            inverse_depth = 1 / depth
            # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
            max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth_vizu = max(1 / 250, inverse_depth.min())
            inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                max_invdepth_vizu - min_invdepth_vizu
            )

            np.savez_compressed(output, depth=depth)

            # Save as color-mapped "turbo" jpg image.
            cmap = plt.get_cmap("turbo")
            color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
                np.uint8
            )
            color_map_output_file = str(output) + ".jpg"
            Image.fromarray(color_depth).save(
                color_map_output_file, format="JPEG", quality=90
            )

        return depth, focal_length
