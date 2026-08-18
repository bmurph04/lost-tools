from src.models.build_geometric_3dsg import Geometric3DSGBuilder
from src.models.lift_gaussian_3d import Gaussian3DLift
import numpy as np

class SceneGraphGenerator3D:

    def __init__(self, name, sgg_method, point_lifting_method_name, extent_std=2.0):
        self.name = name
        self.sgg_method = sgg_method
        self.point_lifting_method_name = point_lifting_method_name

        # Set the number of std used as object extent
        self.extent_std_scale = extent_std
        
    def generate_triplets(self, points_representation, pred_name_to_id):
        """
        Generate triplets
        """
        if self.name == 'geometric_3dsg_builder':
            
            if self.point_lifting_method_name == 'gaussian_3d_lift':
                means, covs, pcds = points_representation
                # Get the stds for each axis (X,Y,Z)
                stds = np.sqrt(np.diagonal(covs, axis1=1, axis2=2))
                # Get the extents
                extents = self.extent_std_scale * np.maximum(stds, 0.0) # shape: ?
            
            scene_graph = self.sgg_method.build_3d_scene_graph(means, extents, pred_name_to_id)

        return scene_graph

    def visualize(self, frame, points_representation, scene_graph, object_labels, pred_id_to_name, output, focal_length=None, optical_center=None, camera_rot=None, camera_pos=None, camera_view_mode="isometric", show_camera=False, auto_zoom=False, zoom_padding=0.15, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), z_range=(0.0, 2.0), std_scale=2.0):
        """
        Visualize
        """
        if self.point_lifting_method_name == 'gaussian_3d_lift':
            means, covs, pcds = points_representation
            # self.point_lifting_method.visualize_3d_gaussians_on_image(
            #     image_input=frame.copy(),
            #     means_3d=means,
            #     covs_3d=covs,
            #     labels=object_labels,
            #     focal_length=focal_length,
            #     output_path=output,
            #     triplets=scene_graph,
            #     pred_id_to_name=pred_id_to_name
            #     camera_rot=camera_rot,
            #     camera_pos=camera_pos,
            #     std_scale=2.0
            # )

            Gaussian3DLift.visualize_3d_gaussians_in_3d(
                means_3d=means,
                covs_3d=covs,
                labels=object_labels,
                output_path=output,
                triplets=scene_graph,
                pred_id_to_name=pred_id_to_name,
                camera_rot=camera_rot,
                camera_pos=camera_pos,
                camera_view_mode=camera_view_mode,
                auto_zoom=auto_zoom,
                zoom_padding=zoom_padding,
                show_camera=show_camera,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                std_scale=std_scale
            )