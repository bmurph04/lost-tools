from src.models.build_geometric_3dsg import Geometric3DSGBuilder
from src.models.lift_gaussian_3d import Gaussian3DLift
import numpy as np

class SceneGraphGenerator3D:

    def __init__(self, sgg_method, point_lifting_method, extent_std=2.0):
        self.sgg_method = sgg_method
        self.point_lifting_method = point_lifting_method
        
        self.point_lifting_method = self.point_lifting_method
        
        # Set the number of std used as object extent
        self.extent_std_scale = extent_std

    def generate_triplets(self, points_representation, instances):

        if isinstance(self.sgg_method, Geometric3DSGBuilder):
            
            if isinstance(self.point_lifting_method, Gaussian3DLift):
                means, covs, pcds = points_representation
                # Get the stds for each axis (X,Y,Z)
                stds = np.diagonal(covs, axis1=1, axis2=2)
                # Get the extents
                extents = self.extent_std_scale * np.maximum(stds, 0.0) # shape: ?
            
            scene_graph = self.sgg_method.build_3d_scene_graph(means, extents, instances)

        return scene_graph

    def visualize(self, frame, focal_length, scene_graph, points_representation, object_instances, object_labels, output, camera_rot=None, camera_trans=None):

        if isinstance(self.point_lifting_method, Gaussian3DLift):
            means, covs, pcds = points_representation
            # self.point_lifting_method.visualize_3d_gaussians_on_image(
            #     image_input=input_img,
            #     means_3d=means,
            #     covs_3d=covs,
            #     instances=instances,
            #     labels=object_labels,
            #     focal_length=focal_length,
            #     output_path=output,
            #     triplets=scene_graph,
            #     camera_rot=camera_rot,
            #     camera_trans=camera_trans,
            #     std_scale=2.0
            # )

            self.point_lifting_method.visualize_3d_gaussians_in_3d(
                means_3d=means,
                covs_3d=covs,
                instances=object_instances,
                labels=object_labels,
                output_path=output,
                triplets=scene_graph,
                std_scale=1.0,
                show_camera_origin=True
            )