from src.models.geometric_3dsg_build import Geometric3DSGBuilder
from src.models.gaussian_3d_lift import Gaussian3DLift

class SceneGraphGenerator3D:

    def __init__(self, method):
        self.method = method

    def generate_graph(self, points_representation, instances):

        if isinstance(self.method, Geometric3DSGBuilder):
            means, covs = points_representation
            scene_graph = self.method.build_3d_scene_graph(means, covs, instances)

        return scene_graph

    def visualize(self, frame, focal_length, scene_graph, points_representation, object_instances, object_labels, output, camera_rot=None, camera_trans=None):

        if isinstance(self.method, Gaussian3DLift):
            means, covs = points_representation
            # self.method.visualize_3d_gaussians_on_image(
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

            self.method.visualize_3d_gaussians_in_3d(
                means_3d=means,
                covs_3d=covs,
                instances=object_instances,
                labels=object_labels,
                output_path=output,
                triplets=scene_graph,
                std_scale=1.0,
                show_camera_origin=True
            )