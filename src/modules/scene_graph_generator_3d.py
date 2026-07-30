from src.models.geometric_3dsg_build import Geometric3DSGBuilder
from src.models.gaussian_3d_lift import Gaussian3DLift

class SceneGraphGenerator3D:

    def __init__(self, method):
        self.method = method

    def generate_graph(self, means, covs, instances, object_labels, output=None, point_lifting_method=None, focal_length=None, camera_rot=None, camera_trans=None, input_img=None):

        if isinstance(self.method, Geometric3DSGBuilder):
            scene_graph = self.method.build_3d_scene_graph(means, covs, instances)

        if output:
            assert point_lifting_method, \
                "Need to know the 3D point representation in order to properly visualize 3D scene graph"

            if isinstance(point_lifting_method, Gaussian3DLift):
                # point_lifting_method.visualize_3d_gaussians_on_image(
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

                point_lifting_method.visualize_3d_gaussians_in_3d(
                    means_3d=means,
                    covs_3d=covs,
                    instances=instances,
                    labels=object_labels,
                    output_path=output,
                    triplets=scene_graph,
                    std_scale=1.0,
                    show_camera_origin=True
                )

        return scene_graph