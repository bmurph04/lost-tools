from src.models.gaussian_3d_lift import Gaussian3DLift

class PointLifter:
    def __init__(self, method):
        self.method = method

    def lift_points(self, objects_info, depth, focal_length, camera_rot=None, camera_trans=None, output=None, input_img=None):
        """
        FIXME
        Lift points from 2D to 3D using the point-lifting method.
        
        Returns representation of objects in 3D.
        """

        if isinstance(self.method, Gaussian3DLift):
            means_3d, covs_3d, valid_object_instances = self.method.gaussian_lift_points(objects_info, depth, focal_length, camera_rot, camera_trans)
            if output:
                # self.method.visualize_3d_gaussians_on_image(
                #     image_input=input_img,
                #     means_3d=means_3d, 
                #     covs_3d=covs_3d, 
                #     valid_indices=valid_object_instances, 
                #     focal_length=focal_length, 
                #     output_path=output, 
                #     camera_rot=camera_rot, 
                #     camera_trans=camera_trans, 
                #     labels=objects_info['class_ids'],
                #     std_scale=2.0
                # )

                self.method.visualize_3d_gaussians_in_3d(
                    means_3d=means_3d,
                    covs_3d=covs_3d,
                    valid_indices=valid_object_instances,
                    output_path=output,
                    labels=objects_info['class_ids'],
                    std_scale=2.0,
                    show_camera_origin=True
                )
                
            return means_3d, covs_3d, valid_object_instances