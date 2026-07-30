from src.models.lift_gaussian_3d import Gaussian3DLift

class PointLifter:
    def __init__(self, method):
        self.method = method

    def lift_points(self, objects_point_list, depth, focal_length, camera_rot=None, camera_trans=None):
        """
        FIXME
        Lift points from 2D to 3D using the point-lifting method.
        
        Returns representation of objects in 3D.
        """
        representation = None
        if isinstance(self.method, Gaussian3DLift):
            means_3d, covs_3d, point_clouds_list, object_instances = self.method.gaussian_lift_points(
                objects_point_list, 
                depth, 
                focal_length, 
                camera_rot, 
                camera_trans
            )
            
            representation = means_3d, covs_3d, point_clouds_list
                
        return representation, object_instances
    
    def visualize(self, frame, focal_length, point_lifter_output, object_instances, object_labels, output, camera_rot=None, camera_trans=None):
        
        if isinstance(self.method, Gaussian3DLift):
            means_3d, covs_3d, point_clouds_list = point_lifter_output
            self.method.visualize_3d_gaussians_on_image(
                image_input=frame,
                means_3d=means_3d, 
                covs_3d=covs_3d, 
                instances=object_instances, 
                labels=object_labels,
                output_path=output,
                focal_length=focal_length, 
                camera_rot=camera_rot, 
                camera_trans=camera_trans, 
                std_scale=2.0
            )
            
            # self.method.visualize_3d_gaussians_in_3d(
            #     means_3d=means_3d,
            #     covs_3d=covs_3d,
            #     instances=object_instances,
            #     labels=object_labels,
            #     output_path=output,
            #     std_scale=2.0,
            #     show_camera_origin=True
            # )