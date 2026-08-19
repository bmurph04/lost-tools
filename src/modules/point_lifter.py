from src.models.lift_gaussian_3d import Gaussian3DLift
import numpy as np

class PointLifter:
    def __init__(self, name, method):
        self.name = name
        self.method = method

    def lift_points(self, objects_point_list, depth, focal_length, optical_center, camera_rot, camera_pos, baseline=None):
        """
        FIXME
        Lift points from 2D to 3D using the point-lifting method.
        
        Returns representation of objects in 3D.
        """
        representation = None
        if self.name == 'gaussian_3d_lift':
            means_3d, covs_3d, point_clouds_list = self.method.gaussian_lift_points(
                objects_point_list, 
                depth, 
                focal_length,
                optical_center,
                camera_rot, 
                camera_pos,
                baseline=baseline
            )
            
            representation = means_3d, covs_3d, point_clouds_list
                
        return representation
    
    def visualize(self, frame, point_lifter_output, object_labels, output, focal_length=None, optical_center=None, camera_rot=None, camera_pos=None, camera_view_mode="isometric"):
        """
        
        """
        frame_copy = frame.detach().cpu().numpy().copy().transpose((1,2,0))
        # 2. Scale float32 [0.0, 1.0] to uint8 [0, 255] if necessary
        if frame_copy.dtype != np.uint8:
            if frame_copy.max() <= 1.0:
                frame_copy = (frame_copy * 255.0).astype(np.uint8)
            else:
                frame_copy = frame_copy.astype(np.uint8)

        frame_copy = np.ascontiguousarray(frame_copy)
        if self.name == 'gaussian_3d_lift':
            means_3d, covs_3d, point_clouds_list = point_lifter_output
            Gaussian3DLift.visualize_3d_gaussians_on_image(
                image_input=frame_copy,
                means_3d=means_3d, 
                covs_3d=covs_3d, 
                labels=object_labels,
                output_path=output,
                focal_length=focal_length,
                optical_center=optical_center,
                camera_rot=camera_rot, 
                camera_pos=camera_pos, 
            )
            
            # self.method.visualize_3d_gaussians_in_3d(
            #     means_3d=means_3d,
            #     covs_3d=covs_3d,
            #     instances=object_instances,
            #     labels=object_labels,
            #     output_path=output,
            #     std_scale=2.0,
            #     show_camera_origin=True,
            # )