from src.utils import unity_pose_to_cv

class PoseMetadata:
    """World pose read from capture metadata."""


    def __init__(self, rectifier, input_camera_coords):
        self.rectifier = rectifier
        
        self.is_stereo = False if rectifier is None else True
        
        if input_camera_coords == 'unity':
            self.conversion_func = unity_pose_to_cv
        else:
            self.conversion_func = lambda x: x

    def __call__(self, metadata):
        
        if self.is_stereo:
            pos, trans = self.rectifier.rectified_left_pose(pos, trans)
 
        return self.conversion_func(pos, trans)
