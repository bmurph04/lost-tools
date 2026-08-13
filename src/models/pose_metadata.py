from src.utils import unity_pose_to_cv

class PoseMetadata:
    """World pose read from capture metadata."""


    def __init__(self, rectifier):
        self.rectifier = rectifier
        
        self.is_stereo = False if rectifier is None else True


    def __call__(self, metadata):
        if self.is_stereo:
            return self.rectifier.rectified_left_pose(metadata)
        
        return unity_pose_to_cv(metadata['leftCamera']['pos'],
                                metadata['leftCamera']['rot'])
