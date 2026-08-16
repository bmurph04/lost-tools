from src.utils import unity_pose_to_cv

class PoseMetadata:
    """World pose read from capture metadata."""

    def __init__(self, rectifier, coord_conversion_func = lambda x: x):
        self.rectifier = rectifier
        self.coord_conversion_func = coord_conversion_func

    def get_pose(self, metadata):
        """
        Get the pose from metadata

        Args:
            metadata (_type_): _description_
            coord_conversion_func (_type_, optional): _description_. Defaults to lambdax:x.

        Returns:
            _type_: _description_
        """
        if self.rectifier:
            pos = metadata['leftCamera']['pos']
            rot = metadata['leftCamera']['rot']
            pos, rot = self.rectifier.rectified_left_pose(pos, rot)
        
        else:
            pos = metadata['pos']
            rot = metadata['rot']
        
        return self.coord_conversion_func((pos, rot))
