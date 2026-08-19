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
        else:
            pos = metadata['pos']
            rot = metadata['rot']
        
        # Convert out of Unity space FIRST: R1 is an OpenCV rotation, so
        # rectifying a Unity-convention quaternion mixes conventions (~0.53 deg,
        # ~1.8 cm lateral error at 2 m).
        pos, rot = self.coord_conversion_func((pos, rot))
        
        if self.rectifier:
            pos, rot = self.rectifier.rectified_left_pose(pos, rot)
        
        return pos, rot
