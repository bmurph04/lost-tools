class GeometryProvider:
    """Sequence-constant camera geometry. Stereo when a rectifier is present."""

    def __init__(self, rectifier=None, focal_length=None, optical_center=None, baseline=None):
        self.rectifier = rectifier
        self.focal_length = focal_length
        self.optical_center = optical_center
        self.baseline = baseline

    @property
    def is_stereo(self):
        return self.rectifier is not None

    def prepare(self, left_chw, right_chw):
        """Rectify the pair, or pass the left image through and drop the right."""
        if self.rectifier is None:
            return left_chw, None
        return self.rectifier.rectify_pair(left_chw, right_chw)


def build_geometry(depth_source, geometry_source, config, intrinsics_source=None, width=None, height=None):
    """
    Build a GeometryProvider class based on the values of depth_source and geometry_source.

    Args:
        depth_source (_type_): _description_
        geometry_source (_type_): _description_
        config (_type_): _description_
        metadata0 (_type_): _description_
        width (_type_): _description_
        height (_type_): _description_

    Returns:
        GeometryProvider: GeometryProvider class that stores constant camera geometry and provides rectifying 
        method if applicable.
    """
    if depth_source == 'mono':
        
        if geometry_source == 'estimate':
            # Populate GeometryProvider on warmup
            return GeometryProvider()
        else:
            
            if geometry_source == 'metadata':
                intrinsics = intrinsics_source['leftCamera']
                
            elif geometry_source == 'external':
                intrinsics = load_serialized_data(intrinsics_source)
                
            return GeometryProvider(focal_length=(intrinsics['fx'], intrinsics['fy']), optical_center=(intrinsics['cx'], intrinsics['cy']))
    
    elif depth_source == 'stereo':
        
        if geometry_source == 'metadata':
            
            
        src = metadata0 if geometry_source == 'metadata' \
            else load_serialized_data(config['stereo_calibration'])
        rectifier = StereoRectifier(src, width, height)
        return Geometry(rectifier=rectifier,
                        focal_length=rectifier.focal_length,
                        optical_center=rectifier.optical_center,
                        baseline=rectifier.baseline)

    else:
        raise ValueError(f"Invalid depth_source {depth_source} was passed into build_geometry method")
