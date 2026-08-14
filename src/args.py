import argparse
import yaml
import warnings

def parse_and_validate_args() -> dict:

    parser = argparse.ArgumentParser(description="Lost Tools pipeline")
    # Path directory args
    parser.add_argument('--config', type=str, help="Path to yaml config for pipeline defaults")
    parser.add_argument("--input", type=str, help="Path to directory with streamed frames")
    parser.add_argument("--input-right", type=str, help="Path to directory with streamed frames from right camera")
    parser.add_argument("--input-metadata", type=str, help="Path to directory with streamed framed metadata")
    parser.add_argument("--input-geometry", type=str, help="Path to camera geometry with intrinsics and relative extrinsics")
    parser.add_argument("--output", type=str, help="Folder path to output visualizations to")
    # Tracker args
    parser.add_argument("--tracker-config", type=str, help="Path to tracker model config .yaml")
    parser.add_argument("--tracker-ckpt", type=str, help="Path to tracker model checkpoint")
    # 2D scene graph generator args
    parser.add_argument("--sgg2d-config", type=str, help="Path to 2D scene graph generator model config .yaml")
    parser.add_argument("--sgg2d-ckpt", type=str, help="Path to 2D scene graph generator model checkpoint")
    # Depth estimator args
    parser.add_argument("--depth-config", type=str, help="Path to depth estimator model config")
    parser.add_argument("--depth-ckpt", type=str, help="Path to depth estimator model checkpoint")
    # Pose estimator args
    parser.add_argument("--pose-config", type=str, help="Path to pose estimator model config .yaml")
    parser.add_argument("--pose-ckpt", type=str, help="Path to pose estimator model checkpoint")
    # Miscellaneous args, TODO: FIX HELP DESCRIPTIONS
    parser.add_argument("--depth-source", type=str, help="Boolean to ???, necessary for runtime")
    parser.add_argument("--geometry-source", type=str, help="Boolean to ???, necessary for runtime")
    parser.add_argument("--pose-source", type=str, help="Boolean to ???, necessary for runtime")
    parser.add_argument("--visualize", type=str, help="Boolean to visualize each step of pipeline")
    
    args = parser.parse_args()
    
    config_data = {}
    
    if args.config:
        try:
            with open(args.config, 'r') as file:
                config_data = yaml.safe_load(file) or {}
        except:
            raise FileNotFoundError(f'Error: config file {args.config} not found.')
        
    for key, value in vars(args).items():
        if key != 'config' and value is not None:
            config_data[key] = value
          
    # Ensure combination of args are valid  
    validate_args(config_data)
    
    return config_data

def validate_args(config_data):
    depth_source = config_data['depth_source']
    geometry_source = config_data['geometry_source']
    pose_source = config_data['pose_source']

    # Invalidate source value if not recognized
    if depth_source not in ('mono', 'stereo'):
        raise ValueError(f"depth_source {depth_source} is not recognized. Please only use 'mono' or 'stereo'.")
    if geometry_source not in ('metadata', 'estimation'):
        raise ValueError(f"geometry_source {geometry_source} is not recognized. Please only use 'metadata' or 'estimation'.")
    if pose_source not in ('metadata', 'estimation'):
        raise ValueError(f"pose_source {pose_source} is not recognized. Please only use 'metadata' or 'estimation'.")
    
    input_camera_coords = config_data.get('input_camera_coords')
    # Invalidate input coords if not recognized
    if input_camera_coords and input_camera_coords not in ('unity',):
        raise ValueError(f"input camera coordinates {input_camera_coords} was not recognized or supported by this system.")
    
    # Invalidate stereo depth_source and estimation geometry_source combination
    if depth_source == 'stereo' and geometry_source == 'estimation':
        raise ValueError(
            """
            Using stereo images to retrieve depth requires knowledge of cameras' geometry (relative position/rotation
            between left and right camera), and estimating this would rely on noisy per-frame intrinsics/pose 
            estimations that would lead to cascading metric error. Please provide a geometry_source to retrieve 
            camera geometry from, or change depth_source to 'mono' to run single-camera geometry estimations.
            """
        )
        
    # Invalidate depth_source value 'stereo' and no input_right directory
    if depth_source == 'stereo' and config_data.get('input_right') is None:
        raise ValueError("depth_source 'stereo' requires input_right")
        
    # Invalidate geometry_source and pose_source value 'metadata' and no input_metadata directory
    if 'metadata' in (geometry_source, pose_source) and config_data.get('input_metadata') is None and config_data.get('input_geometry') is None:
        raise ValueError("geometry_source/pose_source 'metadata' requires input_metadata")    