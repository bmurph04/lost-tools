import argparse
import yaml

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Lost Tools pipeline")
    # Path directory args
    parser.add_argument('--config', type=str, help="Path to yaml config for pipeline defaults")
    parser.add_argument("--input", type=str, help="Path to directory with streamed frames")
    parser.add_argument("--input-right", type=str, help="Path to directory with streamed frames from right camera")
    parser.add_argument("--input-metadata", type=str, help="Path to directory with streamed framed metadata")
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
    # Miscellaneous args
    parser.add_argument("--image-input-type", help="Mono or stereo streamed frames input")
    parser.add_argument("--estimate-intrinsics", type=str, help="Boolean to generate camera intrinsics for input frames, necessary for runtime")
    parser.add_argument("--estimate-pose", type=str, help="Boolean to generate camera pose for input frames, necessary for runtime")
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
            
    return config_data