import torch
import numpy as np
from pathlib import Path
from argparse import Namespace
import warnings
from tqdm import tqdm
import sys
from scipy.spatial.transform import Rotation

# Add external repo root to sys.path
ffs_path = Path(__file__).resolve().parent.parent / "external" / "fast_foundationstereo"
if str(ffs_path) not in sys.path:
    sys.path.insert(0, str(ffs_path))

# -- external model imports --
from external.rfdetr.src.rfdetr import RFDETRMedium
from external.track_on.model.trackon_predictor import Predictor
from external.unidepth.unidepth.models.unidepthv2.unidepthv2 import UniDepthV2
from core.foundation_stereo import FastFoundationStereo
from external.DPVO.dpvo.dpvo import DPVO

# -- lost-tools modules --
from src.modules.detector import Detector
from src.modules.tracker import Tracker
from src.modules.stereo_rectifier import StereoRectifier
from src.modules.depth_provider import DepthProvider
from src.modules.pose_provider import PoseProvider
# from src.sgg2d import SceneGraphGenerator2D
from src.modules.point_lifter import PointLifter
from src.modules.scene_graph_generator_3d import SceneGraphGenerator3D
from src.modules.dynamic_scene_graph_3d import DynamicSceneGraph3D
from src.modules.system_eval import SystemEvaluator

# -- lost-tools models and methods --
from src.models.lift_gaussian_3d import Gaussian3DLift
from src.models.build_geometric_3dsg import Geometric3DSGBuilder
# from src.custom_react_model import CustomReactModel

# -- lost-tools misc --
from src.args import parse_and_validate_args
from src.utils import pick_device, load_serialized_data, load_frame, load_checkpoint, egoobjects_sort_key

# global vars
WARMUP_FRAMES = 8
DETECTOR_FREQ = 5
PRED_NAMES = ['near', 'on'] # NOTE: predicate names are static for current implementation, should change if preds are generated

def main() -> None:

    # ----- Initialization setup -----
    config_dict = parse_and_validate_args()
    device = pick_device()

    frames_dir = sorted([f for f in Path(config_dict['input']).iterdir()], key=egoobjects_sort_key) # Sort input frame seq
    _, image_height, image_width = load_frame(frames_dir[0]).shape # Grab the first frame to get image width and height

    if config_dict['input_right'] is not None:
        right_frames_dir = sorted([f for f in Path(config_dict['input_right']).iterdir()], key=egoobjects_sort_key)
        assert len(right_frames_dir) == len(frames_dir), \
            f"Sequence length mismatch, left frames_dir has {len(frames_dir)} frames but right_frames_dir has {len(frames_dir)} frames"

    if config_dict['input_metadata'] is not None:
        # TODO: Validate metadata schema
        metadata_dir = sorted([f for f in Path(config_dict['input_metadata']).iterdir()], key=egoobjects_sort_key)
        assert len(metadata_dir) == len(frames_dir), \
            f"Sequence length mismatch, left frames_dir has {len(frames_dir)} frames but metadata_dir has {len(metadata_dir)} frames"

        metadata_frame0 = load_serialized_data(metadata_dir[0])
        
    
    # If depth_source is stereo,
        # Initialize a rectifier, necessary for correct mapping of pixels across camera frames. Already validated geometry source is not estimation
        # Initialize a stereo depth estimator model.
        
    # Else, depth_source is mono.
        # Initialize a mono depth estimator model.
        
    # If pose_source is metadata,
        # Initialize a pose metadata retriever.
    # Else, pose_source is estimation.
        # Initialize a pose estimator model.
        
    # If geometry_source is metadata or external,
        # Initialize geometry provider with intrinsic data
    # Else, geometry_source is estimation.
        # Initialize geometry provider with no intrinsic data.
        
    if config_dict['geometry_source'] == 'metadata':
        pass
     
    # Initialize depth provider module
    if config_dict['depth_source'] == 'stereo':
        # Initialize a rectifier, necessary for correct mapping of pixels across camera frames
        # Already validated geometry source is not estimation
        rectifier = StereoRectifier() # TODO: fix this line
        
        # Initialize a stereo depth estimator model
        depth_model = torch.load(config_dict['depth_ckpt'], map_location=device, weights_only=False) # FastFoundationStereo
        depth_model.eval()
        depth_model.to(device)
    else:
        # Initialize a mono depth estimator model
        depth_model = UniDepthV2(load_serialized_data(config_dict['depth_config']))
        depth_model.load_state_dict(load_checkpoint(config_dict['depth_ckpt']), strict=False)
        depth_model.resolution_level = 4
    
    depth_estimator = DepthProvider(device, depth_model)
    
    # Initialize pose provider module
    if config_dict['pose_source'] == 'metadata':
        # Initialize a pose metadata provider
        pass
    else:
        # Initialize a pose estimator model
        pose_model = DPVO(load_serialized_data(config_dict['pose_config']), config_dict['pose_ckpt'], ht=image_height, wd=image_width) # FIXME: Set H and W params later
    
    pose_provider = PoseProvider(device, pose_model)
    
    
    if config_dict['pose_source'] == 'metadata':
        pass
    else:
        # Initialize pose model
        pass
    
    
    
    
    # # Initialize geometry provider module, which provides camera geometry based on depth_source and geometry_source
    # geometry_provider = build_geometry(config_dict['depth_source'], config_dict['geometry_source'], intrinsics_source)
    
    # Initialize detector model and module
    detector_model = RFDETRMedium() # pretrained weights are downloaded within init
    # with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    detector_model.inference()
    detector = Detector(device, detector_model)

    # Initialize tracker model and module
    tracker_model = Predictor(model_args=Namespace(**load_serialized_data(config_dict['tracker_config'])), checkpoint_path=config_dict['tracker_ckpt'], support_grid_size=0)
    tracker_model.eval()
    tracker_model.to(device)
    tracker = Tracker(device, tracker_model)

    # Initialize depth estimator model and module
    image_input_type = config_dict['image_input_type']
    if image_input_type == 'mono':
        # depth_model, depth_preprocessing_transform = depth_pro.create_model_and_transforms() 
        depth_model = UniDepthV2(load_serialized_data(config_dict['depth_config']))
        depth_model.load_state_dict(load_checkpoint(config_dict['depth_ckpt']), strict=False)
        depth_model.resolution_level = 4
    elif image_input_type == 'stereo':
        depth_model = torch.load(config_dict['depth_ckpt'], map_location=device, weights_only=False) # FastFoundationStereo
        depth_model.eval()
        depth_model.to(device)
        depth_estimator = DepthProvider(device, depth_model)

    if estimate_pose:
        pose_model = DPVO(load_serialized_data(config_dict['pose_config']), config_dict['pose_ckpt'], ht=image_height, wd=image_width) # FIXME: Set H and W params later
        pose_estimator = DepthProvider(device, pose_model)

    # Initialize point lifting method and module
    # FIXME: pass in args to configure
    point_lifting_method = Gaussian3DLift()
    point_lifter = PointLifter(point_lifting_method)

    # Initialize 3D scene graph generator method and module
    # FIXME: pass in args to configure
    scene_graph_gen_3d_method = Geometric3DSGBuilder()
    scene_graph_generator_3d = SceneGraphGenerator3D(sgg_method=scene_graph_gen_3d_method, point_lifting_method=point_lifting_method)
    
    # Initialize dynamic 3D scene graph class
    dynamic_scene_graph = DynamicSceneGraph3D(point_lifting_method=point_lifting_method)
    
    # Initialize system evaluator module for metrics
    sys_evaluator = SystemEvaluator(device=device)

    # Initialize maps for relation predicate names to predicate ids
    # NOTE: Static for current implementation, dynamic if preds are generated
    pred_name_to_id = {name: id for id, name in enumerate(PRED_NAMES)} 
    pred_id_to_name = PRED_NAMES
    
    objects_info = {
        'points': [], # size D list, shape (n, 2) tensors
        'object_point_counts': [], # size D list of integers
        'class_ids': [], # size D list of integers
        'confidences': [], # side D list of integers
    } # Object info container that is updated with each frame

    # Initialize variables for camera intrinsics/extrinsics estimation
    focal_length = (None, None) # Focal length of the camera lens
    optical_center = (None, None) # Optical center coordinate of the camera
    camera_baseline = None # Distance between two cameras (in stereo)
    camera_rot, camera_trans = None, None # Euler rotation and translation of the camera
    intrinsics_buffer = [] # Buffer to estimate intrinsics after warmup
    pose_to_depth_scale = None

    # Initialize output strings
    detector_output_prefix = f'outputs/{config_dict['output']}/output_detector'
    tracker_output_prefix = f'outputs/{config_dict['output']}/output_tracker'
    depth_estimator_output_prefix = f'outputs/{config_dict['output']}/output_depth_estimator'
    pose_estimator_output_prefix = f'outputs/{config_dict['output']}/output_pose_estimator'
    point_lifter_output_prefix = f'outputs/{config_dict['output']}/output_point_lifter'
    sgg3d_output_prefix = f'outputs/{config_dict['output']}/output_sgg3d'
    dynamic_sg_output_prefix = f'outputs/{config_dict['output']}/output_dynamic_sg'

    # Initialize other miscellaneous variables before frame loop
    estimate_intrinsics = config_dict['estimate_intrinsics']
    estimate_pose = config_dict['estimate_pose']
    visualize = config_dict['visualize']
    test_speed = False # Set to True on the frame that speed tests should begin
    
    # ----- Main loop -----
    with torch.inference_mode():
        for t in tqdm(range(len(frames_dir))):
            # Load all necessary data from input directories
            frame_path = str(frames_dir[t])     
            frame = torch.from_numpy(load_frame(frame_path)) # shape: (3, H, W)
            
            if right_frames_dir is not None:
                right_frame_path = str(right_frames_dir[t])
                right_frame = torch.from_numpy(load_frame(right_frame_path))
            
            if metadata_dir is not None:
                frame_metadata_path = str(metadata_dir[t])
                frame_metadata = load_serialized_data(frame_metadata_path)

            if not estimate_intrinsics:
                focal_length = frame_metadata['leftCamera']['fx'], frame_metadata['leftCamera']['fy']
                optical_center = frame_metadata['leftCamera']['cx'], frame_metadata['leftCamera']['cy']
                
            if not estimate_pose:
                camera_trans = np.array([frame_metadata['leftCamera']['pos'][0], frame_metadata['leftCamera']['pos'][1], frame_metadata['leftCamera']['pos'][2]])
                camera_quat = np.array([frame_metadata['leftCamera']['rot'][0], frame_metadata['leftCamera']['rot'][1], frame_metadata['leftCamera']['rot'][2], frame_metadata['leftCamera']['rot'][3]])
                camera_rot = Rotation.from_quat(camera_quat).as_matrix()
                camera_baseline = abs(frame_metadata['rightCamera']['pos'][0] - frame_metadata['leftCamera']['pos'][0]) if right_frames_dir is not None else None
                
            test_speed = True if t == WARMUP_FRAMES else None

            sys_evaluator.start_speed_test('frame') if test_speed else None 

            # ----- Depth Estimator -----
            # Create the depth estimation input based on the image input type
            if image_input_type == 'mono':
                depth_est_input = frame
            elif image_input_type == 'stereo':
                depth_est_input = (frame, right_frame)
            else:
                raise RuntimeError(f"Image input type '{image_input_type} not supported in main pipeline")
            
            # Process frame using depth estimator
            sys_evaluator.start_speed_test('depth_estimator') if test_speed else None
            depth, frame_focal_length_est, frame_optical_center_est = depth_estimator.process_frame(depth_est_input, focal_length=focal_length, baseline=camera_baseline)
            sys_evaluator.end_speed_test('depth_estimator') if test_speed else None
            depth_estimator.visualize(depth, output=f'{depth_estimator_output_prefix}_{t:06d}.jpg') if visualize else None

            if estimate_intrinsics: 
                # If still in warmup frames, add results to intrinsics buffer for later averaging
                if t < WARMUP_FRAMES:
                    frame_intrinsics = [frame_focal_length_est[0], frame_focal_length_est[1], frame_optical_center_est[0], frame_optical_center_est[1]]
                    intrinsics_buffer.append(frame_intrinsics)
                    # Allow intrinsics to vary frame to frame for now before we freeze it frame number WARMUP_FRAMES
                    focal_length, optical_center = frame_focal_length_est, frame_optical_center_est
                elif t == WARMUP_FRAMES:
                    # Average the intrinsics buffer to get fixed camera intrinsics for the rest of the sequence
                    intrinsics_est = np.median(intrinsics_buffer, axis=0)
                    focal_length = intrinsics_est[0], intrinsics_est[1]
                    optical_center = intrinsics_est[2], intrinsics_est[3]
            
            if estimate_pose:
                # ----- Pose Estimator -----
                # Run pose estimation
                sys_evaluator.start_speed_test('pose_estimator')
                # Get camera pose from pose estimator
                camera_rot, camera_trans = pose_estimator.process_frame(frame, t, focal_length, optical_center)
                sys_evaluator.end_speed_test('pose_estimator')
                
                if t == WARMUP_FRAMES:
                     # Get pose to depth scale
                    pose_to_depth_scale = pose_estimator.get_metric_scaling(t, depth)
                    
                # We need to know how to convert between the units these modules use to have accurate 3D camera translation tracking
                # Apply scaling if it's been found
                if pose_to_depth_scale is not None:
                    camera_trans = pose_to_depth_scale * camera_trans

                pose_estimator.visualize(frame, camera_rot, camera_trans, output=f'{pose_estimator_output_prefix}_{t:06d}.jpg') if visualize else None
                                
            # ----- Tracker -----
            num_total_points = sum(objects_info['object_point_counts'])
            has_active_points = num_total_points > 0
            points_list, visibles_list = [], []
            # Process frame if there are active points 
            if has_active_points:
                # Set tracker initial capacity based on object point count
                tracker.model.initial_capacity = num_total_points # FIXME: make general for any tracker
                # Process frame using tracker
                sys_evaluator.start_speed_test('tracker') if test_speed else None
                points_list, visibles_list = tracker.process_frame(frame, objects_info['object_point_counts'])
                sys_evaluator.end_speed_test('tracker') if test_speed else None 
                tracker.visualize(frame, points_list, visibles_list, output=f'{tracker_output_prefix}_{t:06d}.jpg') if visualize else None

                # Store points list at frame t
                objects_info['points'] = points_list

            if t % DETECTOR_FREQ == 0:
                # ----- Detector -----
                # Process frame using detector
                sys_evaluator.start_speed_test('detector') if test_speed else None
                detections_info = detector.process_frame(frame)
                sys_evaluator.end_speed_test('detector') if test_speed else None
                detector.visualize(frame, detections_info, output=f'{detector_output_prefix}_{t:06d}.jpg') if visualize else None
                
                # Filter detections againt updated tracker point positions
                detections_info = detector.filter_detections_info(detections_info, objects_info)

                # FIXME: comment description
                if detections_info is not None:
                    # Using the detector bbox info, greate grid of queries for each object
                    new_points_list, new_object_point_counts = tracker.build_detection_grid_points(
                        detections_info, 
                        frame_extent=(image_height, image_width), 
                        margin_div=8
                    )

                    objects_info['points'].extend(new_points_list)
                    objects_info['object_point_counts'].extend(new_object_point_counts)
                    objects_info['class_ids'].extend(detections_info['class_ids'])
                    objects_info['confidences'].extend(detections_info['class_confidences'])
                    tracker.initialize_queries(frame, new_points_list)
                    
                    if visualize:
                        new_visibles_list = [torch.ones(points.shape[0], dtype=torch.bool, device=points.device) 
                                                                for points in new_points_list]
                        visibles_list.extend(new_visibles_list)
                        tracker.visualize(frame, objects_info['points'], visibles_list, output=f'{tracker_output_prefix}_{t:06d}.jpg')

            # ----- Point Lifting to 3D -----
            sys_evaluator.start_speed_test('point_lifter') if test_speed else None
            points3d_representation = point_lifter.lift_points(
                objects_point_list=objects_info['points'], 
                depth=depth, 
                focal_length=focal_length,
                optical_center=optical_center,
                camera_rot=camera_rot,
                camera_trans=camera_trans
            )
            sys_evaluator.end_speed_test('point_lifter') if test_speed else None
            point_lifter.visualize(
                frame,
                focal_length=focal_length,
                optical_center=optical_center,
                camera_rot=camera_rot,
                camera_trans=camera_trans,
                point_lifter_output=points3d_representation, 
                object_labels=objects_info['class_ids'],
                output=f'{point_lifter_output_prefix}_{t:06d}.jpg', 
            ) if visualize else None
                            
            # ----- 3D Scene Graph Generator -----
            sys_evaluator.start_speed_test('3dsg_gen') if test_speed else None
            scene_graph_3d = scene_graph_generator_3d.generate_triplets(points3d_representation, pred_name_to_id)
            sys_evaluator.end_speed_test('3dsg_gen') if test_speed else None
            scene_graph_generator_3d.visualize(
                frame=frame,
                focal_length=focal_length,
                optical_center=optical_center,
                camera_rot=camera_rot,
                camera_trans=camera_trans,
                scene_graph=scene_graph_3d,
                pred_id_to_name=pred_id_to_name,
                points_representation=points3d_representation,
                object_labels=objects_info['class_ids'],
                output=f'{sgg3d_output_prefix}_{t:06d}.jpg',
                camera_view_mode="aligned",
                show_camera=False,
                auto_zoom=True,
                x_range=(-0.3,0.5),
                y_range=(-0.1,0.4),
                z_range=(1.0, 1.8),
                std_scale=1.0
            ) if visualize else None
            
            if t >= WARMUP_FRAMES:
                # ----- 3D Scene Graph Merging -----
                sys_evaluator.start_speed_test('3dsg_merge')
                update_idx = dynamic_scene_graph.add(
                    object_labels=objects_info['class_ids'], 
                    points_representation=points3d_representation, 
                    triplets=scene_graph_3d
                )
                dynamic_scene_graph.merge(update_idx)
                sys_evaluator.end_speed_test('3dsg_merge')
                dynamic_scene_graph.visualize(
                    frame=frame,
                    focal_length=focal_length,
                    optical_center=optical_center,
                    camera_rot=camera_rot,
                    camera_trans=camera_trans,
                    pred_id_to_name=pred_id_to_name,
                    output=f'{dynamic_sg_output_prefix}_{t:06d}.jpg'
                ) if visualize else None
            
            sys_evaluator.end_speed_test('frame') if test_speed else None

            # Print metrics
            if t % 50 == 0:
                sys_evaluator.print_latency_metrics()
            
    # ----- Cleanup and evaluation -----
    # Print metrics
    sys_evaluator.print_latency_metrics()


if __name__ == "__main__":
    main()