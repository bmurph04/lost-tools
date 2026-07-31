import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

from rfdetr.assets.coco_classes import COCO_CLASSES
from src.utils import points_to_bbox

def build_2d_scene_graph(tracker_info, device, image=None, output=None):
    """
    fross_objects: List of tracked objects with ID, Class, and 3D Cartesian Centroid (x, y, z)
    """

    all_points = tracker_info['points'] # shape: (B, T, N, 2)
    all_class_ids = tracker_info['class_ids'] # shape: (B, N)
    all_class_instances = tracker_info['class_instances'] # shape: (B, N)
    all_class_confidences = tracker_info['class_confidences'] # shape: (B, N)
    features = tracker_info['features']

    # Get the points from the last timestep
    current_points = all_points[:, -1, :, :] # shape: (B, N, 2)
    num_batches = all_points.size(0)
    batch_scene_graphs = []
    
    for b in range(num_batches):

        points = current_points[b, :, :] # shape: (N, 2)
        class_ids = all_class_ids[b, :] # shape: (N,)
        instances = all_class_instances[b, :] # shape: (N,)
        # class_confidences = all_class_confidences[b, :] # shape: (N,)

        combined_keys = (class_ids * 1000) + instances # shape: (N,) assuming < 1000 instances

        # Get the unique keys and group sizes
        unique_keys, combined_key_counts = torch.unique_consecutive(combined_keys, return_counts=True)

        # Split into tensors of same (class_id, instance)
        grouped_points_list = list(torch.split(points, combined_key_counts.tolist()))


        # Build (D, 4) tensor of bboxes where D is number of object class ids
        bboxes_list = []
        for grouped_points in grouped_points_list:
            # Convert points info to bbox info
            bbox = points_to_bbox(grouped_points) # shape: (4, 2) (top_left, top_right, bottom_right, bottom_left)
            bboxes_list.append(bbox)
            
        bboxes = torch.stack(bboxes_list, dim=0).cpu().numpy() # shape: (D, 4, 2)
        # Get class_id labels for each box out of the keys
        class_labels = (unique_keys // 1000).cpu().numpy() # shape: (D,)
        # Get instance labels for each box out of the keys
        instance_labels = (unique_keys % 1000).cpu().numpy() # shape: (D,)

        widths = np.linalg.norm(bboxes[:, 3] - bboxes[:, 2], axis=1) + 1e-5
        centers = np.mean(bboxes, axis=1)
        top_centers = np.mean(bboxes[:, :2], axis=1)
        bottom_centers = np.mean(bboxes[:, 2:], axis=1)

        if image is not None:
            assert output, "must have an output dir for building 2d scene graph intermediate bboxes"
            vis_img = draw_bboxes_on_image(image, bboxes, class_labels)
            vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output), vis_img_bgr)

        # # Get the starting idxs for each box by cumulatively adding num_points of each key starting from 0 
        # box_starting_idxs = torch.cat([torch.tensor([0]), torch.cumsum(combined_key_counts, dim=0)[:-1]], dim=0)
        # # Get confidences for each box
        # box_confidences = class_confidences[box_starting_idxs]

        num_objects = len(bboxes)

        node_names = []
        for i in range(num_objects):
            class_name = COCO_CLASSES[class_labels[i]]
            node_names.append(f'{instance_labels[i]}_{class_name}')
        
        scene_graph = []
        
        for i in range(num_objects):
            for j in range(num_objects):
                # Skip comparing an object to itself
                if i == j:
                    continue
                
                a_node, b_node = node_names[i], node_names[j]

                if (b_node, "resting on", a_node) in scene_graph:
                    continue

                a_width, b_width = widths[i], widths[j]
                a_center, b_center = centers[i], centers[j]
                a_top_center, b_top_center = top_centers[i], top_centers[j]
                a_bottom_center, b_bottom_center = bottom_centers[i], bottom_centers[j]

                # near
                center_distance = np.linalg.norm(a_center - b_center)
                is_near = center_distance < 200.0
                        
                # on
                y_diff = a_bottom_center[1] - b_top_center[1]
                x_diff = abs(a_bottom_center[0] - b_top_center[0])

                is_on = -10.0 < y_diff < 30.0 and x_diff < max(a_width, b_width) * 0.5 and a_center[1] < b_center[1]

                print(f'{a_node}, {b_node}, {center_distance=}, {y_diff=}, {x_diff=}')

                # Bounded threshold: A's bottom must touch B's top within a small margin
                if is_on:
                    scene_graph.append((a_node, "resting on", b_node))
                    continue

                if is_near:
                    scene_graph.append((a_node, "near", b_node))

                # # --- 3. Interaction ("holding") ---
                # if class_a_str == 'person' and class_b_str != 'person':
                #     if center_distance_px < (a_height * 1.5):
                #         scene_graph.append((node_a, "holding", node_b))

        batch_scene_graphs.append(scene_graph)

    return batch_scene_graphs[0] if num_batches == 1 else batch_scene_graphs

def build_3d_scene_graph(fross_objects):
    """
    fross_objects: List of tracked objects with ID, Class, and 3D Cartesian Centroid (x, y, z)
    """
    scene_graph = []
    
    for obj_a in fross_objects:
        for obj_b in fross_objects:
            # Skip comparing an object to itself
            if obj_a.id == obj_b.id:
                continue
                
            pos_a = np.array(obj_a.centroid)
            pos_b = np.array(obj_b.centroid)
            
            # 1. Calculate 3D Euclidean Distance
            distance_3d = np.linalg.norm(pos_a - pos_b)
            
            # --- HEURISTIC RULES ---
            
            # Rule A: Proximity ("near")
            # If the objects are within 30 centimeters of each other
            if distance_3d < 0.1:
                scene_graph.append((obj_a, "near", obj_b))
                
            # Rule B: Vertical Stacking ("resting_on")
            # We check if X and Y (the flat plane) are tightly aligned, 
            # and if obj_a's Z (height) is resting directly on top of obj_b
            xy_distance = np.linalg.norm(pos_a[:2] - pos_b[:2])
            z_difference = pos_a[2] - pos_b[2]
            
            if xy_distance < 0.1 and 0 < z_difference < 0.05:
                scene_graph.append((obj_a, "resting_on", obj_b))
                
            # Rule C: Hand-Tool Interaction ("holding")
            if obj_a.class_name == "Surgeon_Hand" and obj_b.class_name in ["Scalpel", "Forceps"]:
                if distance_3d < 0.05: # Very tight threshold
                    scene_graph.append((obj_a, "holding", obj_b))

    return scene_graph

def save_scene_graph_frame(triplets, output):
    """
    Generates a static .png of the directed graph for a specific frame.
    """
    # 1. Initialize the Directed Graph
    G = nx.DiGraph()
    
    # 2. Populate Nodes and Edges
    for subj, pred, obj in triplets:
        G.add_edge(subj, obj, label=pred)
        
    # 3. Setup the Matplotlib Figure (e.g., 1280x720 resolution)
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    
    # 4. Calculate Node Layout (Spring layout simulates anti-gravity repulsion)
    # Using a fixed seed prevents the layout from violently spinning between frames
    pos = nx.spring_layout(G, seed=42, k=0.5) 
    
    # 5. Draw the Components
    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightgray', 
                           node_size=2000, edgecolors='black')
    
    # Draw Edges (Arrows)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', 
                           arrows=True, arrowsize=20, node_size=2000)
    
    # Draw Node Text
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
    
    # Draw Edge Text (Predicates)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.axis('off') # Hide the standard chart axes
    
    # Save as a zero-padded frame (e.g., frame_0001.png, frame_0002.png)
    plt.savefig(output, format="PNG", bbox_inches='tight')
    
    # CRITICAL: Close the figure to prevent a massive memory leak in your loop
    plt.close(fig)

# --- Example Usage in your Loop ---
# fross_triplets = [("Surgeon_Hand_1", "holding", "Scalpel_10"), ...]
# save_scene_graph_frame(fross_triplets, frame_index=1)


def draw_bboxes_on_image(image: np.ndarray, bboxes, labels=None, color=(0, 255, 0), thickness=2):
    """
    Draws bounding boxes defined by 4 corner points on an image.
    
    Args:
        image: Original image array (H, W, 3) in BGR format.
        bboxes: Tensor or Numpy array of shape (N, 4, 2) containing corner points.
        labels: Optional list of strings for labeling each box.
        color: Tuple representing BGR color (default is Green).
        thickness: Line thickness for the boxes.
        
    Returns:
        np.ndarray: A copy of the image with the bounding boxes drawn.
    """
    # Create a copy so we don't overwrite the original image in memory
    vis_img = image.copy()
    
    # Safely move to CPU and convert to NumPy if it's a PyTorch tensor
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
        
    for i, bbox in enumerate(bboxes):
        # OpenCV requires integer coordinates for drawing
        pts = np.round(bbox).astype(np.int32)
        
        # Draw the 4 points as a closed polygon
        cv2.polylines(vis_img, [pts], isClosed=True, color=color, thickness=thickness)
        
        # Draw labels if provided
        if labels is not None and i < len(labels):
            # Find the highest point (min Y) to place the text above the box
            top_point = pts[np.argmin(pts[:, 1])]
            text_pos = (int(top_point[0]), max(int(top_point[1]) - 8, 15))
            
            # Add a slight black outline to text for readability against light backgrounds
            cv2.putText(vis_img, str(labels[i]), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(vis_img, str(labels[i]), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 1, cv2.LINE_AA)
            
    return vis_img