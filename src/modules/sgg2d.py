import torch
import cv2
import numpy as np
from pathlib import Path
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph 

class SceneGraphGenerator2D:
    """
    2D Scene graph generator class that generates a per-frame scene graph.

    Args:
        model - 2D scene graph generator model.
    """
    def __init__(self, device, model):

        self.device = device
        self.model = model

        self.bbox_threshold = 0.5
        self.rel_threshold = 0.1
    
    def process_tracking_data(self, tracking_info, extent, output=None):
        """
        Given ..., process a frame using the initialized scene graph generator.

        Return a scene graph.
        """
        with torch.inference_mode():
            predictions = self.model.forward(tracking_info, extent)
            # Output is a list of dicts, but only one frame so grab its prediction
            prediction = predictions[0]

        # Decode pytorch dict into nodes (bboxes) and edges (relations)
        nodes, rels = self._post_process_simplified(prediction)

        print(
            f"nodes={len(nodes)}, rels={len(rels)}, "
            f"object_threshold={self.bbox_threshold}, "
            f"relation_threshold={self.rel_threshold}"
        )

        if output:
            # Convert to numpy for rendering
            nodes_np = nodes.cpu().numpy()
            rels_np = rels.cpu().numpy()

            # Generate visual of topological directed graph
            graph_image = self.visualize_graph(rels_np, nodes_np)
            graph_image_bgr = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output, graph_image_bgr)
        
        return nodes, rels
    
    
    def visualize_graph(self, rels: np.ndarray, bboxes: np.ndarray, color: str = 'blue') -> np.ndarray:
        """
        Renders a pure directed topological graph diagram using NetworkX and Graphviz.
        Displays all detected nodes even if no relationship edges exist.
        """
        # 1. Only exit if there are zero detected object nodes
        if len(bboxes) == 0:
            return np.full((400, 600, 3), 255, dtype=np.uint8)

        G = nx.MultiDiGraph()
        obj_classes = getattr(self.model, 'obj_classes', [])
        bbox_labels = [b[5] for b in bboxes]

        # 2. Pre-create and add ALL nodes to NetworkX first
        node_names = []
        for idx, label_val in enumerate(bbox_labels):
            cls_id = int(label_val)
            cls_name = obj_classes[cls_id] if cls_id < len(obj_classes) else f"Class_{cls_id}"
            node_str = f"{cls_name}\n(ID: {idx})"
            
            node_names.append(node_str)
            G.add_node(node_str)

        # 3. Add relationship edges ONLY if relations exist
        if rels is not None and len(rels) > 0:
            for i, r_label in enumerate(rels[:, 2]):
                rel_idx = int(r_label)
                
                label_rel = self.model.rel_classes[rel_idx] if rel_idx < len(self.model.rel_classes) else f"Rel_{rel_idx}"
                if label_rel == "beside":
                    label_rel = "near"

                r = rels[i].astype(int)
                
                subj = node_names[r[0]]
                obj = node_names[r[1]]
                
                G.add_edge(subj, obj, label=label_rel, color=color)

        # Graphviz dot styling configuration
        G.graph['edge'] = {'arrowsize': '0.6', 'splines': 'curved', 'color': color, 'fontcolor': 'black'}
        G.graph['graph'] = {'scale': '2', 'bgcolor': 'white'}
        G.graph['node'] = {'shape': 'rectangle', 'color': color, 'fontcolor': 'black', 'style': 'bold'}

        # Compile to Graphviz AGraph and render to buffer
        img_graph = to_agraph(G)
        img_graph.layout('dot')
        png_byte_array = img_graph.draw(format='png', prog='dot')

        # Decode buffer directly into an OpenCV BGR image
        graph_cv2 = cv2.imdecode(np.frombuffer(png_byte_array, np.uint8), cv2.IMREAD_COLOR)

        return graph_cv2
    
    def _post_process_simplified(self, pred_dict):
        """
        Post-processes predictions and maps all relations to strictly 'on' or 'near'.
        It uses the correct dataset vocabulary indices for downstream consistency.
        """
        bbox_scores = pred_dict['pred_scores']
        bbox_labels = pred_dict['pred_labels']
        raw_boxes   = pred_dict['boxes']

        # Filter objects by threshold
        valid_obj_mask = bbox_scores >= self.bbox_threshold
        filtered_bbox_ids = torch.where(valid_obj_mask)[0]

        if filtered_bbox_ids.numel() == 0:
            return torch.tensor([]), torch.empty((0, 4), device=bbox_scores.device)

        # Build node tensor: (N, 6) -> [x1, y1, x2, y2, score, label]
        node_tensor = torch.cat([
            raw_boxes[filtered_bbox_ids].float(),
            bbox_scores[filtered_bbox_ids].unsqueeze(1).float(),
            bbox_labels[filtered_bbox_ids].unsqueeze(1).float(),
        ], dim=1)

        rel_scores_full = pred_dict['pred_rel_scores']  # (M, num_rel_classes)
        pairs           = pred_dict['rel_pair_idxs']      # (M, 2)

        if pairs.shape[0] == 0:
            return node_tensor, torch.empty((0, 4), device=bbox_scores.device)

        # -----------------------------------------------------------------
        # 1. Define the Two Candidate Pools
        # -----------------------------------------------------------------
        rel_classes = self.model.rel_classes  # List of string labels

        on_words   = ['on', 'standing on', 'sitting on', 'lying on', 'attached to', 'painted on', 'on back of', 'parked on', 'riding']
        near_words = ['beside', 'in front of', 'touching', 'crossing', 'enclosing', 'leaning on']

        idx_on   = [i for i, name in enumerate(rel_classes) if name in on_words]
        idx_near = [i for i, name in enumerate(rel_classes) if name in near_words]

        # Max score per group for each pair (M,)
        score_on   = rel_scores_full[:, idx_on].max(dim=1)[0]   if len(idx_on) > 0   else torch.zeros(pairs.shape[0], device=pairs.device)
        score_near = rel_scores_full[:, idx_near].max(dim=1)[0] if len(idx_near) > 0 else torch.zeros(pairs.shape[0], device=pairs.device)

        # Compare the two clusters: [M, 2] -> (0: "on" cluster won, 1: "near" cluster won)
        simplified_scores = torch.stack([score_on, score_near], dim=1)
        max_scores, best_group_idx = simplified_scores.max(dim=1)

        # Compute triplet score
        triplet_scores = (max_scores * bbox_scores[pairs[:, 0]] * bbox_scores[pairs[:, 1]]) ** (1.0 / 3.0)

        # -----------------------------------------------------------------
        # 2. Map Back to Real Dataset IDs
        # -----------------------------------------------------------------
        # Look up the real dataset integer IDs for our canonical words
        canon_idx_on = rel_classes.index("on") if "on" in rel_classes else 4
        canon_idx_near = rel_classes.index("beside") if "beside" in rel_classes else 3 # 'near' isn't in vocab

        # Map the argmax result (0 or 1) back to the real dataset IDs (e.g. 4 or 3)
        target_rel_ids = torch.tensor([canon_idx_on, canon_idx_near], dtype=torch.int32, device=pairs.device)
        final_mapped_labels = target_rel_ids[best_group_idx]

        # Create simplified relation tensor
        all_rels = torch.cat([
            pairs.int(),
            final_mapped_labels.unsqueeze(1),  # The correct dataset ID (e.g., 4 or 3)
            triplet_scores.unsqueeze(1).float()
        ], dim=1)

        # -----------------------------------------------------------------
        # 3. Apply Filtering Thresholds
        # -----------------------------------------------------------------
        fids = filtered_bbox_ids.unsqueeze(0)
        subj_ok = (all_rels[:, 0].unsqueeze(1) == fids).any(dim=1)
        obj_ok  = (all_rels[:, 1].unsqueeze(1) == fids).any(dim=1)
        all_rels = all_rels[subj_ok & obj_ok]

        # Filter by confidence threshold
        all_rels = all_rels[all_rels[:, 3] > self.rel_threshold]

        if all_rels.size(0) == 0:
            return node_tensor, torch.empty((0, 4), device=bbox_scores.device)

        # Remap indices from original tensor space to filtered node space
        max_orig_idx = int(filtered_bbox_ids.max().item()) + 1
        idx_map = torch.full((max_orig_idx,), -1, dtype=torch.long, device=filtered_bbox_ids.device)
        idx_map[filtered_bbox_ids] = torch.arange(len(filtered_bbox_ids), device=filtered_bbox_ids.device)

        all_rels[:, 0] = idx_map[all_rels[:, 0].long()]
        all_rels[:, 1] = idx_map[all_rels[:, 1].long()]

        return node_tensor, all_rels