# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized YOLO framework for Scene Graph Generation.

Uses Hydra/OmegaConf configuration format.
"""
import torch
from torch import nn
from typing import List, Optional, Dict, Tuple, Union, Any
from omegaconf import DictConfig
from pathlib import Path
import torch.nn.functional as F

from external.react.sgg_benchmark.structures.image_list import to_image_list
from external.react.sgg_benchmark.structures.box_ops import cat_instances

# from ..backbone import build_backbone
from external.react.sgg_benchmark.modeling.roi_heads.roi_heads import build_roi_heads
from external.react.sgg_benchmark.config import load_config_from_file
from external.react.sgg_benchmark.data import get_dataset_statistics
from old_scripts.feature_processor import FeatureProcessor
from src.utils import build_coco_to_react_mapping

class CustomReactModel(nn.Module):
    """
    Main class for Generalized YOLO. Currently supports boxes and relations.
    Combines YOLO backbone with ROI heads for scene graph generation.
    """

    def __init__(self, config: DictConfig, device, ckpt):
        """
        Initialize Generalized YOLO model.
        
        Args:
            cfg: Hydra configuration
        """
        super(CustomReactModel, self).__init__()
        
        self.cfg = load_config_from_file(config)  
        statistics = get_dataset_statistics(self.cfg)
        self.obj_classes = statistics['obj_classes']
        self.rel_classes = statistics['rel_classes']

        self.feature_processor = FeatureProcessor(device, self.obj_classes)      
        
        # Expected channels per level from YAML: [256, 512, 512]
        self.target_channels = self.cfg.model.yolo.out_channels

        self.roi_heads = build_roi_heads(self.cfg, self.target_channels)
        self.load_roi_checkpoint(ckpt)

        self._roi_heads_compiled = False

        # Get config values
        self.predcls = self.cfg.model.roi_relation_head.use_gt_object_label
        self.add_gt = self.cfg.model.roi_relation_head.add_gtbox_to_proposal_in_train
        

        self.export = False
        self.export_obj_thres = 0.05

        self.to(device)

    def compile_for_inference(self):
        """Apply torch.compile to roi_heads for inference latency reduction.

        Must be called AFTER model.eval() and weight loading, never during training.
        torch.compile on the loss function (used only during training) causes graph breaks
        due to data-dependent shapes (nonzero) and CPU tensors (priors buffer).
        """
        if self._roi_heads_compiled:
            return
        try:
            # reduce-overhead: replaces repeated Python kernel dispatch with a CUDA graph,
            # closing the ~9ms gap between CUDA time and wall-clock time.
            # dynamic=True: handles variable N_objects / N_pairs across images.
            self.roi_heads = torch.compile(
                self.roi_heads, mode="reduce-overhead", dynamic=False
            )
            self._roi_heads_compiled = True
            print("roi_heads compiled for inference (reduce-overhead + dynamic).")
        except Exception as e:
            print(f"torch.compile unavailable, running eager: {e}")

    def forward(
        self,
        tracker_info: Dict[str, torch.Tensor],
        extent: Tuple[int, int],
        targets: Optional[List[Dict[str, Any]]] = None,
        logger = None,
        return_attention: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
        """

        """
        # if self.roi_heads.training and targets is None:
        #     raise ValueError("In training mode, targets should be passed")
        
        # images = to_image_list(images)

        # # Determine if we should track gradients for the backbone
        # # We only do this in training and if the backbone is not frozen
        # can_train_backbone = self.training and not self.cfg.model.backbone.freeze
        
        # if can_train_backbone:
        #     # Full forward pass with gradients enabled for backbone
        #     outputs, features = self.backbone(images.tensors, visualize=False, embed=True)
        # else:
        #     # Inference mode or frozen backbone: no gradients for backbone
        #     with torch.no_grad():
        #         outputs, features = self.backbone(images.tensors, visualize=False, embed=True)

        # # Post-processing (NMS, box decoding) should always be non-differentiable
        # with torch.no_grad():
        #     if targets is None:
        #         img_sizes = images.image_sizes # (H, W)
        #     else:
        #         # targets image_size is (W, H), but postprocess expects (H, W)
        #         img_sizes = [(t["image_size"][1], t["image_size"][0]) for t in targets]
        #     proposals = self.backbone.postprocess(outputs, img_sizes)

        proposals, features = self.feature_processor.forward(tracker_info, extent)
        
        ############################
        x, result, detector_losses = self.roi_heads(
            features, proposals, targets, logger, proposals, return_attention=return_attention
        )
        
        return result

        if self.roi_heads.training and (targets is not None) and self.add_gt:
            proposals = self.add_gt_proposals(proposals, targets)

        # to avoid the empty list to be passed into roi_heads during testing and cause error in the pooler
        if not self.training and proposals[0]["boxes"].shape[0] == 0:
            # add empty missing fields
            for p in proposals:
                p["pred_rel_scores"] = torch.tensor([], dtype=torch.float32, device=p["boxes"].device)
                p["pred_rel_labels"] = torch.tensor([], dtype=torch.float32, device=p["boxes"].device)
                p["rel_pair_idxs"] = torch.tensor([], dtype=torch.int64, device=p["boxes"].device)
            return proposals

        if self.roi_heads:
            if self.predcls:  # in predcls mode, we pass the targets as proposals
                for t in targets:
                    if "image_path" in t:
                        del t["image_path"]
                    t["pred_labels"] = t["labels"]
                    t["pred_scores"] = torch.ones_like(t["labels"], dtype=torch.float32)
                x, result, detector_losses = self.roi_heads(features, proposals, targets, logger, targets, return_attention=return_attention)
            else:
                x, result, detector_losses = self.roi_heads(features, proposals, targets, logger, proposals, return_attention=return_attention)
        else:
            # RPN-only models don't have roi_heads
            result = proposals
            detector_losses = {}

        if self.roi_heads.training:
            losses = {}
            losses.update(detector_losses)
            return losses

        if self.export:
            boxes, rels = self.generate_detect_sg(result[0], obj_thres=self.export_obj_thres)
            return [boxes, rels] 
        
        if return_attention:
            for i, res in enumerate(result):
                res["backbone_features"] = [f[i] for f in features]

        return result
        
    @staticmethod
    def _gt_feat_idx(gt_lb: torch.Tensor, lb_sz: int) -> torch.Tensor:
        """
        Compute feat_idx for GT boxes from their center cell on the FPN pyramid.

        Matches the flat-index convention used by the relation head's feat_idx gather:
          [0,      n3):       P3  stride=8,  grid=80×80  (for lb_sz=640)
          [n3,   n3+n4):      P4  stride=16, grid=40×40
          [n3+n4, n3+n4+n5):  P5  stride=32, grid=20×20

        Level assignment is by box side length (≈ standard FPN thresholds):
          side ≤ lb_sz/8  (≤80px)  → P3  (fine detail, small objects)
          side >  lb_sz/4  (>160px) → P5  (coarse, large objects)
          otherwise                 → P4
        """
        s3, s4, s5 = 8, 16, 32
        g3 = lb_sz // s3          # 80
        g4 = lb_sz // s4          # 40
        g5 = lb_sz // s5          # 20
        n3 = g3 * g3              # 6400
        n4 = g4 * g4              # 1600

        cx = ((gt_lb[:, 0] + gt_lb[:, 2]) * 0.5).clamp(0, lb_sz - 1)
        cy = ((gt_lb[:, 1] + gt_lb[:, 3]) * 0.5).clamp(0, lb_sz - 1)

        side = ((gt_lb[:, 2] - gt_lb[:, 0]).clamp(min=0)
                * (gt_lb[:, 3] - gt_lb[:, 1]).clamp(min=0)).sqrt()

        use_p5 = side >  (lb_sz / 4)
        use_p3 = side <= (lb_sz / 8)
        # use_p4: everything in between

        idx3 = (cy / s3).long().clamp(0, g3-1) * g3 + (cx / s3).long().clamp(0, g3-1)
        idx4 = n3 + (cy / s4).long().clamp(0, g4-1) * g4 + (cx / s4).long().clamp(0, g4-1)
        idx5 = n3 + n4 + (cy / s5).long().clamp(0, g5-1) * g5 + (cx / s5).long().clamp(0, g5-1)

        return torch.where(use_p5, idx5, torch.where(use_p3, idx3, idx4))

    def add_gt_proposals(
        self,
        proposals: List[Dict[str, Any]],
        targets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add ground-truth boxes to proposals during training.
        
        Arguments:
            proposals: List of proposed boxes
            targets: List of ground-truth boxes
            
        Returns:
            Concatenated proposals with ground-truth boxes
        """
        new_targets = []
        for proposal, t in zip(proposals, targets):
            # Convert GT boxes from original-image space to letterbox space so
            # lb_boxes is valid for GT proposals (used by feature-map lookups).
            from external.react.sgg_benchmark.structures.box_ops import box_convert
            gt_xyxy = box_convert(t["boxes"], t["mode"], "xyxy").to(proposal["boxes"].device)
            lb_sz   = float(proposal.get("lb_input_size", 640))
            lb_gain = float(proposal.get("lb_gain", 1.0))
            lb_pw   = float(proposal.get("lb_pad_w", 0.0))
            lb_ph   = float(proposal.get("lb_pad_h", 0.0))
            gt_lb   = torch.stack([
                (gt_xyxy[:, 0] * lb_gain + lb_pw).clamp(0, lb_sz),
                (gt_xyxy[:, 1] * lb_gain + lb_ph).clamp(0, lb_sz),
                (gt_xyxy[:, 2] * lb_gain + lb_pw).clamp(0, lb_sz),
                (gt_xyxy[:, 3] * lb_gain + lb_ph).clamp(0, lb_sz),
            ], dim=1)
            device = proposal["boxes"].device
            new_t = {
                "boxes": t["boxes"].to(device),
                "lb_boxes": gt_lb,
                "lb_input_size": int(lb_sz),
                "lb_gain": lb_gain,
                "lb_pad_w": lb_pw,
                "lb_pad_h": lb_ph,
                "labels": t["labels"].to(device),
                "image_size": t["image_size"],
                "mode": t["mode"],
                "pred_labels": t["labels"].to(device) - 1,
                "pred_scores": torch.ones_like(t["labels"], dtype=torch.float32, device=device),
                # Proper center-cell feat_idx: box center → FPN level (area-based) → flat index.
                # Old code was: torch.ones_like(...) = always cell #1 on P3 (wrong for all boxes).
                "feat_idx": self._gt_feat_idx(gt_lb, int(lb_sz)),
            }
            new_targets.append(new_t)

        proposals = [
            cat_instances((proposal, gt_box))
            for proposal, gt_box in zip(proposals, new_targets)
        ]

        return proposals
    
    def generate_detect_sg(
        self,
        predictions: Dict[str, Any],
        obj_thres: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate scene graph from predictions for export.
        
        Args:
            predictions: Dict with predictions
            obj_thres: Object confidence threshold
            
        Returns:
            Tuple of (boxes, relations) tensors:
            - boxes: shape (num_obj, 6) with columns (x1, y1, x2, y2, label, score)
            - rels: shape (num_rel, 5) with columns (obj1, obj2, label, triplet_score, rel_score)
        """
        all_obj_labels = predictions['pred_labels']
        all_obj_scores = predictions['pred_scores']
        all_rel_pairs = predictions['rel_pair_idxs']
        all_rel_prob = predictions['pred_rel_scores']
        
        from external.react.sgg_benchmark.structures.box_ops import box_convert
        all_boxes = box_convert(predictions['boxes'], predictions['mode'], 'xyxy')

        all_rel_scores, all_rel_labels = all_rel_prob[:, 1:].max(-1)
        all_rel_labels = all_rel_labels + 1  # shift back to 1-indexed (skip background class 0)

        # filter objects and relationships (vectorized for ONNX tracing)
        obj_mask = all_obj_scores >= obj_thres
        # Geometric mean: (rel × subj × obj)^(1/3) — keeps the score on the same [0,1] scale
        # as each individual component, so rel_conf=0.1 is a meaningful threshold.
        # Raw product would give e.g. 0.3^3 = 0.027, forcing users to use rel_conf=0.001.
        triplet_score = (all_obj_scores[all_rel_pairs[:, 0]] * all_obj_scores[all_rel_pairs[:, 1]] * all_rel_scores) ** (1.0 / 3.0)
        rel_mask = (all_rel_labels > 0) & (triplet_score > 0)

        # filter boxes
        filter_boxes = all_boxes[obj_mask]
        filter_obj_labels = all_obj_labels[obj_mask]
        filter_obj_scores = all_obj_scores[obj_mask]

        # Map old indices to new indices for relationships
        obj_indices = torch.cumsum(obj_mask.long(), dim=0) - 1
        
        # Only keep relations where both subject and object pass the object threshold
        valid_rel_mask = rel_mask & obj_mask[all_rel_pairs[:, 0]] & obj_mask[all_rel_pairs[:, 1]]
        
        final_rel_pairs = all_rel_pairs[valid_rel_mask]
        final_rel_labels = all_rel_labels[valid_rel_mask]
        final_triplet_scores = triplet_score[valid_rel_mask]
        final_rel_scores = all_rel_scores[valid_rel_mask]
        
        new_subj_indices = obj_indices[final_rel_pairs[:, 0].long()]
        new_obj_indices = obj_indices[final_rel_pairs[:, 1].long()]
        
        # make 2 output tensors
        # boxes: [x1, y1, x2, y2, label, score]
        boxes = torch.cat((
            filter_boxes, 
            filter_obj_labels.unsqueeze(1).float(), 
            filter_obj_scores.unsqueeze(1)
        ), dim=1)
        
        # rels: [subj_idx, obj_idx, label, triplet_score, rel_score]
        rels = torch.cat((
            new_subj_indices.unsqueeze(1).float(),
            new_obj_indices.unsqueeze(1).float(),
            final_rel_labels.unsqueeze(1).float(),
            final_triplet_scores.unsqueeze(1),
            final_rel_scores.unsqueeze(1)
        ), dim=1)

        return boxes, rels

    def load_roi_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_state = checkpoint["model"]

        # Original checkpoint: "roi_heads.relation...."
        # CustomReact.roi_heads state: "relation...."
        saved_roi_state = {
            key.removeprefix("roi_heads."): value
            for key, value in saved_state.items()
            if key.startswith("roi_heads.")
        }

        # current_state = self.roi_heads.state_dict()

        # # Avoid a shape error if your config differs from the checkpoint.
        # compatible_state = {
        #     key: value
        #     for key, value in saved_roi_state.items()
        #     if key in current_state and value.shape == current_state[key].shape
        # }

        missing, unexpected = self.roi_heads.load_state_dict(
            saved_roi_state, strict=False
        )

        print(f"Loaded ROI-head weights from {checkpoint_path}")
        print(f"Missing ROI tensors: {len(missing)}")
        print(f"Unexpected ROI tensors: {len(unexpected)}")