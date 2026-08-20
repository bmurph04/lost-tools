from dataclasses import dataclass, field
from typing import Any, List
import torch

@dataclass
class TrackedObject:
    """
    Identity-owning container for tracked objects.
    """
    object_id: int
    class_id: int
    confidence: float
    points: Any # (n, 2) tensor, replaced each frame by the tracker
    visibles: Any # (n,) boolean tensor
    frames_tracked: int = 0 # Tracker frames this object has survived, drives point decay

    @property
    def num_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def visible_fraction(self) -> float:
        return float(self.visibles.float().mean()) if self.num_points else 0.0

class TrackedObjectSet:
    """
    Append-only registry. Index order is tracker's query order.
    """
    
    def __init__(self):
        self._objects: List[TrackedObject] = []
        self._next_id = 0
        self._floor_warned = False

    def __len__(self): return len(self._objects)
    def __iter__(self): return iter(self._objects)
    def __getitem__(self, i): return self._objects[i]

    # -- views the modules consume (never the container itself) --
    @property
    def points(self):       return [o.points for o in self._objects]
    @property
    def visibles(self):     return [o.visibles for o in self._objects]
    @property
    def point_counts(self): return [o.num_points for o in self._objects]
    @property
    def class_ids(self):    return [o.class_id for o in self._objects]
    @property
    def object_ids(self):   return [o.object_id for o in self._objects]
    @property
    def total_points(self): return sum(self.point_counts)

    def extend(self, class_ids, confidences, points_list, visibles_list):
        """
        Register newly detected objects, minting ids.
        """
        lengths = {'class_ids': len(class_ids), 'confidences': len(confidences),
                   'points_list': len(points_list), 'visibles_list': len(visibles_list)}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                f"Detection fields must be aligned before minting objects, got {lengths}. "
                f"Registering only {min(lengths.values())} objects while the tracker receives "
                f"{len(points_list)} query groups would desynchronise the two.")

        for class_id, confidence, points, visibles in zip(class_ids, confidences, points_list, visibles_list):
            self._objects.append(TrackedObject(self._next_id, int(class_id), float(confidence), points, visibles))
            self._next_id += 1
     
    def update_from_tracker(self, points_list, visibles_list):
        """
        Update from one frame of tracker output.
        Order is positional
        """
        assert len(points_list) == len(self._objects), \
            f"Tracker returned {len(points_list)} point groups for {len(self._objects)} objects"
        
        for object, points, visibles, in zip(self._objects, points_list, visibles_list):
            object.points, object.visibles = points, visibles
            object.frames_tracked += 1
    
    def observed(self, min_visible_frac=0.5):
        """
        Objects reliable enough to lift this frame.
        
        Occluded objects are skipped, and resume under the same object_id when they reappear.
        """
        return [o for o in self._objects if o.visible_fraction >= min_visible_frac]

    def plan_decay(self, config):
        """
        Choose which tracked points are worth their tracker cost.

        Args:
            config (_type_): _description_
            
        Returns a flat boolean keep-mask over every point in tracker order, or
        None when nothing needs trimming. The mask must be handed to both
        `Tracker.prune` and `apply_decay`, or the tracker's buffers and this
        registry go out of sync.
        """
        if not self._objects or not config.enabled:
            return None
        
        # An object that has been tracked for a while holds its identity with
        # fewer points than the detector seeded it with.
        budgets = []
        for object in self._objects:
            settled = config.decay_after > 0 and object.frames_tracked >= config.decay_after
            budget = config.decay_points if settled else object.num_points
            budget = min(object.num_points, max(budget, config.min_points)) # Clamp budget between min_points and current number of points
            budgets.append(budget)
        
        # The per-object budget alone does not bound the total, because objects
        # keep being added. This is the part that actually holds the frame rate.
        total = sum(budgets)
        if config.max_total_points > 0 and total > config.max_total_points:
            # Squeeze the most established objects first: their geometry is
            # already well determed in the scene graph, so they lose the least.
            order = sorted(range(len(self._objects)), key=lambda i: self._objects[i].frames_tracked, reverse=True)
            
            for i in order:
                # Break if we've satisfied getting back under the max allowed total points
                if total <= config.max_total_points:
                    break
                
                # Get the floor on how many points can be removed from this object
                floor = min(config.min_points, self._objects[i].num_points)
                # Subtract the floor from the budget to get excess points, and then take that from the total
                total -= budgets[i] - floor
                
                # min_points wins over the budget: below it the lifter's 2D
                # covariance goes rank-deficient and the object stops merging
                # sensibly, which is worse than the frame rate it would buy. Once
                # every object is already at the floor the only way left to cut cost
                # is to have fewer objects, so say so -- it means duplicate objects,
                # not oversized ones, are now what is driving the cost.
                if total > config.max_total_points and not self._floor_warned:
                    self._floor_warned = True
                    print(f"[objects] {len(self._objects)} objects at the {config.min_points}-point "
                      f"floor need {total} points, over the {config.max_total_points} budget. "
                      f"Point decay cannot bound this further; object count is the cost driver.")
        
        # Return none if we have enough budget for the number of points allocated for each object
        if all(budget >= object.num_points for budget, object in zip(budgets, self._objects)):
            return None
        
        return torch.cat([self._keep_mask(object, budget) for object, budget in zip(self._objects, budgets)])
    
    def apply_decay(self, keep_mask):
        """
        Drop the points `plan_decay` did not keep.

        Args:
            keep_mask (_type_): _description_
        """
        assert keep_mask.shape[0] == self.total_points, \
            f"Decay mask covers {keep_mask.shape[0]} but {self.total_points} are tracked"
        
        offset = 0
        for object in self._objects:
            num_points = object.num_points
            mask = keep_mask[offset:(offset + num_points)].to(object.points.device)
            object.points, object.visibles = object.points[mask], object.visibles[mask]
            
    @staticmethod
    def _keep_mask(object, budget):
        """
        Pick `budget` of an object's points, preferring the points the tracker still
        has lock on and spreading the survivors out. The lifter builds the object's
        2D covariance from these points, so keeping a tight cluster would shrink its
        gaussian and change how it merges.

        Args:
            object (_type_): _description_
            budget (_type_): _description_
        """
        num_points = object.num_points
        # Initialize keep mask
        
        # Keep everything if we have enough budget to cover num_points
        if budget >= num_points:
            keep = torch.ones(num_points, dtype=torch.bool, device=object.points.device)
            return keep
        
        # If code gets here, need to create a keep mask to prune some points
        
        visible_idx = torch.nonzero(object.visibles.bool(), as_tuple=False).squeeze(1) # The points that are visible
        hidden_idx = torch.nonzero(~object.visibles.bool(), as_tuple=False).squeeze(1) # The points that are not visible
        points = object.points.float() # All points
        
        # If the total number of visible elements exceeds budget,
        if visible_idx.numel() >= budget:
            # Get the spread of indices to keep
            visible_indices_spread = _spread_indices(points[visible_idx], budget)
            chosen = visible_idx[visible_indices_spread] # No room for hidden indices
        # Else, the total number of hidden elements is what's exceeding the budget
        else:
            fill = budget - int(visible_idx.numel())
            hidden_indices_spread = _spread_indices(points[hidden_idx], fill)
            chosen = torch.cat([visible_idx, hidden_idx[hidden_indices_spread]])
        
        # Initialize keep as zeros
        keep = torch.zeros(num_points, dtype=torch.bool, device=object.points.device)
        # Only flip chosen indices to true
        keep[chosen] = True
        # Return keep
        return keep
    
def _spread_indices(points, k):
    """
    Farthest-point sample k of n points.
    Deterministic: seeded from the point nearest the centroid so the result does not depend
    on tracker ordering.

    Args:
        points (_type_): _description_
        k (_type_): _description_
    """
    num_points = points.shape[0]
    
    # Return zeros if sampling no n points
    if k <= 0:
        return torch.zeros(0, type=torch.long, device=points.device)
    
    # Return all the points (returning point indices) if we're sampling more points than we have
    if k >= num_points:
        return torch.arange(num_points, device=points.device)
    
    # Get the center point by finding the point closest to the average of all point (coordinate) values
    point_distances_to_mean = ((points - points.mean(dim=0)) ** 2).sum(dim=1)
    center_point = int(torch.argmin(point_distances_to_mean))
    
    # Initialize chosen array and put center_point in it (automatically part of spread)
    chosen = [center_point]
    
    # Get distances to center point
    active_point_distances = ((points - points[center_point]) ** 2).sum(dim=1)
    # For the rest of the points we have left to sample
    for _ in range(k-1):
        # Get the point furthest from the center. Want the biggest spread possible
        farthest_point = int(torch.argmax(active_point_distances))
        chosen.append(farthest_point)
        
        # Get the distances of every point from the point just chosen 
        point_distances_from_current = ((points - points[farthest_point]) ** 2).sum(dim=1)
        
        # Get the distances that sum up to be smaller. I don't know why
        active_point_distances = torch.minimum(active_point_distances, point_distances_from_current)
    
    return torch.tensor(chosen, dtype=torch.long, device=points.device)