from dataclasses import dataclass, field
from typing import Any, List

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
            
    def observed(self, min_visible_frac=0.5):
        """
        Objects reliable enough to lift this frame.
        
        Occluded objects are skipped, and resume under the same object_id when they reappear.
        """
        return [o for o in self._objects if o.visible_fraction >= min_visible_frac]