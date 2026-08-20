from dataclasses import dataclass, fields

@dataclass(frozen=True)
class MergeConfig:
    merge_threshold: float = 0.7
    max_merge_dist: float = 0.30
    broken_track_dist: float = 0.50
    pcd_cap: int = 20000
    evict_age: int = 150
    evict_min_obs: int = 3
    global_threshold: float = 0.85
    global_max_dist: float = 0.60
    global_disjoint: bool = True

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})
    
@dataclass(frozen=True)
class PointDecayConfig:
    """
    Bounds the tracker's working set.

    Tracker cost scales with the total number of tracked points, and objects are
    only ever added, so without a bound the frame rate decays with session
    length. Points are dropped from *within* objects rather than retiring whole
    objects: object_id survives, so the scene graph's identity association
    (`_track_node`) survives too, and a trimmed object does not come back as a
    duplicate node.
    """
    decay_after: int = 30        # tracker frames an object must survive before it is trimmed (0 disables)
    decay_points: int = 12       # points a settled object keeps
    min_points: int = 6          # hard floor; below this the lifter's 2D covariance degenerates
    max_total_points: int = 512  # global budget across all objects (0 disables)

    @property
    def enabled(self) -> bool:
        return self.decay_after > 0 or self.max_total_points > 0

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})

@dataclass(frozen=True)
class ReassociationConfig:
    """
    Re-attach a detection to the scene-graph node it already belongs to, instead
    of minting a fresh object_id.

    A fresh object_id has no entry in GaussianSGMerge._track_node, so it starts a
    new node, and from then on only the geometric pass can consolidate it -- which
    the disjoint gate refuses whenever both nodes are observed on the same frame,
    exactly the case a broken track produces. Reusing the id keeps the identity
    association and repairs the track instead.
    """
    enabled: bool = False
    max_depth_diff: float = 0.30   # metres; MUST stay below MergeConfig.broken_track_dist
    bbox_margin: float = 8.0       # px slack when testing whether a node projects into a box
    max_visible_frac: float = 0.5  # skip nodes whose track is still healthy

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d[f.name] for f in fields(cls) if f.name in d})
