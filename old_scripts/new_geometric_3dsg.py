"""Geometric 3D scene-graph relations for :class:`PointLifter` output."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import torch


ObjectInstance: TypeAlias = int | str
SceneGraphTriplet: TypeAlias = tuple[ObjectInstance, str, ObjectInstance]


def _as_numpy(value: np.ndarray | torch.Tensor | Sequence[object], name: str) -> np.ndarray:
    """Detach tensors and return a floating point NumPy array."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a numeric array.") from exc


def _standard_deviations(covs_3d: np.ndarray) -> np.ndarray:
    """Return per-axis Gaussian standard deviations, robust to tiny numerical errors."""
    diagonal = np.diagonal(covs_3d, axis1=1, axis2=2)
    return np.sqrt(np.maximum(diagonal, 0.0))


def build_3d_scene_graph(
    means_3d: np.ndarray | torch.Tensor | Sequence[object],
    covs_3d: np.ndarray | torch.Tensor | Sequence[object],
    valid_object_instances: Sequence[ObjectInstance] | np.ndarray | torch.Tensor,
    *,
    near_distance: float = 0.10,
    contact_distance: float = 0.05,
    lateral_tolerance: float = 0.10,
    extent_std_scale: float = 2.0,
    up_axis: int = 1,
    up_direction: float = -1.0,
) -> list[SceneGraphTriplet]:
    """Build ``near`` and ``on`` relations from PointLifter's 3D Gaussians.

    ``means_3d[i]`` and ``covs_3d[i]`` describe the same object whose stable
    graph-node ID is ``valid_object_instances[i]``.  The default coordinate
    convention is PointLifter's camera convention: X is right, Y is down, and
    Z is forward.  Therefore smaller Y is physically higher.  For a world
    coordinate system whose positive Y points upward, pass ``up_direction=1``.

    The diagonal covariance entries define a conservative, axis-aligned
    ``extent_std_scale``-sigma ellipsoid.  Two objects are ``on`` when the
    first is above the second, their horizontal ellipsoids overlap (within
    ``lateral_tolerance``), and their vertical surfaces are within
    ``contact_distance``.  ``near`` is symmetric, but is omitted entirely for
    a pair when either directed ``on`` relation exists.

    Args:
        means_3d: PointLifter means, shape ``(M, 3)``.
        covs_3d: PointLifter covariances, shape ``(M, 3, 3)``.
        valid_object_instances: Original object indices aligned with the means.
        near_distance: Maximum centroid distance for ``near``.
        contact_distance: Maximum vertical surface gap for ``on``.
        lateral_tolerance: Extra horizontal clearance allowed for ``on``.
        extent_std_scale: Number of standard deviations used as object extent.
        up_axis: Vertical coordinate axis (0, 1, or 2).
        up_direction: ``-1`` when decreasing coordinates mean up; ``1`` when
            increasing coordinates mean up.

    Returns:
        Directed triplets ``(subject_instance, relation, object_instance)``.
        ``near`` appears in both directions; ``on`` appears only as
        ``(upper_object, "on", supporting_object)``.
    """
    means = _as_numpy(means_3d, "means_3d")
    covs = _as_numpy(covs_3d, "covs_3d")
    if torch.is_tensor(valid_object_instances):
        valid_object_instances = valid_object_instances.detach().cpu().numpy()
    instances = list(valid_object_instances)

    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"means_3d must have shape (M, 3), got {means.shape}.")
    if covs.shape != (len(means), 3, 3):
        raise ValueError(f"covs_3d must have shape ({len(means)}, 3, 3), got {covs.shape}.")
    if len(instances) != len(means):
        raise ValueError(
            "valid_object_instances must contain one entry per mean; "
            f"got {len(instances)} entries for {len(means)} means."
        )
    if up_axis not in (0, 1, 2):
        raise ValueError(f"up_axis must be 0, 1, or 2, got {up_axis}.")
    if up_direction not in (-1.0, 1.0):
        raise ValueError("up_direction must be -1.0 or 1.0.")
    if min(near_distance, contact_distance, lateral_tolerance, extent_std_scale) < 0:
        raise ValueError("Distance thresholds and extent_std_scale must be non-negative.")

    extents = extent_std_scale * _standard_deviations(covs)
    horizontal_axes = [axis for axis in range(3) if axis != up_axis]
    graph: list[SceneGraphTriplet] = []

    # Compare each unordered pair once. This avoids duplicate relations while
    # still allowing us to emit the two directed edges for the symmetric near
    # predicate.
    for first in range(len(means)):
        if not (np.isfinite(means[first]).all() and np.isfinite(extents[first]).all()):
            continue
        for second in range(first + 1, len(means)):
            if not (np.isfinite(means[second]).all() and np.isfinite(extents[second]).all()):
                continue

            a, b = means[first], means[second]
            a_extent, b_extent = extents[first], extents[second]
            horizontal_distance = float(np.linalg.norm(a[horizontal_axes] - b[horizontal_axes]))
            horizontal_reach = float(
                np.linalg.norm(a_extent[horizontal_axes] + b_extent[horizontal_axes])
            )

            # Sign-adjusting makes "height" increase upward irrespective of
            # whether the source coordinates are camera-Y-down or world-Y-up.
            a_height = up_direction * a[up_axis]
            b_height = up_direction * b[up_axis]
            vertical_gap = abs(a_height - b_height) - (a_extent[up_axis] + b_extent[up_axis])
            horizontally_aligned = horizontal_distance <= lateral_tolerance + horizontal_reach
            surfaces_in_contact = abs(vertical_gap) <= contact_distance

            on_relation: tuple[int, int] | None = None
            if horizontally_aligned and surfaces_in_contact and a_height != b_height:
                on_relation = (first, second) if a_height > b_height else (second, first)

            if on_relation is not None:
                upper, lower = on_relation
                graph.append((instances[upper], "on", instances[lower]))
                # ``near`` is intentionally suppressed for this whole pair.
                continue

            if float(np.linalg.norm(a - b)) <= near_distance:
                graph.append((instances[first], "near", instances[second]))
                graph.append((instances[second], "near", instances[first]))

    return graph
