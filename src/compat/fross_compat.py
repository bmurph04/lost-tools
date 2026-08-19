"""Fixes for the vendored FROSS GaussianSG merge step.

Beyond the original argmin/non-finite fixes, this addresses two failure modes
seen in the diagnostics:

  * Covariance inflation. _merge_gaussians added a between-group term,
    n1*n2*outer(mean_diff)/total^2, so a node's covariance grew on every merge
    (~12%/frame observed). Since the Hellinger gate is 0.125*d^2/cov, a growing
    cov silently LOOSENS the merge criterion until distinct objects get
    swallowed. We now keep a running weighted average of the per-observation
    covariances instead, which converges to the typical single-observation
    extent and stays stable.

  * No eviction. Nothing was ever deleted, so a node born from one bad early
    observation persisted forever -- the hallucination source and an unbounded
    memory growth path. Nodes that stay unconfirmed past an age limit are now
    dropped.

Environment knobs:
    LOST_TOOLS_SG_DEBUG=1          per-candidate diagnostics
    LOST_TOOLS_SG_MAX_DIST=0.30    metres; hard cap on merge separation
    LOST_TOOLS_SG_PCD_CAP=20000    max points retained per object
    LOST_TOOLS_SG_MIN_STD=0.005    metres; floor on per-axis std
    LOST_TOOLS_SG_MAX_STD=0.0      metres; ceiling on per-axis std (0 = off)
    LOST_TOOLS_SG_EVICT_AGE=150    frames without confirmation before eviction
    LOST_TOOLS_SG_EVICT_MIN_OBS=3  observations that make a node permanent
"""

import os

import numpy as np

_APPLIED = False
_ORIG = {}

_DEBUG = os.environ.get('LOST_TOOLS_SG_DEBUG', '') == '1'
_MAX_DIST = float(os.environ.get('LOST_TOOLS_SG_MAX_DIST', '0.30'))
_PCD_CAP = int(os.environ.get('LOST_TOOLS_SG_PCD_CAP', '20000'))
_MIN_STD = float(os.environ.get('LOST_TOOLS_SG_MIN_STD', '0.005'))
_MAX_STD = float(os.environ.get('LOST_TOOLS_SG_MAX_STD', '0.0'))
_EVICT_AGE = int(os.environ.get('LOST_TOOLS_SG_EVICT_AGE', '150'))
_EVICT_MIN_OBS = int(os.environ.get('LOST_TOOLS_SG_EVICT_MIN_OBS', '3'))


# --------------------------------------------------------------------------
# per-node bookkeeping (weights, observation counts, recency)
# --------------------------------------------------------------------------

def _state(self):
    """Side tables kept parallel to GaussianSG's own slot arrays.

    weight -- true observation-point count, decoupled from the capped pcd
    obs    -- how many observations this node has absorbed
    seen   -- frame index of the last observation absorbed
    frame  -- monotonically increasing counter, bumped once per merge() call
    """
    n = self._valid_mask.shape[0]
    st = getattr(self, '_lt_state', None)
    if st is None:
        st = {'weight': np.zeros(n, np.int64),
              'obs': np.zeros(n, np.int64),
              'seen': np.zeros(n, np.int64),
              'frame': 0}
    elif st['weight'].shape[0] != n:          # _expand_if_needed grew the slots
        for key in ('weight', 'obs', 'seen'):
            grown = np.zeros(n, np.int64)
            m = min(n, st[key].shape[0])
            grown[:m] = st[key][:m]
            st[key] = grown
    self._lt_state = st
    return st


def _weight_of(self, idx):
    """Tracked observation count if known, else the stored pcd length."""
    st = _state(self)
    if st['weight'][idx] > 0:
        return int(st['weight'][idx])
    pcd = self._pcd[idx]
    return 0 if pcd is None else len(pcd)


def _clamp_cov(cov):
    """Bound the covariance spectrum.

    The floor keeps near-collinear point sets from producing singular matrices
    (the source of the non-finite distances). The ceiling, if enabled, is a
    backstop against any remaining growth path.
    """
    if _MIN_STD <= 0 and _MAX_STD <= 0:
        return cov
    vals, vecs = np.linalg.eigh(cov)
    if _MIN_STD > 0:
        vals = np.maximum(vals, _MIN_STD ** 2)
    if _MAX_STD > 0:
        vals = np.minimum(vals, _MAX_STD ** 2)
    return (vecs * vals) @ vecs.T


# --------------------------------------------------------------------------
# distance decomposition (diagnostics only)
# --------------------------------------------------------------------------

def _decompose(self, idx, cand):
    """Split the Bhattacharyya exponent into its position and shape terms."""
    d_vec = self._means[cand] - self._means[idx]
    sep = np.linalg.norm(d_vec, axis=1)
    cov_i = self._covs[idx][None]
    cov_c = self._covs[cand]
    cm = (cov_i + cov_c) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        # pinv, not inv: a singular candidate must not raise inside diagnostics
        maha = 0.125 * np.einsum('ni,nij,nj->n', d_vec, np.linalg.pinv(cm), d_vec)
        shape = 0.5 * np.log(np.linalg.det(cm) /
                             np.sqrt(np.linalg.det(cov_i) * np.linalg.det(cov_c)))
        size_ratio = np.sqrt(np.trace(cov_c, axis1=1, axis2=2) /
                             np.trace(self._covs[idx]))
    return sep, maha, shape, size_ratio


# --------------------------------------------------------------------------
# patched methods
# --------------------------------------------------------------------------

def _add(self, *args, **kwargs):
    """Original add(), plus bookkeeping init for the new slots."""
    idxs = _ORIG['add'](self, *args, **kwargs)
    st = _state(self)
    st['obs'][idxs] = 1
    st['seen'][idxs] = st['frame']
    st['weight'][idxs] = 0          # 0 => fall back to len(pcd)
    return idxs


def _merge_gaussians(self, idx1, idx2):
    """Merge idx1 into idx2, keeping a RUNNING AVERAGE covariance."""
    mean1, cov1 = self._means[idx1], self._covs[idx1]
    mean2, cov2 = self._means[idx2], self._covs[idx2]

    num_pnts1 = _weight_of(self, idx1)
    num_pnts2 = _weight_of(self, idx2)
    if num_pnts1 == 0 and num_pnts2 == 0:      # too small to have a pcd
        num_pnts1 = num_pnts2 = 1
    total_pnts = num_pnts1 + num_pnts2

    # Mean: unchanged, weighted average of the observations.
    self._means[idx2] = (num_pnts1 * mean1 + num_pnts2 * mean2) / total_pnts

    # Covariance: weighted average ONLY. The upstream between-group term
    #   + num_pnts1 * num_pnts2 * np.outer(mean_diff, mean_diff) / total_pnts**2
    # is deliberately dropped -- it made cov grow with every merge and
    # progressively relax the merge gate.
    self._covs[idx2] = _clamp_cov(
        (num_pnts1 * cov1 + num_pnts2 * cov2) / total_pnts)

    self._rels[idx2, self._valid_mask] += self._rels[idx1, self._valid_mask]
    self._rels[self._valid_mask, idx2] += self._rels[self._valid_mask, idx1]

    # Bounded concatenation -- unbounded growth here was the memory leak.
    pcd1 = self._pcd[idx1] if self._pcd[idx1] is not None else np.empty((0, 3))
    pcd2 = self._pcd[idx2] if self._pcd[idx2] is not None else np.empty((0, 3))
    merged = np.concatenate((pcd1, pcd2))
    if _PCD_CAP > 0 and len(merged) > _PCD_CAP:
        merged = merged[np.random.choice(len(merged), _PCD_CAP, replace=False)]
    self._pcd[idx2] = merged

    st = _state(self)
    st['weight'][idx2] = total_pnts            # true weight, ignores the cap
    st['obs'][idx2] = max(1, st['obs'][idx2]) + max(1, st['obs'][idx1])
    st['seen'][idx2] = st['frame']             # idx2 was just confirmed
    st['weight'][idx1] = 0
    st['obs'][idx1] = 0
    st['seen'][idx1] = 0

    self._classes[idx1] = -9999999
    self._means[idx1] = np.nan
    self._covs[idx1] = np.nan
    self._rels[idx1, self._valid_mask] = 0
    self._rels[self._valid_mask, idx1] = 0
    self._pcd[idx1] = None
    self._valid_mask[idx1] = False


def _drop(self, idx):
    """Retire a node entirely, freeing its slot for reuse by add()."""
    st = _state(self)
    self._classes[idx] = -9999999
    self._means[idx] = np.nan
    self._covs[idx] = np.nan
    self._rels[idx, self._valid_mask] = 0
    self._rels[self._valid_mask, idx] = 0
    self._pcd[idx] = None
    self._valid_mask[idx] = False
    st['weight'][idx] = 0
    st['obs'][idx] = 0
    st['seen'][idx] = 0


def _evict(self):
    """Drop nodes that were never confirmed by later observations.

    A node crossing _EVICT_MIN_OBS is treated as real and kept forever, even if
    it later goes out of view. Only nodes that stayed unconfirmed past
    _EVICT_AGE frames are removed -- i.e. one-off phantoms.
    """
    if _EVICT_AGE <= 0:
        return 0
    st = _state(self)
    age = st['frame'] - st['seen']
    stale = self._valid_mask & (age > _EVICT_AGE) & (st['obs'] < _EVICT_MIN_OBS)
    victims = np.nonzero(stale)[0]
    for idx in victims:
        if _DEBUG:
            print(f'[sg] EVICT idx={idx} class={self._classes[idx]} '
                  f'obs={st["obs"][idx]} age={age[idx]}')
        _drop(self, idx)
    return len(victims)


def _merge(self, update_idx):
    st = _state(self)
    st['frame'] += 1                            # one merge() call == one frame

    update_idx = np.asarray(update_idx).tolist()
    while update_idx:
        idx = update_idx.pop()

        # May already have been absorbed earlier in this same call.
        if not self._valid_mask[idx]:
            continue

        # Keep absorbing into the survivor until nothing else qualifies.
        while True:
            cand = (self._classes == self._classes[idx]) & self._valid_mask
            cand[idx] = False
            cand = np.nonzero(cand)[0]
            if cand.size == 0:
                if _DEBUG:
                    print(f'[sg] idx={idx} class={self._classes[idx]}: '
                          f'no same-class candidates')
                break

            with np.errstate(divide='ignore', invalid='ignore'):
                dist = self._batched_hellinger_distance(
                    self._means[idx], self._covs[idx],
                    self._means[cand], self._covs[cand])
            dist = np.where(np.isfinite(dist), dist, np.inf)

            sep = np.linalg.norm(self._means[cand] - self._means[idx], axis=1)
            gated = np.where(sep <= _MAX_DIST, dist, np.inf)

            best = int(np.argmin(gated))
            ok = gated[best] < self.merge_threshold

            if _DEBUG:
                s, maha, shape, ratio = _decompose(self, idx, cand)
                b = int(np.argmin(dist))
                print(f'[sg] idx={idx} class={self._classes[idx]} '
                      f'cand={cand.size} obs={st["obs"][idx]} '
                      f'best={dist[b]:.3f} sep={s[b] * 100:.1f}cm '
                      f'maha={maha[b]:.3f} shape={shape[b]:.3f} '
                      f'size_ratio={ratio[b]:.2f} '
                      f'-> {"MERGE" if ok else "no merge"}'
                      f'{"" if sep[b] <= _MAX_DIST else "  [GATED: too far]"}')

            if not ok:
                break

            target = cand[best]
            _merge_gaussians(self, idx, target)
            idx = target        # continue absorbing from the survivor

    _evict(self)


def apply():
    """Idempotently install the corrected merge. Returns True on success."""
    global _APPLIED
    if _APPLIED:
        return True
    try:
        from external.FROSS.Merging.utils import GaussianSG
    except ImportError:
        return False
    _ORIG['add'] = GaussianSG.add                # wrapped, not replaced
    GaussianSG.add = _add
    GaussianSG.merge = _merge
    GaussianSG._merge_gaussians = _merge_gaussians
    _APPLIED = True
    return True