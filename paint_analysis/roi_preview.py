"""Bounded-memory ROI counts and raw-coordinate display samples."""
import numpy as np


ROI_PREVIEW_LIMIT = 20_000
ROI_SCAN_CHUNK = 250_000


def roi_preview(corrected, raw, roi_nm, pixelsize_nm,
                limit=ROI_PREVIEW_LIMIT, chunk_size=ROI_SCAN_CHUNK):
    """Count corrected hits exactly; sample raw counterparts without full indexes.

    Random priorities give a reproducible uniform sample across all chunks.
    Only coordinate columns, chunk-sized masks and bounded samples are allocated.
    Raw lookup scans index chunks to avoid pandas building a full-table hash index.
    """
    if limit < 1 or chunk_size < 1:
        raise ValueError('Preview limit and chunk size must be positive.')
    x0, x1 = sorted(np.asarray(roi_nm[:2]) / pixelsize_nm)
    y0, y1 = sorted(np.asarray(roi_nm[2:]) / pixelsize_nm)
    rng = np.random.default_rng(0)
    priorities = np.empty(0)
    labels = corrected.index[:0]
    count = 0
    for start in range(0, len(corrected), chunk_size):
        stop = start + chunk_size
        x = corrected['x'].iloc[start:stop].to_numpy(copy=False)
        y = corrected['y'].iloc[start:stop].to_numpy(copy=False)
        hits = np.flatnonzero((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
        count += len(hits)
        if not len(hits):
            continue
        keys = np.concatenate((priorities, rng.random(len(hits))))
        candidates = labels.append(corrected.index[start:stop].take(hits))
        keep = np.argpartition(keys, limit-1)[:limit] if len(keys) > limit else np.arange(len(keys))
        priorities, labels = keys[keep], candidates.take(keep)
    if not len(labels):
        return count, np.empty((0, 2))
    # Match by original row label, even when corrected rows are filtered/reordered.
    wanted = set(labels)
    points = []
    remaining = limit
    for start in range(0, len(raw), chunk_size):
        stop = start + chunk_size
        hits = np.flatnonzero(raw.index[start:stop].isin(wanted))[:remaining]
        if not len(hits):
            continue
        x = raw['x'].iloc[start:stop].to_numpy(copy=False)[hits]
        y = raw['y'].iloc[start:stop].to_numpy(copy=False)[hits]
        points.append(np.column_stack((x, y)).astype(float) * pixelsize_nm)
        remaining -= len(hits)
        if remaining == 0:
            break
    return count, np.concatenate(points) if points else np.empty((0, 2))
