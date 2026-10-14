"""Memory-bounded selection and coordinate conversion for classification sources."""
import numpy as np

CHUNK_ROWS = 250_000


def select_source_roi(locs, roi_nm, pixelsize_nm, chunk_size=CHUNK_ROWS):
    """Select once, preserving all columns and original row labels/order.

    A full selection shares existing storage; consumers must treat it as read-only.
    """
    if roi_nm is None:
        return locs.copy(deep=False)
    x0, x1 = sorted(np.asarray(roi_nm[:2]) / pixelsize_nm)
    y0, y1 = sorted(np.asarray(roi_nm[2:]) / pixelsize_nm)
    def hits(start):
        x = locs['x'].iloc[start:start+chunk_size].to_numpy(copy=False)
        y = locs['y'].iloc[start:start+chunk_size].to_numpy(copy=False)
        return np.flatnonzero((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1))
    count = sum(len(hits(start)) for start in range(0, len(locs), chunk_size))
    if count == len(locs):
        return locs.copy(deep=False)
    positions = np.empty(count, dtype=np.intp)
    offset = 0
    for start in range(0, len(locs), chunk_size):
        selected = hits(start)
        positions[offset:offset+len(selected)] = selected + start
        offset += len(selected)
    return locs.take(positions)


def source_points_nm(locs, pixelsize_nm, chunk_size=CHUNK_ROWS):
    points = np.empty((len(locs), 2), dtype=np.float64, order='C')
    for start in range(0, len(locs), chunk_size):
        for axis, column in enumerate(('x', 'y')):
            np.multiply(locs[column].iloc[start:start+chunk_size].to_numpy(copy=False),
                        pixelsize_nm, out=points[start:start+chunk_size, axis])
    return points
