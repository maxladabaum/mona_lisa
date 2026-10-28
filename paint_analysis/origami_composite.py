"""Measured image-side reconstruction from classified, rigidly aligned origami."""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
import threading
import time
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


@contextmanager
def _progress_heartbeat(callback, interval=1.0):
    """Keep opaque NumPy/SciPy operations visible without inventing percentages.

    Callbacks run on the worker and heartbeat threads; GUI callers must enqueue
    messages rather than touch widgets. Join before returning, including errors,
    so a stale heartbeat can never overwrite completion or failure.
    """
    if callback is None:
        yield None
        return
    stopped = threading.Event()
    lock = threading.Lock()
    started = time.monotonic()
    stage = ['Preparing composite…', started]

    def report(message):
        with lock:
            stage[:] = [message, time.monotonic()]
            callback(message)

    def heartbeat():
        while not stopped.wait(interval):
            with lock:
                now = time.monotonic()
                callback(f'{stage[0]} · working, {now - stage[1]:.0f}s since last update '
                         f'({now - started:.0f}s elapsed)')

    report(stage[0])
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        yield report
    finally:
        stopped.set()
        thread.join()


@dataclass(frozen=True)
class CompositeSettings:
    # Image rectangle in aligned, centered, y-up coordinates. Output is y-down.
    left_nm: float
    top_nm: float
    width_nm: float
    height_nm: float
    pixel_nm: float = 1.0
    sigma_nm: float = 1.5
    average_species: bool = True
    rotation_deg: int = 0
    exclusion_radius_nm: float = 2.0  # Gaussian exclusion sigma; retain serialized key.
    mirror_x: bool = False
    mirror_y: bool = False
    global_sigma_nm: float = 0.0
    blend_width_nm: float = 0.0  # Zero follows the normal rectangle dimension.
    blend_height_nm: float = 0.0
    normalize_overlap: bool = False

    @property
    def blend_size(self):
        normal = np.array([self.width_nm, self.height_nm])
        requested = np.array([self.blend_width_nm or self.width_nm,
                              self.blend_height_nm or self.height_nm])
        # Identical displayed values may differ after geometry arithmetic/JSON.
        return np.where(np.isclose(requested, normal, rtol=0, atol=1e-9), normal, requested)

    @property
    def blend_margin(self):
        return (self.blend_size - [self.width_nm, self.height_nm]) / 2

    def validate(self):
        values = [self.left_nm, self.top_nm, self.width_nm, self.height_nm,
                  self.pixel_nm, self.sigma_nm, self.global_sigma_nm, self.exclusion_radius_nm,
                  self.blend_width_nm, self.blend_height_nm]
        if not np.all(np.isfinite(values)):
            raise ValueError("Image geometry and rendering settings must be finite.")
        if min(self.width_nm, self.height_nm, self.pixel_nm) <= 0:
            raise ValueError("Image width, height and pixel size must be positive.")
        if min(self.sigma_nm, self.global_sigma_nm, self.exclusion_radius_nm) < 0:
            raise ValueError("Smoothing and exclusion radius cannot be negative.")
        issues = []
        for axis, requested, normal, effective in zip(
                ('width', 'height'), (self.blend_width_nm, self.blend_height_nm),
                (self.width_nm, self.height_nm), self.blend_size):
            if requested < 0 or effective < normal:
                issues.append(f"Overlap blend {axis} is {requested:.12g} nm; normal {axis} is {normal:.12g} nm. "
                              f"Enter 0 to use the normal {axis}, or at least {normal:.12g} nm. "
                              "Enter the total size, not the extra margin.")
        if issues:
            raise ValueError("\n\n".join(issues))
        if self.rotation_deg not in (0, 90, 180, 270):
            raise ValueError("Display rotation must be 0, 90, 180 or 270 degrees.")


def _orient_composite_raster(array, rotation_deg, mirror_x=False, mirror_y=False):
    oriented = np.rot90(array, -int(rotation_deg) // 90)
    if mirror_x:
        oriented = np.fliplr(oriented)
    if mirror_y:
        oriented = np.flipud(oriented)
    return np.ascontiguousarray(oriented)


def reorient_composite(result, rotation_deg, mirror_x=False, mirror_y=False):
    """Losslessly update density, coverage and RGB without re-extracting origami."""
    if rotation_deg not in (0, 90, 180, 270):
        raise ValueError("Display rotation must be 0, 90, 180 or 270 degrees.")
    previous = result['report']['settings']
    changes = dict(rotation_deg=int(rotation_deg), mirror_x=bool(mirror_x), mirror_y=bool(mirror_y))
    updated = dict(result)
    for key in ('density', 'coverage', 'rgb'):
        array = result[key]
        # Undo the previous display operations in reverse order first.
        if previous.get('mirror_y', False):
            array = np.flipud(array)
        if previous.get('mirror_x', False):
            array = np.fliplr(array)
        canonical = np.rot90(array, int(previous.get('rotation_deg', 0)) // 90)
        updated[key] = _orient_composite_raster(canonical, **changes)
    report = dict(result['report'], settings={**previous, **changes})
    if 'mapping_configuration' in report:
        config = report['mapping_configuration']
        report['mapping_configuration'] = dict(config, settings={**config['settings'], **changes})
    updated['report'] = report
    return updated


def encoding_codebook(document):
    geometry = document.get("geometry", document)
    entries = {}
    for row in geometry.get("codebook", []):
        code = ''.join(str(int(bit)) for bit in row['bits'])
        if not code or set(code) - {'0', '1'} or code in entries:
            raise ValueError("Encoding settings contain an invalid or duplicate barcode.")
        color = {'R1 R': 'Red', 'R1 G': 'Green', 'R1 B': 'Blue'}.get(row['color'], row['color'])
        mapping = dict(code=code, color=color, x_shift_nm=float(row['x_shift_nm']),
                       y_shift_nm=float(row['y_shift_nm']))
        validate_mapping(mapping)
        entries[code] = mapping
    if not entries:
        raise ValueError("No barcode codebook found in the image encoding settings.")
    return geometry, entries


def uniform_fill_fraction(mapping):
    """Optional constant replacement for an excluded template, in display units."""
    if mapping.get('include', True):
        return None
    value = mapping.get('fill_percent')
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('Excluded fill must be blank or a percentage from 0 to 100.') from None
    if not np.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('Excluded fill must be blank or a percentage from 0 to 100.')
    return value / 100


def validate_mapping(mapping):
    if mapping.get('color') not in ('Red', 'Green', 'Blue'):
        raise ValueError("Assign Red, Green or Blue to each included template.")
    try:
        shifts_valid = np.all(np.isfinite([float(mapping['x_shift_nm']), float(mapping['y_shift_nm'])]))
    except (KeyError, TypeError, ValueError):
        shifts_valid = False
    if not shifts_valid:
        raise ValueError("XY shifts must be finite nanometre values.")

    try:
        rotation = float(mapping.get('rotation_deg', 0))
    except (TypeError, ValueError):
        rotation = -1
    if rotation not in (0, 90, 180, 270):
        raise ValueError("Section rotation must be 0, 90, 180 or 270 degrees clockwise.")

    try:
        brightness = float(mapping.get('brightness', 1.0))
    except (TypeError, ValueError):
        brightness = -1
    if not np.isfinite(brightness) or brightness < 0:
        raise ValueError("Brightness must be a finite number greater than or equal to zero (1 = unchanged).")


def orient_section(points, width, height, mapping, *, clip=True):
    """Rotate clockwise, then mirror in rotated axes; rebase to the upper-left."""
    points = np.asarray(points, dtype=float).copy()
    rotation = int(float(mapping.get('rotation_deg', 0)))
    x, y = points[:, 0].copy(), points[:, 1].copy()
    if rotation == 90:
        points[:, 0], points[:, 1] = height - y, x
        width, height = height, width
    elif rotation == 180:
        points[:, 0], points[:, 1] = width - x, height - y
    elif rotation == 270:
        points[:, 0], points[:, 1] = y, width - x
        width, height = height, width
    if mapping.get('mirror_x', False):
        points[:, 0] = width - points[:, 0]
    if mapping.get('mirror_y', False):
        points[:, 1] = height - points[:, 1]
    # Reflections can put a boundary point at the formerly excluded far edge.
    # Keep it within its own section instead of spilling into an adjacent one.
    if clip:
        points = np.clip(points, [0, 0], np.nextafter([width, height], [0, 0]))
    return points, (width, height)


def reconstruction_mapping_issues(results, mappings):
    """Name every included template that needs attention before a build."""
    issues = []
    codes = {}
    for name in results:
        mapping = mappings.get(name)
        if mapping is not None and not mapping.get('include', True):
            try:
                if uniform_fill_fraction(mapping) is None:
                    continue
            except ValueError as exc:
                issues.append((name, str(exc)))
                continue
        if mapping is None:
            issues.append((name, 'Assign a color and XY shift.'))
            continue
        try:
            validate_mapping(mapping)
        except (KeyError, TypeError, ValueError) as exc:
            message = ('Choose Red, Green or Blue, or uncheck Include.'
                       if mapping.get('color') not in ('Red', 'Green', 'Blue')
                       else str(exc))
            issues.append((name, message))
        code = mapping.get('code', '').strip()
        if code:
            codes.setdefault(code, []).append(name)
    for code, names in codes.items():
        if len(names) > 1:
            issues.extend((name, f'duplicate code {code!r}; also assigned to ' + ', '.join(n for n in names if n != name)) for name in names)
    return issues


def pose_basis(corners):
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("Missing or invalid saved alignment footprint. Rerun classification.")
    basis = np.column_stack((corners[1] - corners[0], corners[3] - corners[0]))
    lengths = np.linalg.norm(basis, axis=0)
    if np.any(lengths <= 0):
        raise ValueError("Degenerate alignment footprint.")
    basis /= lengths
    if not np.allclose(basis.T @ basis, np.eye(2), atol=1e-5):
        raise ValueError("Saved alignment footprint is not rectangular.")
    return corners.mean(axis=0), basis


def section_blend_weights(points, size, margin):
    """Flat core with Gaussian tails; the outer rectangle lies at 3 sigma.

    Points are relative to the normal upper-left, in y-down coordinates.
    Axis falloffs multiply at corners. Zero margin preserves a hard edge.
    """
    points = np.asarray(points, dtype=float)
    size, margin = np.asarray(size), np.asarray(margin)
    distance = np.maximum(np.maximum(-points, points - size), 0)
    weights = np.ones(points.shape[:-1])
    for axis in (0, 1):
        if margin[axis] > 0:
            weights *= np.exp(-0.5 * (3 * distance[..., axis] / margin[axis]) ** 2)
            weights *= distance[..., axis] <= margin[axis]
        else:
            weights *= (points[..., axis] >= 0) & (points[..., axis] < size[axis])
    return weights


def marker_exclusion_weights(points, excluded_sites, sigma_nm):
    """Transmission 1 - Gaussian(distance to nearest marker), in aligned nm.

    Zero sigma disables exclusion. Nearest-site weighting avoids multiplying
    suppression where marker neighborhoods overlap.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    sites = np.asarray(excluded_sites, dtype=float).reshape(-1, 2)
    if not np.isfinite(sigma_nm) or sigma_nm < 0:
        raise ValueError('Marker exclusion sigma must be finite and nonnegative.')
    if not len(points) or not len(sites) or sigma_nm == 0:
        return np.ones(len(points))
    distances = cKDTree(sites).query(points)[0]
    with np.errstate(over='ignore'):
        return -np.expm1(-0.5 * (distances / sigma_nm) ** 2)


def image_region_weights(points, excluded_sites, settings):
    """Blend weights in aligned coordinates; marker suppression only in tails."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    local = np.column_stack((points[:, 0] - settings.left_nm, settings.top_nm - points[:, 1]))
    size = np.array([settings.width_nm, settings.height_nm])
    weights = section_blend_weights(local, size, settings.blend_margin)
    outside_normal = np.any((local < 0) | (local > size), axis=1)
    tail = outside_normal & (weights > 0)
    weights[tail] *= marker_exclusion_weights(points[tail], excluded_sites, settings.exclusion_radius_nm)
    return weights


def extract_image_points(source, tree, corners, settings, excluded_sites=(), *, return_weights=False):
    """Fetch from the source, including signal outside the classification crop."""
    center, basis = pose_basis(corners)
    local_center = np.array([settings.left_nm + settings.width_nm / 2,
                             settings.top_nm - settings.height_nm / 2])
    world_center = center + local_center @ basis.T
    ids = np.asarray(tree.query_ball_point(world_center, np.linalg.norm(settings.blend_size) / 2, return_sorted=True), dtype=int)
    points = (source[ids] - center) @ basis
    mx, my = settings.blend_margin
    inside = ((points[:, 0] >= settings.left_nm - mx) & (points[:, 0] < settings.left_nm + settings.width_nm + mx)
              & (points[:, 1] <= settings.top_nm + my) & (points[:, 1] > settings.top_nm - settings.height_nm - my))
    points, ids = points[inside], ids[inside]
    weights = image_region_weights(points, excluded_sites, settings)
    keep = weights > 0
    points, ids, weights = points[keep], ids[keep], weights[keep]
    points = np.column_stack((points[:, 0] - settings.left_nm, settings.top_nm - points[:, 1]))
    return (ids, points, weights) if return_weights else (ids, points)


class RegionPreviewCancelled(RuntimeError):
    """The user closed the window or cancelled a region preview."""


def image_region_preview_points(source, corners, grid, settings, progress=None,
                                cancelled=None, max_points=20_000, chunk_size=100_000):
    """Scan a single example region without indexing/copying the acquisition.

    Keep a bounded, reproducible random sample for display only. Reconstruction
    continues to use every localization. Return the true nearby count as well.
    """
    if max_points <= 0 or chunk_size <= 0:
        raise ValueError('Preview point and chunk limits must be positive.')
    settings.validate()
    center, basis = pose_basis(corners)
    grid = np.asarray(grid, dtype=float).reshape(-1, 2)
    radius = max(float(np.linalg.norm(grid, axis=1).max()) if len(grid) else 0.,
                 np.hypot(settings.left_nm, settings.top_nm) + np.linalg.norm(settings.blend_size))
    source = np.asarray(source)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError('Load source localization coordinates before previewing.')
    points = np.empty((0, 2), dtype=float)
    priorities = np.empty(0)
    rng = np.random.default_rng(0)
    count = 0
    if progress:
        progress(f'Scanning image-region preview: 0/{len(source):,} source localizations')
    for start in range(0, len(source), chunk_size):
        if cancelled is not None and cancelled():
            raise RegionPreviewCancelled()
        stop = min(start + chunk_size, len(source))
        block = np.asarray(source[start:stop], dtype=float)
        if not np.all(np.isfinite(block)):
            raise ValueError('Source localization coordinates must be finite.')
        delta = block - center
        nearby = delta[np.einsum('ij,ij->i', delta, delta) <= radius ** 2]
        count += len(nearby)
        if len(nearby):
            # Lowest random priorities yield a uniform bounded sample, without
            # retaining all matches or favoring the earliest source chunks.
            points = np.concatenate((points, nearby @ basis))
            priorities = np.concatenate((priorities, rng.random(len(nearby))))
            if len(points) > max_points:
                keep = np.argpartition(priorities, max_points - 1)[:max_points]
                points, priorities = points[keep], priorities[keep]
        if progress:
            progress(f'Scanning image-region preview: {stop:,}/{len(source):,} '
                     f'({stop / len(source):.0%}); {count:,} nearby localizations')
    if cancelled is not None and cancelled():
        raise RegionPreviewCancelled()
    return points, count


def _reconstruction_source(source, results, names, settings, progress=None, chunk_size=100_000):
    """Screen in bounded chunks using a small tree of accepted region centers.

    Keep the union of circumscribed circles, including duplicate localizations.
    Exact rectangle/exclusion tests still happen in extract_image_points. This
    avoids building a tree (and a float64 copy) of the entire acquisition.
    """
    local_center = np.array([settings.left_nm + settings.width_nm / 2,
                             settings.top_nm - settings.height_nm / 2])
    centers = []
    total = sum(np.count_nonzero(results[n]['picks'].accepted_mask) for n in names)
    if progress:
        progress(f'Preparing extraction regions: 0/{total}')
    for name in names:
        picks = results[name]['picks']
        for index in np.flatnonzero(picks.accepted_mask):
            center, basis = pose_basis(picks.rectangle_corners_nm[index])
            centers.append(center + local_center @ basis.T)
            if progress and (len(centers) % 100 == 0 or len(centers) == total):
                progress(f'Preparing extraction regions: {len(centers)}/{total}')
    centers = np.asarray(centers)
    if progress:
        progress(f'Indexing {len(centers):,} extraction-region centers…')
    tree = cKDTree(centers)
    radius = np.linalg.norm(settings.blend_size) / 2
    # Inflate the prefilter slightly to preserve boundary points despite roundoff.
    margin = 32 * np.finfo(float).eps * max(1., radius, np.abs(centers).max())
    radius += margin
    lower, upper = centers.min(axis=0) - radius, centers.max(axis=0) + radius
    chunks = []
    retained = 0
    if progress:
        progress(f'Scanning source coordinates: 0/{len(source):,} (0%)')
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        block = np.asarray(source[start:stop], dtype=float)
        if not np.all(np.isfinite(block)):
            raise ValueError("Load finite source localization coordinates before reconstruction.")
        nearby = block[np.all((block >= lower) & (block <= upper), axis=1)]
        if len(nearby):
            distances, _ = tree.query(nearby, distance_upper_bound=radius)
            selected = nearby[np.isfinite(distances)]
            if len(selected):
                chunks.append(selected)
                retained += len(selected)
        if progress:
            progress(f'Scanning source coordinates: {stop:,}/{len(source):,} '
                     f'({stop / len(source):.0%}); {retained:,} nearby localizations')
    if progress:
        progress(f'Assembling {retained:,} nearby source coordinates…')
    return np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=float)


def reconstruct_composite(source, results, mappings, settings, exclusions=None, progress=None):
    with _progress_heartbeat(progress) as report:
        result = _reconstruct_composite(source, results, mappings, settings, exclusions, report)
    if progress:
        progress('Composite reconstruction complete.')
    return result


def _reconstruct_composite(source, results, mappings, settings, exclusions, progress):
    if progress:
        progress('Validating settings, mappings, and output dimensions…')
    settings.validate()
    source = np.asarray(source)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("Load finite source localization coordinates before reconstruction.")
    if not results:
        raise ValueError("Run Step 4 to classify barcode templates first.")
    names = [name for name in results if mappings.get(name, {}).get('include', True)]
    if not names:
        raise ValueError("Include at least one template.")
    issues = reconstruction_mapping_issues(results, mappings)
    if issues:
        raise ValueError("Check barcode mappings:\n" + "\n".join(f'Template "{name}": {message}' for name, message in issues))
    filled_names = [name for name in results if uniform_fill_fraction(mappings.get(name, {})) is not None]
    layout_names = names + filled_names
    layout_shifts = np.array([[float(mappings[n]['x_shift_nm']), float(mappings[n]['y_shift_nm'])] for n in layout_names])
    layout_sizes = np.array([orient_section(np.empty((0, 2)), settings.width_nm, settings.height_nm, mappings[n])[1] for n in layout_names])
    shifts, sizes = layout_shifts[:len(names)], layout_sizes[:len(names)]
    pad = 4 * (settings.sigma_nm + settings.global_sigma_nm)
    blending = bool(np.any(settings.blend_margin > 0))
    layout_margins = np.array([settings.blend_margin[::-1] if int(float(mappings[n].get('rotation_deg', 0))) in (90, 270)
                               else settings.blend_margin for n in layout_names])
    # Synthetic replacements occupy only their normal rectangle.
    layout_margins[len(names):] = 0
    lower = (layout_shifts - layout_margins).min(axis=0) - pad
    upper = (layout_shifts + layout_sizes + layout_margins).max(axis=0) + pad
    dimensions = np.ceil((upper - lower) / settings.pixel_nm)
    if not np.all(np.isfinite(dimensions)) or np.prod(dimensions) > 4_000_000:
        raise ValueError("Composite exceeds four million pixels. Increase the output pixel size.")
    shape = dimensions.astype(int)
    total = sum(np.count_nonzero(results[n]['picks'].accepted_mask) for n in names)
    if total == 0:
        raise ValueError("No accepted origami in the included templates.")
    source = _reconstruction_source(source, results, names, settings, progress)
    if progress:
        progress(f'Indexing {len(source):,} nearby source coordinates for reconstruction…')
    tree = cKDTree(source)
    # Only unseen / unique / shared matter. Saturate at two instead of
    # allocating four bytes per source localization for an unbounded count.
    if progress:
        progress(f'Allocating overlap tracking for {len(source):,} localizations…')
    ownership = np.zeros(len(source), dtype=np.uint8)
    counts = {}
    checked = 0
    def regions(name):
        picks = results[name]['picks']
        for index in np.flatnonzero(picks.accepted_mask):
            yield int(index), extract_image_points(source, tree, picks.rectangle_corners_nm[index], settings,
                                                   (exclusions or {}).get(name, ()), return_weights=True)
    if progress:
        progress(f'Checking shared signal: 0/{total} accepted origami')
    for name in names:
        counts[name] = int(np.count_nonzero(results[name]['picks'].accepted_mask))
        for _, (ids, _, _) in regions(name):
            ownership[ids] = np.minimum(ownership[ids] + 1, 2)
            checked += 1
            if progress and (checked % 100 == 0 or checked == total):
                progress(f'Checking shared signal: {checked}/{total} accepted origami')
    if progress:
        progress(f'Allocating {shape[0]:,} × {shape[1]:,} composite and coverage rasters…')
    density = np.zeros((shape[1], shape[0], 3), dtype=float)
    coverage = np.zeros_like(density, dtype=np.uint32)
    normalize_overlap = blending and settings.normalize_overlap
    blend_total = np.zeros_like(density) if normalize_overlap else None
    species = []
    candidates = []
    done = 0
    for species_index, (name, shift, size) in enumerate(zip(names, shifts, sizes), 1):
        label = f'{name} ({species_index}/{len(names)} templates)'
        if progress:
            progress(f'Rendering {label}: {done}/{total} accepted origami')
        mapping = mappings[name]
        channel = ('Red', 'Green', 'Blue').index(mapping['color'])
        plane = np.zeros(density.shape[:2])
        signal_count = 0
        weighted_signal_count = 0.0
        for index, (ids, points, weights) in regions(name):
            keep = ownership[ids] == 1
            points = points[keep].copy()
            weights = weights[keep]
            # A reflected identification reference changes the local Y frame.
            # Restore design top-to-bottom before decoded placement and RGB rendering.
            if results[name].get('params', {}).get('mirror_all_templates', False):
                points[:, 0] = settings.width_nm - points[:, 0]
            if results[name].get('params', {}).get('mirror_template_y', False):
                points[:, 1] = settings.height_nm - points[:, 1]
            points, _ = orient_section(points, settings.width_nm, settings.height_nm, mapping, clip=not blending)
            points += shift
            cells = np.floor((points - lower) / settings.pixel_nm).astype(int)
            cells = np.clip(cells, [0, 0], shape - 1)
            np.add.at(plane, (cells[:, 1], cells[:, 0]), weights)
            signal_count += len(points)
            weighted_signal_count += float(weights.sum())
            saved_indices = getattr(results[name]['picks'], 'candidate_indices', None)
            original_index = index if saved_indices is None else int(saved_indices[index])
            candidates.append(dict(template=name, candidate_index=original_index, image_localizations=len(points),
                                   weighted_image_localizations=float(weights.sum()),
                                   overlapping_localizations_excluded=int(np.count_nonzero(~keep))))
            done += 1
            if progress:
                progress(f"Rendering {label}: {done}/{total} accepted origami")
        if progress:
            progress(f'Averaging and smoothing {label}…')
        if settings.average_species and counts[name]:
            plane /= counts[name]
        if settings.sigma_nm:
            plane = gaussian_filter(plane, settings.sigma_nm / settings.pixel_nm, mode='constant')
        brightness = float(mapping.get('brightness', 1.0))
        if progress:
            progress(f'Combining weighted signal and coverage for {label}…')
        density[:, :, channel] += brightness * plane / settings.pixel_nm ** 2
        if counts[name]:
            xs = lower[0] + (np.arange(shape[0]) + .5) * settings.pixel_nm
            ys = lower[1] + (np.arange(shape[1]) + .5) * settings.pixel_nm
            if normalize_overlap:
                if progress:
                    progress(f'Preparing overlap blend weights for {label}…')
                # Work row-wise to avoid full-sized coordinate meshes.
                blend_plane = np.empty(density.shape[:2])
                for row, y in enumerate(ys):
                    coordinates = np.column_stack((xs - shift[0], np.full(len(xs), y - shift[1])))
                    blend_plane[row] = section_blend_weights(coordinates, size, layout_margins[species_index - 1])
                if settings.sigma_nm:
                    blend_plane = gaussian_filter(blend_plane, settings.sigma_nm / settings.pixel_nm, mode='constant')
                blend_total[:, :, channel] += blend_plane
            coverage[:, :, channel] += ((ys[:, None] >= shift[1]) & (ys[:, None] < shift[1] + size[1])
                                        & (xs[None, :] >= shift[0]) & (xs[None, :] < shift[0] + size[0]))
        species.append(dict(template=name, **dict(mapping, brightness=brightness), accepted_origami=counts[name],
                            image_localizations=signal_count, weighted_image_localizations=weighted_signal_count, missing=counts[name] == 0,
                            mirror_all_templates=bool(results[name].get('params', {}).get('mirror_all_templates', False)),
                            mirror_template_y=bool(results[name].get('params', {}).get('mirror_template_y', False))))
    if normalize_overlap:
        if progress:
            progress('Normalizing adjacent-section overlap blends…')
        # Preserve the fade at exposed edges while preventing bright overlap seams.
        density /= np.maximum(blend_total, 1.0)
    if settings.global_sigma_nm:
        if progress:
            progress('Smoothing the assembled image…')
        # Smooth spatial axes only: never mix red, green and blue channels.
        sigma = settings.global_sigma_nm / settings.pixel_nm
        gaussian_filter(density, sigma=(sigma, sigma, 0), mode='constant', output=density)
    # Synthetic replacements are applied after smoothing so every pixel in
    # the section has the requested fraction of the measured display reference.
    reference = float(np.max(density))
    synthetic_replacements = []
    if filled_names:
        if progress:
            progress('Applying uniform replacements for excluded templates…')
        reference = reference if reference > 0 else 1.0
        xs = lower[0] + (np.arange(shape[0]) + .5) * settings.pixel_nm
        ys = lower[1] + (np.arange(shape[1]) + .5) * settings.pixel_nm
        filled = np.zeros(density.shape, dtype=bool)
        for name, shift, size in zip(filled_names, layout_shifts[len(names):], layout_sizes[len(names):]):
            mapping = mappings[name]
            channel = ('Red', 'Green', 'Blue').index(mapping['color'])
            fraction = uniform_fill_fraction(mapping)
            mask = ((ys[:, None] >= shift[1]) & (ys[:, None] < shift[1] + size[1])
                    & (xs[None, :] >= shift[0]) & (xs[None, :] < shift[0] + size[0]))
            if np.any(filled[:, :, channel] & mask):
                raise ValueError(f'Uniform replacement for "{name}" overlaps another replacement in the same color channel.')
            filled[:, :, channel] |= mask
            density[:, :, channel][mask] = fraction * reference
            synthetic_replacements.append(dict(template=name, color=mapping['color'],
                fill_percent=100 * fraction, x_shift_nm=float(shift[0]), y_shift_nm=float(shift[1]),
                width_nm=float(size[0]), height_nm=float(size[1]), pixels=int(mask.sum()),
                density_value=fraction * reference, synthetic=True))
    if progress:
        progress('Applying final image orientation…')
    density = _orient_composite_raster(density, settings.rotation_deg, settings.mirror_x, settings.mirror_y)
    coverage = _orient_composite_raster(coverage, settings.rotation_deg, settings.mirror_x, settings.mirror_y)
    if progress:
        progress('Scaling RGB preview…')
    rgb = np.round(np.clip(density / max(reference, 1e-30), 0, 1) * 255).astype(np.uint8)
    if progress:
        progress('Compiling reconstruction report and overlap totals…')
    return dict(density=density, coverage=coverage, rgb=rgb, report=dict(
        format='paint-origami-composite-v1', settings=asdict(settings), species=species, candidates=candidates,
        excluded_templates=[n for n in results if n not in names],
        synthetic_replacements=synthetic_replacements,
        synthetic_replacement_meaning='Excluded sections may be overwritten after smoothing with a fixed percentage of the measured shared display reference (1 density unit if all measured signal is zero). Density and RGB include these synthetic values; coverage and localization counts remain measured. Section orientation and final image orientation still apply.',
        shared_localizations_excluded=int(np.count_nonzero(ownership > 1)),
        display_reference_density=reference, pre_rotation_origin_nm=lower.tolist(),
        output_pixel_nm=settings.pixel_nm, coordinate_frame='x right, y down; section clockwise rotation then mirrors, rebase section upper-left, decoded shifts, then clockwise display rotation, then horizontal/vertical display mirrors',
        marker_exclusion_meaning='Only outside the normal image rectangle, within the overlap extension: Gaussian transmission 1 - exp(-d^2 / (2*sigma^2)), where d is distance to the nearest excluded site and sigma is exclusion_radius_nm. The normal rectangle is unaffected, including marker centers. With no overlap extension marker exclusion has no effect. Zero sigma disables exclusion. Weighted localization counts are before averaging, smoothing, and brightness.',
        intensity_units='weighted localizations / nm^2' + (' / origami within species' if settings.average_species else ''),
        brightness_meaning='Per-species multiplier applied after averaging and smoothing, before summing channels; density contains weighted signal. RGB uses a shared maximum across all weighted channels.',
        overlap_blend_meaning='Centered expanded rectangle; full weight inside the normal rectangle, exp(-0.5*((3*dx/margin_x)^2+(3*dy/margin_y)^2)) outside, truncated at the blend rectangle. Zero dimension follows normal size. By default, weighted sections add without overlap correction. If normalize_overlap is enabled, same-channel sections divide by max(1, summed geometric blend weights) before whole-image smoothing. Physical source localizations shared by accepted extraction regions are still rejected.',
        smoothing_meaning='sigma_nm smooths each species before summation; global_sigma_nm smooths the assembled density spatially per RGB channel before display scaling. Coverage remains geometric and unsmoothed.',
        coverage_meaning='Number of observed species whose extraction rectangles cover each pixel; not site-level sampling coverage'))


def composite_figure(result):
    """Render the exported RGB raster at its calibrated physical dimensions."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    rgb = result['rgb']
    pixel_nm = float(result['report']['output_pixel_nm'])
    figure = Figure(figsize=(6, 7), layout='constrained')
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.imshow(rgb, origin='upper', interpolation='nearest', aspect='equal',
                extent=(0, rgb.shape[1] * pixel_nm, rgb.shape[0] * pixel_nm, 0))
    axis.set(xlabel='Display X (nm)', ylabel='Display Y (nm)',
             title='Reconstructed composite')
    return figure


def export_composite(result, path, progress=None):
    with _progress_heartbeat(progress) as report:
        _export_composite(result, path, report)
    if progress:
        progress('Composite export complete.')


def _export_composite(result, path, progress):
    path = Path(path)
    if progress:
        progress('Saving RGB image (1/4)…')
    Image.fromarray(result['rgb']).save(path)
    if progress:
        progress('Rendering and saving image with axes (2/4)…')
    figure = composite_figure(result)
    figure.savefig(path.with_name(path.stem + '_axes.png'), dpi=300)
    figure.clear()
    if progress:
        progress('Compressing and saving numerical density and coverage (3/4)…')
    np.savez_compressed(path.with_suffix('.npz'), density=result['density'], coverage=result['coverage'],
                        pixel_nm=result['report']['output_pixel_nm'])
    if progress:
        progress('Saving reconstruction report (4/4)…')
    path.with_suffix('.json').write_text(json.dumps(result['report'], indent=2) + '\n')


def image_region_defaults(geometry, grid, params):
    """Map the design rectangle into the saved identification reference frame."""
    grid = np.asarray(grid, dtype=float).copy()
    mirror_x = bool(params.get('mirror_all_templates', False))
    mirror_y = bool(params.get('mirror_template_y', False))
    grid *= [-1 if mirror_x else 1, -1 if mirror_y else 1]
    sx, sy = float(geometry['site_pitch_x_nm']), float(geometry['site_pitch_y_nm'])
    origin = geometry['image_origin_from_C12_H01_nm']
    width = len(geometry.get('image_columns_left_to_right', range(6))) * sx
    height = 8 * sy
    left = float(grid[:, 0].min()) + float(origin[0]) - sx / 2
    top = float(grid[:, 1].max()) - float(origin[1]) + sy / 2
    return dict(left_nm=-left-width if mirror_x else left,
                top_nm=height-top if mirror_y else top,
                width_nm=width, height_nm=height,
                rotation_deg=int(geometry.get('origami_rotation_degrees_clockwise', 0)))


def saved_rgb_batch_mapping(document, template_names, grid, params):
    """Read a saved RGB picklist manifest; never infer image placement from Echo rows."""
    designs = document.get('designs', [])
    if not designs or not document.get('runs'):
        raise ValueError('Choose image_manifest.json from Save image batch; a transfer CSV has no RGB placement metadata.')
    geometry = document['geometry']
    columns = geometry.get('image_columns_left_to_right', [])
    if not columns or any(int(a)-int(b)!=1 for a,b in zip(columns,columns[1:])):
        raise ValueError('The saved image region must contain contiguous columns.')
    sx, sy = float(geometry['site_pitch_x_nm']), float(geometry['site_pitch_y_nm'])
    grid = np.asarray(grid)
    if (int(params.get('rows',8)) != 8 or int(params.get('columns',12)) != 12
        or not np.allclose([params.get('spacing_x_nm',sx),params.get('spacing_y_nm',sy)],[sx,sy])
        or not np.allclose(np.ptp(grid,axis=0),[11*sx+10,7*sy],atol=.05)):
        raise ValueError('Load compatible full 12-column / 8-row templates with the central 10 nm gap first.')
    codebook, aliases = {}, {}
    for index, design in enumerate(designs,1):
        bits=design['bits']
        if not bits or any(bit not in (0,1) for bit in bits):
            raise ValueError('Invalid saved barcode bits.')
        code=''.join(str(int(bit)) for bit in bits)
        if code in codebook:
            raise ValueError('Saved RGB batch contains duplicate barcodes.')
        color={'R1 R':'Red','R1 G':'Green','R1 B':'Blue'}.get(design['color'])
        mapping=dict(code=code,color=color,x_shift_nm=float(design['x_shift_nm']),
                     y_shift_nm=float(design['y_shift_nm']),include=True,rotation_deg=0,
                     mirror_x=False,mirror_y=False,brightness=1)
        validate_mapping(mapping)
        codebook[code]=mapping
        for alias in (design['name'],f'design_{index:04d}_code_{int(code,2):03d}'):
            if alias in aliases and aliases[alias]!=code:
                raise ValueError('Ambiguous saved design names.')
            aliases[alias]=code
    mappings={name:dict(codebook[aliases[name]]) for name in template_names if name in aliases}
    defaults = image_region_defaults(geometry, grid, params)
    if defaults['rotation_deg'] not in (0,90,180,270):
        raise ValueError('Invalid saved image rotation.')
    return geometry, codebook, mappings, defaults
