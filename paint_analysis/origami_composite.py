"""Measured image-side reconstruction from classified, rigidly aligned origami."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


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
    exclusion_radius_nm: float = 2.0
    mirror_x: bool = False
    mirror_y: bool = False
    global_sigma_nm: float = 0.0

    def validate(self):
        values = [self.left_nm, self.top_nm, self.width_nm, self.height_nm,
                  self.pixel_nm, self.sigma_nm, self.global_sigma_nm, self.exclusion_radius_nm]
        if not np.all(np.isfinite(values)):
            raise ValueError("Image geometry and rendering settings must be finite.")
        if min(self.width_nm, self.height_nm, self.pixel_nm) <= 0:
            raise ValueError("Image width, height and pixel size must be positive.")
        if min(self.sigma_nm, self.global_sigma_nm, self.exclusion_radius_nm) < 0:
            raise ValueError("Smoothing and exclusion radius cannot be negative.")
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


def orient_section(points, width, height, mapping):
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
    points = np.clip(points, [0, 0], np.nextafter([width, height], [0, 0]))
    return points, (width, height)


def reconstruction_mapping_issues(results, mappings):
    """Name every included template that needs attention before a build."""
    issues = []
    codes = {}
    for name in results:
        mapping = mappings.get(name)
        if mapping is not None and not mapping.get('include', True):
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


def extract_image_points(source, tree, corners, settings, excluded_sites=()):
    """Fetch from the source, including signal outside the classification crop."""
    center, basis = pose_basis(corners)
    local_center = np.array([settings.left_nm + settings.width_nm / 2,
                             settings.top_nm - settings.height_nm / 2])
    world_center = center + local_center @ basis.T
    ids = np.asarray(tree.query_ball_point(world_center, np.hypot(settings.width_nm, settings.height_nm) / 2), dtype=int)
    points = (source[ids] - center) @ basis
    inside = ((points[:, 0] >= settings.left_nm) & (points[:, 0] < settings.left_nm + settings.width_nm)
              & (points[:, 1] <= settings.top_nm) & (points[:, 1] > settings.top_nm - settings.height_nm))
    excluded_sites = np.asarray(excluded_sites, dtype=float).reshape(-1, 2)
    if len(excluded_sites) and settings.exclusion_radius_nm > 0:
        inside &= cKDTree(excluded_sites).query(points)[0] > settings.exclusion_radius_nm
    points, ids = points[inside], ids[inside]
    points = np.column_stack((points[:, 0] - settings.left_nm, settings.top_nm - points[:, 1]))
    return ids, points


def reconstruct_composite(source, results, mappings, settings, exclusions=None, progress=None):
    settings.validate()
    source = np.asarray(source, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2 or not len(source):
        raise ValueError("Load finite source localization coordinates before reconstruction.")
    for start in range(0, len(source), 100_000):
        if not np.all(np.isfinite(source[start:start + 100_000])):
            raise ValueError("Load finite source localization coordinates before reconstruction.")
    if not results:
        raise ValueError("Run Step 4 to classify barcode templates first.")
    names = [name for name in results if mappings.get(name, {}).get('include', True)]
    if not names:
        raise ValueError("Include at least one template.")
    issues = reconstruction_mapping_issues(results, mappings)
    if issues:
        raise ValueError("Check barcode mappings:\n" + "\n".join(f'Template "{name}": {message}' for name, message in issues))
    shifts = np.array([[float(mappings[n]['x_shift_nm']), float(mappings[n]['y_shift_nm'])] for n in names])
    sizes = np.array([orient_section(np.empty((0, 2)), settings.width_nm, settings.height_nm, mappings[n])[1] for n in names])
    pad = 4 * (settings.sigma_nm + settings.global_sigma_nm)
    lower = shifts.min(axis=0) - pad
    upper = (shifts + sizes).max(axis=0) + pad
    dimensions = np.ceil((upper - lower) / settings.pixel_nm)
    if not np.all(np.isfinite(dimensions)) or np.prod(dimensions) > 4_000_000:
        raise ValueError("Composite exceeds four million pixels. Increase the output pixel size.")
    shape = dimensions.astype(int)
    total = sum(np.count_nonzero(results[n]['picks'].accepted_mask) for n in names)
    if total == 0:
        raise ValueError("No accepted origami in the included templates.")
    if progress:
        progress('Indexing source coordinates for reconstruction…')
    tree = cKDTree(source)
    # Only unseen / unique / shared matter. Saturate at two instead of
    # allocating four bytes per source localization for an unbounded count.
    ownership = np.zeros(len(source), dtype=np.uint8)
    counts = {}
    checked = 0
    def regions(name):
        picks = results[name]['picks']
        for index in np.flatnonzero(picks.accepted_mask):
            yield int(index), extract_image_points(source, tree, picks.rectangle_corners_nm[index], settings,
                                                   (exclusions or {}).get(name, ()))
    for name in names:
        counts[name] = int(np.count_nonzero(results[name]['picks'].accepted_mask))
        for _, (ids, _) in regions(name):
            ownership[ids] = np.minimum(ownership[ids] + 1, 2)
            checked += 1
            if progress and (checked % 100 == 0 or checked == total):
                progress(f'Checking shared signal: {checked}/{total} accepted origami')
    density = np.zeros((shape[1], shape[0], 3), dtype=float)
    coverage = np.zeros_like(density, dtype=np.uint32)
    species = []
    candidates = []
    done = 0
    for name, shift, size in zip(names, shifts, sizes):
        mapping = mappings[name]
        channel = ('Red', 'Green', 'Blue').index(mapping['color'])
        plane = np.zeros(density.shape[:2])
        signal_count = 0
        for index, (ids, points) in regions(name):
            keep = ownership[ids] == 1
            points = points[keep].copy()
            # A reflected identification reference changes the local Y frame.
            # Restore design top-to-bottom before decoded placement and RGB rendering.
            if results[name].get('params', {}).get('mirror_all_templates', False):
                points[:, 0] = settings.width_nm - points[:, 0]
            if results[name].get('params', {}).get('mirror_template_y', False):
                points[:, 1] = settings.height_nm - points[:, 1]
            points, _ = orient_section(points, settings.width_nm, settings.height_nm, mapping)
            points += shift
            cells = np.floor((points - lower) / settings.pixel_nm).astype(int)
            cells = np.clip(cells, [0, 0], shape - 1)
            np.add.at(plane, (cells[:, 1], cells[:, 0]), 1)
            signal_count += len(points)
            candidates.append(dict(template=name, candidate_index=index, image_localizations=len(points),
                                   overlapping_localizations_excluded=int(np.count_nonzero(~keep))))
            done += 1
            if progress:
                progress(f"Rendering {done}/{total} accepted origami")
        if settings.average_species and counts[name]:
            plane /= counts[name]
        if settings.sigma_nm:
            plane = gaussian_filter(plane, settings.sigma_nm / settings.pixel_nm, mode='constant')
        brightness = float(mapping.get('brightness', 1.0))
        density[:, :, channel] += brightness * plane / settings.pixel_nm ** 2
        if counts[name]:
            xs = lower[0] + (np.arange(shape[0]) + .5) * settings.pixel_nm
            ys = lower[1] + (np.arange(shape[1]) + .5) * settings.pixel_nm
            coverage[:, :, channel] += ((ys[:, None] >= shift[1]) & (ys[:, None] < shift[1] + size[1])
                                        & (xs[None, :] >= shift[0]) & (xs[None, :] < shift[0] + size[0]))
        species.append(dict(template=name, **dict(mapping, brightness=brightness), accepted_origami=counts[name],
                            image_localizations=signal_count, missing=counts[name] == 0,
                            mirror_all_templates=bool(results[name].get('params', {}).get('mirror_all_templates', False)),
                            mirror_template_y=bool(results[name].get('params', {}).get('mirror_template_y', False))))
    if settings.global_sigma_nm:
        if progress:
            progress('Smoothing the assembled image…')
        # Smooth spatial axes only: never mix red, green and blue channels.
        sigma = settings.global_sigma_nm / settings.pixel_nm
        gaussian_filter(density, sigma=(sigma, sigma, 0), mode='constant', output=density)
    density = _orient_composite_raster(density, settings.rotation_deg, settings.mirror_x, settings.mirror_y)
    coverage = _orient_composite_raster(coverage, settings.rotation_deg, settings.mirror_x, settings.mirror_y)
    reference = float(np.max(density))
    rgb = np.round(np.clip(density / max(reference, 1e-30), 0, 1) * 255).astype(np.uint8)
    return dict(density=density, coverage=coverage, rgb=rgb, report=dict(
        format='paint-origami-composite-v1', settings=asdict(settings), species=species, candidates=candidates,
        excluded_templates=[n for n in results if n not in names],
        shared_localizations_excluded=int(np.count_nonzero(ownership > 1)),
        display_reference_density=reference, pre_rotation_origin_nm=lower.tolist(),
        output_pixel_nm=settings.pixel_nm, coordinate_frame='x right, y down; section clockwise rotation then mirrors, rebase section upper-left, decoded shifts, then clockwise display rotation, then horizontal/vertical display mirrors',
        intensity_units='localizations / nm^2' + (' / origami within species' if settings.average_species else ''),
        brightness_meaning='Per-species multiplier applied after averaging and smoothing, before summing channels; density contains weighted signal. RGB uses a shared maximum across all weighted channels.',
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


def export_composite(result, path):
    path = Path(path)
    Image.fromarray(result['rgb']).save(path)
    figure = composite_figure(result)
    figure.savefig(path.with_name(path.stem + '_axes.png'), dpi=300)
    figure.clear()
    np.savez_compressed(path.with_suffix('.npz'), density=result['density'], coverage=result['coverage'],
                        pixel_nm=result['report']['output_pixel_nm'])
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
