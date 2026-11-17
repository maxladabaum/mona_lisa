"""Equal-weight molecular occupancy, branching after barcode assignment/alignment.

No intensity fit or concentration inference is performed. Unknown states are -1,
never OFF. Manifest order is retained throughout the molecular matrices.
"""
from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from occupancy_rendering import (rendering_settings, gaussian_raster, overlay_on_gaussian_grid,
                                 draw_template_heatmaps)

from origami_composite import (CompositeSettings, _orient_composite_raster,
    _reconstruction_source, extract_image_points, orient_section, pose_basis,
    reconstruct_composite, reconstruction_mapping_issues)


MODES = ('Localization Overlay (Existing)', 'Binary Occupancy (New)', 'Compare Both')


@dataclass(frozen=True)
class OccupancySettings:
    radius_nm: float = 2.0
    min_localizations: int = 3
    per_position_thresholds: dict | None = None  # manifest site name -> count
    uncertainty_margin: int = 0  # [T-margin, T+margin), disabled at zero
    manual_uncertain: dict | None = None  # template -> candidate ID string -> site names
    overlay_density_per_on: float | None = None  # independent calibration, not target fit
    diagnostic_origami: int = 6
    gaussian_fwhm_nm: float | None = None  # None uses the saved picklist width

    def validate(self):
        if self.gaussian_fwhm_nm is not None:
            if not np.isfinite(self.gaussian_fwhm_nm) or self.gaussian_fwhm_nm <= 0:
                raise ValueError('Gaussian FWHM must be finite and positive, or blank to use the manifest.')
        if not np.isfinite(self.radius_nm) or self.radius_nm <= 0:
            raise ValueError('Occupancy ROI radius must be finite and positive.')
        for label, value, lower in [('Minimum localizations', self.min_localizations, 1),
                                    ('Uncertainty margin', self.uncertainty_margin, 0),
                                    ('Diagnostic origami', self.diagnostic_origami, 0)]:
            if not np.isfinite(value) or int(value) != value or value < lower:
                raise ValueError(f'{label} must be an integer >= {lower}.')
        if not isinstance(self.per_position_thresholds or {}, dict):
            raise ValueError('Per-position thresholds must be a site-name to count object.')
        for value in (self.per_position_thresholds or {}).values():
            if not np.isfinite(value) or int(value) != value or value < 1:
                raise ValueError('Per-position thresholds must be positive integers.')
        if not isinstance(self.manual_uncertain or {}, dict):
            raise ValueError('Manual uncertainty must be a template / candidate / site-name object.')
        if self.overlay_density_per_on is not None:
            if not np.isfinite(self.overlay_density_per_on) or self.overlay_density_per_on <= 0:
                raise ValueError('Independent overlay calibration must be finite and positive.')


def classify_sites(points, sites, settings):
    """Nearest site within a circular ROI; ties go to one site only."""
    settings.validate()
    sites = np.asarray(sites, float).reshape(-1, 2)
    points = np.asarray(points, float).reshape(-1, 2)
    if not len(sites) or not np.all(np.isfinite(sites)) or not np.all(np.isfinite(points)):
        raise ValueError('Site and localization coordinates must be finite, with at least one site.')
    distances, assigned = cKDTree(sites).query(points)
    valid = distances <= settings.radius_nm
    counts = np.bincount(assigned[valid], minlength=len(sites))
    return counts, np.where(valid, assigned, -1)


def summarize_states(states):
    """At most one vote per molecule/site; no observations returns NaN."""
    states = np.asarray(states)
    if states.ndim != 2 or not np.isin(states, [-1, 0, 1]).all():
        raise ValueError('Occupancy states must be a 2D matrix containing only -1, 0, 1.')
    valid = (states >= 0).sum(axis=0)
    on = (states == 1).sum(axis=0)
    occupancy = np.divide(on, valid, out=np.full(valid.shape, np.nan), where=valid > 0)
    se = np.sqrt(np.divide(occupancy * (1 - occupancy), valid,
                          out=np.full(valid.shape, np.nan), where=valid > 0))
    return dict(on=on, valid=valid, excluded=len(states) - valid, occupancy=occupancy, standard_error=se)


def _design_code(design):
    return ''.join(str(int(bit)) for bit in design['bits'])


def manifest_layout(manifest, results, mappings, grids, settings):
    """Resolve design names to physical sites and exact programmed pixel cells.

    Full grids are centered y-up in the saved alignment reference. The design
    labels run C12->C01 left-to-right and H01->H15 top-to-bottom.
    """
    settings.validate()
    if not manifest or not manifest.get('designs'):
        raise ValueError('Binary occupancy requires the saved RGB image_manifest.json with image_sites.')
    issues = reconstruction_mapping_issues(results, mappings)
    if issues:
        raise ValueError('Check barcode mappings: ' + '; '.join(f'{n}: {m}' for n, m in issues))
    geometry = manifest['geometry']
    edges = [np.asarray(geometry[f'pixel_edges_{axis}_nm'], float) for axis in ('x', 'y')]
    for edge in edges:
        if edge.ndim != 1 or len(edge) < 2 or not np.isfinite(edge).all() or np.any(np.diff(edge) <= 0):
            raise ValueError('Manifest pixel edges must be finite and strictly increasing.')
    if settings.pixel_nm > min(np.diff(edge).min() for edge in edges) + 1e-8:
        raise ValueError('Output pixel size exceeds a programmed pixel width; reduce Output pixel (nm).')
    designs = {_design_code(d): d for d in manifest['designs']}
    if len(designs) != len(manifest['designs']):
        raise ValueError('Duplicate barcode codes in manifest.')
    observed = {}
    for name in results:
        if mappings[name].get('include', True):
            code = mappings[name].get('code', '')
            if code not in designs:
                raise ValueError(f'{name}: barcode is absent from the saved RGB manifest.')
            observed[code] = name
    if not observed:
        raise ValueError('Include at least one mapped barcode template.')
    reference = next(iter(observed.values()))
    layout = {}
    # Include unobserved designs: their pixels remain missing, including classes
    # for which no classification template was loaded at all.
    for code, design in designs.items():
        name = observed.get(code)
        key = name if name is not None else f'missing:{code}'
        params = results[name or reference].get('params', {})
        grid = np.asarray(grids[name or reference], float)
        signs = np.array([-1 if params.get('mirror_all_templates') else 1,
                          -1 if params.get('mirror_template_y') else 1])
        canonical = grid * signs
        xs, ys = np.unique(canonical[:, 0]), np.unique(canonical[:, 1])[::-1]
        if len(xs) != 12 or len(ys) != 8:
            raise ValueError('Occupancy requires the full 12-column / 8-row physical design grid.')
        sx, sy = float(geometry['site_pitch_x_nm']), float(geometry['site_pitch_y_nm'])
        expected_x = np.arange(12) * sx + np.where(np.arange(12) >= 6, 10., 0.)
        if not np.allclose(xs - xs[0], expected_x, atol=.05) or not np.allclose(ys[0] - ys, np.arange(8) * sy, atol=.05):
            raise ValueError('Manifest site pitch or central gap differs from the aligned physical grid.')
        sites = design.get('image_sites', [])
        ids = [s['name'] for s in sites]
        if not sites or len(set(ids)) != len(ids):
            raise ValueError(f'{code}: missing or duplicate analog site identifiers.')
        physical, cells, target, pixels, centers = [], [], [], [], []
        mapping = mappings[name] if name else dict(code=code, color={'R1 R': 'Red', 'R1 G': 'Green', 'R1 B': 'Blue'}[design['color']],
            x_shift_nm=design['x_shift_nm'], y_shift_nm=design['y_shift_nm'])
        channel = ('Red', 'Green', 'Blue').index(mapping['color'])
        design_shift = np.array([design['x_shift_nm'], design['y_shift_nm']], float)
        shift = np.array([mapping['x_shift_nm'], mapping['y_shift_nm']], float)
        for site in sites:
            row = int(str(site['row']).removeprefix('H'))
            column = int(site['column'])
            if row not in range(1, 16, 2) or column not in range(1, 13):
                raise ValueError(f'Invalid physical site {site["name"]}.')
            physical.append(np.array([xs[12 - column], ys[(row - 1) // 2]]) * signs)
            x, y = site['pixel_x'], site['pixel_y']
            if int(x) != x or int(y) != y or not (0 <= x < len(edges[0])-1 and 0 <= y < len(edges[1])-1):
                raise ValueError('Invalid manifest output pixel index.')
            x, y = int(x), int(y)
            rect = np.array([[edges[0][x], edges[1][y]], [edges[0][x+1], edges[1][y+1]]]) - design_shift
            # Pixel cells follow the same per-section and whole-image transforms
            # as the existing renderer, without smoothing or intensity multipliers.
            rect, _ = orient_section(rect, settings.width_nm, settings.height_nm, mapping, clip=False)
            rect += shift
            cells.append([rect.min(axis=0), rect.max(axis=0)])
            # The generator's coordinate origin is the first emitter center.
            # Translate by half the first pixel pitch to our upper-left frame,
            # then apply exactly the same section transforms and decoded shifts.
            pitch = np.array([geometry.get('pixel_pitch_x_nm', np.diff(edges[0])[0]),
                              geometry.get('pixel_pitch_y_nm', np.diff(edges[1])[0])], float)
            center = np.array([site.get('x_nm', edges[0][x]), site.get('y_nm', edges[1][y])], float) + pitch/2 - design_shift
            if not np.isfinite(center).all():
                raise ValueError('Gaussian site coordinates must be finite.')
            center, _ = orient_section(center[None, :], settings.width_nm, settings.height_nm, mapping, clip=False)
            centers.append(center[0] + shift)
            value = site.get('replacement_fraction', np.nan)
            if value is None:
                value = np.nan
            if not np.isnan(value) and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Programmed fractions must be between zero and one.')
            target.append(value)
            pixels.append([x, y])
        if len(np.unique(pixels, axis=0)) != len(pixels):
            raise ValueError('Two sites in one barcode map to the same output pixel.')
        if len(np.unique(physical, axis=0)) != len(physical):
            raise ValueError('Two analog identifiers refer to the same physical site.')
        gaussian_weight = (design.get('quadrant_normalization_weight', 1.0)
            if (manifest.get('gaussian_reconstruction') or {}).get('normalize_designs', False)
            else 1.0 / float(design.get('redundancy', 1)))
        if not np.isfinite(gaussian_weight) or gaussian_weight <= 0:
            raise ValueError('Gaussian design weights must be finite and positive.')
        layout[key] = dict(gaussian_centers=np.asarray(centers), gaussian_weight=float(gaussian_weight), template=name, code=code, ids=ids, sites=np.asarray(physical),
            grid_nm=grid.copy(), cells=np.asarray(cells), targets=np.asarray(target), pixels=np.asarray(pixels),
            channel=channel, mapping=mapping, design_name=design['name'])
    return layout


def _measure(source, results, layout, settings, classifier, progress):
    names = [row['template'] for row in layout.values() if row['template'] is not None]
    accepted = sum(np.count_nonzero(results[n]['picks'].accepted_mask) for n in names)
    if accepted:
        source = _reconstruction_source(source, results, names, settings, progress)
    else:
        source = np.empty((0, 2))
    tree = cKDTree(source)
    ownership = np.zeros(len(source), np.uint8)
    checked = 0
    # Use the existing rectangle-level shared-signal rule. Affected site
    # observations become uncertain instead of turning into false OFF calls.
    for name in names:
        picks = results[name]['picks']
        for index in np.flatnonzero(picks.accepted_mask):
            ids, _ = extract_image_points(source, tree, picks.rectangle_corners_nm[index], settings)
            ownership[ids] = np.minimum(ownership[ids] + 1, 2)
            checked += 1
            if progress and (checked % 100 == 0 or checked == accepted):
                progress(f'Checking shared source signal: {checked}/{accepted} origami')
    populations = {}
    known_ids = {site for row in layout.values() for site in row['ids']}
    unknown = set(classifier.per_position_thresholds or {}) - known_ids
    if unknown:
        raise ValueError('Unknown per-position threshold identifiers: ' + ', '.join(sorted(unknown)))
    if set(classifier.manual_uncertain or {}) - set(names):
        raise ValueError('Manual uncertainty contains an unknown or excluded template.')
    done = 0
    for key, row in layout.items():
        name, sites = row['template'], row['sites']
        thresholds = np.array([(classifier.per_position_thresholds or {}).get(s, classifier.min_localizations) for s in row['ids']])
        picks = results[name]['picks'] if name else None
        indices = np.flatnonzero(picks.accepted_mask) if name else np.empty(0, int)
        candidate_ids = np.asarray(getattr(picks, 'candidate_indices', np.arange(len(picks.accepted_mask))))[indices] if name else indices
        if len(np.unique(candidate_ids)) != len(candidate_ids):
            raise ValueError(f'{name}: duplicate candidate identifiers.')
        manual = (classifier.manual_uncertain or {}).get(name, {})
        if set(manual) - {str(i) for i in candidate_ids}:
            raise ValueError(f'{name}: manual uncertainty refers to an unavailable candidate.')
        counts = np.zeros((len(indices), len(sites)), np.int64)
        states = np.full(counts.shape, -1, np.int8)
        reasons = np.full(counts.shape, '', dtype='<U32')
        examples, qc = [], []
        # A barcode-only aligned localization overlay for diagnostics. Retain
        # integer events here; this image does not determine occupancy.
        bins_x = np.linspace(settings.left_nm, settings.left_nm + settings.width_nm,
                             min(513, max(2, int(np.ceil(settings.width_nm / settings.pixel_nm)) + 1)))
        bins_y = np.linspace(settings.top_nm - settings.height_nm, settings.top_nm,
                             min(513, max(2, int(np.ceil(settings.height_nm / settings.pixel_nm)) + 1)))
        localization_overlay = np.zeros((len(bins_y)-1, len(bins_x)-1))
        # A radius truncated by a saved extraction region cannot establish OFF.
        local_sites = np.column_stack((sites[:, 0] - settings.left_nm, settings.top_nm - sites[:, 1]))
        margin = settings.blend_margin
        truncated = np.any((local_sites - classifier.radius_nm < -margin) |
                           (local_sites + classifier.radius_nm > np.array([settings.width_nm, settings.height_nm]) + margin), axis=1)
        for n, index in enumerate(indices):
            corners = picks.rectangle_corners_nm[index]
            center, basis = pose_basis(corners)
            ids, _ = extract_image_points(source, tree, corners, settings)
            points = (source[ids] - center) @ basis
            unique = ownership[ids] == 1
            localization_overlay += np.histogram2d(points[unique, 1], points[unique, 0], bins=(bins_y, bins_x))[0]
            counts[n], _ = classify_sites(points[unique], sites, classifier)
            shared_counts, _ = classify_sites(points[~unique], sites, classifier)
            states[n] = counts[n] >= thresholds
            band = classifier.uncertainty_margin
            if band:
                reasons[n, (counts[n] >= thresholds - band) & (counts[n] < thresholds + band)] = 'threshold_band'
            reasons[n, shared_counts > 0] = 'shared_localizations'
            reasons[n, truncated] = 'roi_truncated'
            requested = manual.get(str(candidate_ids[n]), [])
            if set(requested) - set(row['ids']):
                raise ValueError(f'{name}: manual uncertainty contains an unknown site identifier.')
            reasons[n, np.isin(row['ids'], requested)] = 'manual'
            ambiguous = np.asarray(getattr(picks, 'orientation_ambiguous', []))
            if len(ambiguous) > index and ambiguous[index]:
                reasons[n, :] = 'alignment_ambiguous'
            states[n, reasons[n] != ''] = -1
            info = dict(candidate_index=int(candidate_ids[n]), qc_accepted=True, corners_nm=np.asarray(corners).tolist())
            for field in ('rectangle_confidence', 'rectangle_fit_rms_nm', 'orientation_ambiguous', 'orientation_flipped'):
                array = np.asarray(getattr(picks, field, []))
                if len(array) > index:
                    info[field] = array[index].item()
            qc.append(info)
            if n < classifier.diagnostic_origami:
                examples.append(dict(index=n, points=points, shared=~unique))
            done += 1
            if progress and (done % 100 == 0 or done == accepted):
                progress(f'Counting individual analog sites: {done}/{accepted} origami')
        populations[key] = dict(**row, counts=counts, states=states, reasons=reasons,
            candidate_ids=candidate_ids, thresholds=thresholds, qc=qc, examples=examples,
            localization_overlay=localization_overlay,
            overlay_extent=[float(bins_x[0]), float(bins_x[-1]), float(bins_y[0]), float(bins_y[-1])],
            rejected_candidate_ids=(np.asarray(getattr(picks, 'candidate_indices', np.arange(len(picks.accepted_mask))))[~np.asarray(picks.accepted_mask, bool)] if name else np.empty(0, int)),
            rejected_candidates=int(np.count_nonzero(~np.asarray(picks.accepted_mask, bool))) if name else 0,
            **summarize_states(states))
    return populations, int(np.count_nonzero(ownership > 1))


def _raster(populations, settings, frame=None):
    cells = np.concatenate([p['cells'] for p in populations.values()])
    lower = cells[:, 0].min(axis=0)
    upper = cells[:, 1].max(axis=0)
    if frame is not None:
        lower, shape = frame
    else:
        shape = np.ceil((upper - lower) / settings.pixel_nm).astype(int)
    if np.prod(shape) > 4_000_000 or min(shape) <= 0:
        raise ValueError('Occupancy image exceeds four million pixels. Increase output pixel size.')
    on = np.zeros((shape[1], shape[0], 3), float)
    valid = np.zeros_like(on, dtype=np.int64)
    targets = np.zeros_like(on)
    target_n = np.zeros_like(on, dtype=np.int32)
    xs = lower[0] + (np.arange(shape[0]) + .5) * settings.pixel_nm
    ys = lower[1] + (np.arange(shape[1]) + .5) * settings.pixel_nm
    for population in populations.values():
        channel = population['channel']
        for j, (lo, hi) in enumerate(population['cells']):
            x0, x1 = np.searchsorted(xs, [lo[0]-1e-9, hi[0]-1e-9])
            y0, y1 = np.searchsorted(ys, [lo[1]-1e-9, hi[1]-1e-9])
            if x0 == x1 or y0 == y1:
                raise ValueError('An analog pixel has no output samples; decrease Output pixel (nm).')
            region = np.s_[y0:y1, x0:x1, channel]
            on[region] += population['on'][j]
            valid[region] += population['valid'][j]
            target = population['targets'][j]
            if np.isfinite(target):
                targets[region] += target
                target_n[region] += 1
    occupancy = np.divide(on, valid, out=np.full(on.shape, np.nan), where=valid > 0)
    target = np.divide(targets, target_n, out=np.full(on.shape, np.nan), where=target_n > 0)
    orient = lambda a: _orient_composite_raster(a, settings.rotation_deg, settings.mirror_x, settings.mirror_y)
    return orient(occupancy), orient(valid), orient(target), np.asarray(lower)


def display_rgb(values):
    """Fixed linear 0..1 scale. Missing channels shown magenta in the preview."""
    rgb = np.floor(np.clip(np.nan_to_num(values), 0, 1) * 255 + .5).astype(np.uint8)
    rgb[np.any(~np.isfinite(values), axis=2)] = [255, 0, 255]
    return rgb


def error_metrics(values, targets):
    keep = np.isfinite(values) & np.isfinite(targets)
    error = np.asarray(values)[keep] - np.asarray(targets)[keep]
    return dict(n=int(keep.sum()), mae=float(np.abs(error).mean()) if len(error) else None,
                rmse=float(np.sqrt(np.mean(error ** 2))) if len(error) else None)


def reconstruct_occupancy(source, results, mappings, settings, manifest, grids,
                          classifier=None, exclusions=None, compare=False, progress=None):
    classifier = classifier or OccupancySettings()
    classifier.validate()
    rendering = rendering_settings(manifest, classifier.gaussian_fwhm_nm) if manifest else None
    source = np.asarray(source)
    if source.ndim != 2 or source.shape[1] != 2:
        raise ValueError('Source coordinates must be an N × 2 array.')
    layout = manifest_layout(manifest, results, mappings, grids, settings)
    populations, shared = _measure(source, results, layout, settings, classifier, progress)
    overlay = None
    if compare:
        if not any(len(p['states']) for p in populations.values()):
            raise ValueError('No accepted origami are available for an overlay comparison.')
        overlay = reconstruct_composite(source, results, mappings, settings, exclusions, progress)
    # A shared physical frame includes missing designs even if absent from the
    # overlay's template list. Overlay samples are placed into this frame below.
    density, coverage, target, lower = _raster(populations, settings)
    report = dict(format='paint-binary-occupancy-v1', settings=asdict(settings),
        occupancy_settings=asdict(classifier), output_pixel_nm=settings.pixel_nm,
        pre_rotation_origin_nm=lower.tolist(), shared_localizations_excluded=shared,
        intensity_units='ON origami / valid origami; fixed linear 0–1 scale',
        missing_display='Magenta denotes any missing RGB channel; NaN and masks in numerical exports are authoritative.',
        qc_policy='Only accepted_mask candidates contribute. Rejected candidates are not assigned this barcode; unclassified candidates never enter the denominator. Compact reconstruction archives retain accepted candidates only.',
        overlap_policy='Shared source localizations are removed; affected observations are uncertain. Multiple measured barcodes mapped to one output pixel pool ON and valid counts.',
        ignored_overlay_controls=['brightness', 'fill_percent', 'average_species', 'sigma_nm', 'global_sigma_nm', 'normalize_overlap'],
        species=[dict(template=p['template'] or p['design_name'], code=p['code'],
                      accepted_origami=len(p['states']), rejected_candidates=p['rejected_candidates'],
                      mapping=dict(p['mapping']), channel=p['channel'],
                      missing=not len(p['states'])) for p in populations.values()],
        metrics=error_metrics(np.concatenate([p['occupancy'] for p in populations.values()]),
                              np.concatenate([p['targets'] for p in populations.values()])))
    report['programmed_endpoint_controls'] = control_rates(populations)
    result = dict(density=density, coverage=coverage, site_rgb=display_rgb(density), target=target,
                  populations=populations, report=report)
    rendered = gaussian_raster(populations, settings, rendering, progress)
    report['gaussian_rendering'] = rendered.pop('metadata')
    result.update(rendered)
    if overlay is not None:
        add_comparison(result, overlay, classifier)
    return result


def _canonical_plane(overlay, key, channel):
    values = overlay[key][:, :, channel]
    settings = overlay['report']['settings']
    if settings.get('mirror_y'):
        values = np.flipud(values)
    if settings.get('mirror_x'):
        values = np.fliplr(values)
    return np.rot90(values, int(settings.get('rotation_deg', 0)) // 90)


def _sample_overlay(overlay, cells, channel):
    """Area-average the unchanged density in the corresponding programmed pixel."""
    lower = np.array(overlay['report']['pre_rotation_origin_nm'])
    px = overlay['report']['output_pixel_nm']
    density = _canonical_plane(overlay, 'density', channel)
    coverage = _canonical_plane(overlay, 'coverage', channel)
    xs = lower[0] + (np.arange(density.shape[1]) + .5) * px
    ys = lower[1] + (np.arange(density.shape[0]) + .5) * px
    values = []
    for lo, hi in cells:
        x0, x1 = np.searchsorted(xs, [lo[0]-1e-9, hi[0]-1e-9])
        y0, y1 = np.searchsorted(ys, [lo[1]-1e-9, hi[1]-1e-9])
        patch = density[y0:y1, x0:x1]
        valid = coverage[y0:y1, x0:x1] > 0
        values.append(float(patch[valid].mean()) if valid.any() else np.nan)
    return np.asarray(values)


def add_comparison(result, overlay, classifier):
    reference = overlay['report']['display_reference_density']
    scale = classifier.overlay_density_per_on or max(reference, 1e-30)
    scaled = {}
    for key, p in result['populations'].items():
        values = _sample_overlay(overlay, p['cells'], p['channel']) / scale
        if not len(p['states']):
            values[:] = np.nan
        p['overlay_values'] = values
        scaled[key] = dict(p, on=np.nan_to_num(values), valid=np.isfinite(values).astype(int))
    settings = CompositeSettings(**result['report']['settings'])
    sampled, _, _, _ = _raster(scaled, settings)
    result['overlay'] = overlay
    result['overlay_sampled'] = sampled
    result['difference'] = result['density'] - sampled
    # The site comparison stays quantitative; the image comparison uses the
    # Gaussian forward rendering and the original overlay on the same nm grid.
    result['gaussian_overlay_sampled'] = overlay_on_gaussian_grid(
        overlay, settings, result['report']['gaussian_rendering'], max(reference, 1e-30))
    result['gaussian_difference'] = result['gaussian_signal'] - result['gaussian_overlay_sampled']
    result['report']['comparison'] = dict(
        scale='independent density per ON calibration' if classifier.overlay_density_per_on else 'display normalization only; not an occupancy calibration',
        density_scale=scale,
        sampling='Unchanged overlay density area-averaged within each programmed pixel; shared maximum normalization unless independently calibrated.',
        overlay_metrics=error_metrics(np.concatenate([p['overlay_values'] for p in result['populations'].values()]),
                                      np.concatenate([p['targets'] for p in result['populations'].values()])),
        quantitative=classifier.overlay_density_per_on is not None,
        gaussian_image_comparison='Gaussian occupancy forward signal minus shared-maximum-normalized overlay density, sampled at matching physical coordinates. Display-scale diagnostic, not an occupancy accuracy metric; no gamma or clipping before subtraction.')


def comparison_rows(result):
    rows = []
    for p in result['populations'].values():
        for j, site in enumerate(p['ids']):
            target, value = p['targets'][j], p['occupancy'][j]
            overlay = p.get('overlay_values', np.full(len(p['ids']), np.nan))[j]
            rows.append(dict(barcode=p['code'], position=site, origami_count=len(p['states']),
                programmed=target, occupancy=value, occupancy_error=value-target,
                overlay=overlay, overlay_error=overlay-target))
    return rows


def subsample_occupancy(result, sample_sizes=(10, 20, 50, 100, 200, 300), repeats=20, seed=0):
    """Whole-molecule resampling without replacement; never fill deficient classes.

    Return reconstructed vectors and error per barcode/repeat. A pixel raster can
    be regenerated with _raster using these vectors and the saved population cells.
    """
    if repeats < 1 or int(repeats) != repeats or any(int(n) != n or n <= 0 for n in sample_sizes):
        raise ValueError('Sample sizes and repeats must be positive integers.')
    rng = np.random.default_rng(seed)
    rows = []
    for size in sample_sizes:
        for repeat in range(repeats):
            for p in result['populations'].values():
                available = len(p['states'])
                if available < size:
                    rows.append(dict(barcode=p['code'], sample_size=size, repeat=repeat,
                                     available=available, status='insufficient', seed=seed))
                    continue
                selected = rng.choice(available, size, replace=False)
                summary = summarize_states(p['states'][selected])
                rows.append(dict(barcode=p['code'], sample_size=size, repeat=repeat, available=available,
                    status='ok', seed=seed, occupancy=summary['occupancy'].tolist(), valid=summary['valid'].tolist(),
                    **error_metrics(summary['occupancy'], p['targets'])))
    return rows


def _write_csv(path, rows, fields):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
                             for k, v in row.items()})


def export_occupancy(result, path, progress=None):
    """Lossless matrices plus tidy molecular/summary CSVs and fixed-scale previews."""
    from PIL import Image
    path = Path(path)
    Image.fromarray(result['rgb']).save(path)
    arrays = dict(occupancy=result['density'], valid=result['coverage'], missing=~np.isfinite(result['density']),
                  target=result['target'])
    Image.fromarray(result['site_rgb']).save(path.with_name(path.stem+'_sites.png'))
    for key in ('gaussian_density', 'gaussian_signal', 'gaussian_target_signal',
                'gaussian_overlay_sampled', 'gaussian_difference', 'overlay_sampled', 'difference'):
        if key in result:
            arrays[key] = result[key]
    arrays['gaussian_missing'] = ~np.isfinite(result['gaussian_density'])
    for index, p in enumerate(result['populations'].values()):
        for key in ('counts', 'states', 'reasons', 'candidate_ids', 'sites', 'cells', 'pixels', 'thresholds', 'targets', 'rejected_candidate_ids', 'localization_overlay', 'overlay_extent', 'gaussian_centers', 'gaussian_weight', 'grid_nm'):
            arrays[f'population_{index}_{key}'] = p[key]
        arrays[f'population_{index}_site_ids'] = np.asarray(p['ids'])
        arrays[f'population_{index}_barcode'] = np.asarray(p['code'])
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    def observations():
        for p in result['populations'].values():
            for n, candidate in enumerate(p['candidate_ids']):
                for j, site in enumerate(p['ids']):
                    yield dict(origami_id=f'{p["template"]}:{candidate}', template=p['template'], barcode=p['code'],
                        candidate_index=int(candidate), position=site, localizations=int(p['counts'][n, j]),
                        threshold=int(p['thresholds'][j]), state='' if p['states'][n, j] < 0 else int(p['states'][n, j]),
                        status='uncertain' if p['states'][n, j] < 0 else 'accepted', reason=p['reasons'][n, j],
                        qc_accepted=True, alignment_qc=json.dumps(p['qc'][n]))
    _write_csv(path.with_name(path.stem+'_individual.csv'), observations(),
        ['origami_id', 'template', 'barcode', 'candidate_index', 'position', 'localizations', 'threshold', 'state', 'status', 'reason', 'qc_accepted', 'alignment_qc'])
    def summaries():
        for p in result['populations'].values():
            for j, site in enumerate(p['ids']):
                yield dict(barcode=p['code'], position=site, usable_origami=len(p['states']),
                    on=int(p['on'][j]), valid=int(p['valid'][j]), excluded=int(p['excluded'][j]),
                    occupancy=p['occupancy'][j], standard_error=p['standard_error'][j], programmed=p['targets'][j])
    _write_csv(path.with_name(path.stem+'_summary.csv'), summaries(),
               ['barcode', 'position', 'usable_origami', 'on', 'valid', 'excluded', 'occupancy', 'standard_error', 'programmed'])
    rows = comparison_rows(result)
    _write_csv(path.with_name(path.stem+'_errors.csv'), rows, list(rows[0]))
    barcode_rows = []
    for p in result['populations'].values():
        for method, values in [('occupancy', p['occupancy']), ('overlay', p.get('overlay_values'))]:
            if values is not None:
                barcode_rows.append(dict(barcode=p['code'], method=method, origami_count=len(p['states']), **error_metrics(values, p['targets'])))
    _write_csv(path.with_name(path.stem+'_barcode_errors.csv'), barcode_rows, ['barcode', 'method', 'origami_count', 'n', 'mae', 'rmse'])
    _write_csv(path.with_name(path.stem+'_qc_exclusions.csv'),
        (dict(template=p['template'], candidate_index=int(index), status='not_accepted_for_this_template')
         for p in result['populations'].values() for index in p['rejected_candidate_ids']),
        ['template', 'candidate_index', 'status'])
    path.with_suffix('.json').write_text(json.dumps(result['report'], indent=2, allow_nan=False)+'\n')
    if 'overlay' in result:
        from origami_composite import export_composite
        export_composite(result['overlay'], path.with_name(path.stem+'_overlay.png'), progress)
    figure = occupancy_figure(result, 'Comparison' if 'overlay' in result else 'Composite')
    figure.savefig(path.with_name(path.stem+'_comparison.png'), dpi=160)
    figure.clear()
    if progress:
        progress('Saved occupancy matrices, individual and barcode tables, metrics, images and report.')


def occupancy_figure(result, view='Composite', barcode=None, position=0, figure=None, *,
                     show_grid=True, show_values=True, full_grid=False):
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    if figure is None:
        figure = Figure(figsize=(12, 7), layout='constrained')
    figure.clear()
    if figure.get_layout_engine() is not None:
        figure.get_layout_engine().set(rect=(0, 0, 1, 1))
    populations = result['populations']
    p = populations[barcode] if barcode in populations else next(iter(populations.values()))
    if view == 'Comparison':
        if 'overlay' not in result:
            figure.add_subplot(111).text(.1, .5, 'Choose Compare Both and rebuild.')
            return figure
        grid = figure.add_gridspec(2, 6)
        overlay_axis = figure.add_subplot(grid[0, :3])
        occupancy_axis = figure.add_subplot(grid[0, 3:])
        overlay_axis.imshow(result['overlay']['rgb'], interpolation='nearest')
        overlay_axis.set_title('Existing overlay · shared maximum scale', fontsize=10)
        occupancy_axis.imshow(result['rgb'], interpolation='nearest')
        occupancy_axis.set_title('Occupancy · Gaussian rendering; missing = magenta', fontsize=10)
        axes = [figure.add_subplot(grid[1, 2*c:2*c+2]) for c in range(3)]
        for c, ax in enumerate(axes):
            im = ax.imshow(result['gaussian_difference'][:, :, c], cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(('Red', 'Green', 'Blue')[c] + ' · Gaussian occupancy − normalized overlay', fontsize=9)
        figure.colorbar(im, ax=axes, shrink=.8)
    elif view == 'Individual sites':
        examples = p['examples']
        axes = figure.subplots(2, 3).ravel()
        for ax, example in zip(axes, examples[:6]):
            n = example['index']
            points = example['points']
            ax.scatter(points[:, 0], points[:, 1], s=2, color='black', alpha=.3)
            for j, (x, y) in enumerate(p['sites']):
                color = {-1: 'orange', 0: 'gray', 1: 'tab:blue'}[int(p['states'][n, j])]
                ax.add_patch(Circle((x, y), result['report']['occupancy_settings']['radius_nm'], fill=False, color=color))
                ax.text(x, y, str(p['counts'][n, j]), fontsize=6, ha='center', color=color)
            ax.set_aspect('equal'); ax.autoscale_view()
            ax.set_title(f'{p["code"]} / {p["candidate_ids"][n]}')
        figure.suptitle('Aligned nm · counts · blue ON, gray OFF, orange uncertain')
    elif view == 'Count distributions':
        ax, controls = figure.subplots(1, 2)
        for target, label in [(0, 'Programmed OFF'), (1, 'Programmed ON')]:
            blocks = [pop['counts'][:, pop['targets'] == target].ravel() for pop in populations.values()]
            values = np.concatenate(blocks)
            if len(values):
                controls.hist(values, bins=min(50, max(1, len(np.unique(values)))), alpha=.5, label=label)
        controls.set(xlabel='Localizations per site/origami', ylabel='Observations', title='Available programmed endpoint controls')
        if controls.has_data():
            controls.legend()
        position = max(0, min(int(position), len(p['ids'])-1))
        values = p['counts'][:, position]
        ax.hist(values, bins=min(50, max(1, len(np.unique(values)))))
        ax.axvline(p['thresholds'][position], color='red', label='ON threshold')
        ax.set(xlabel='Localizations per origami', ylabel='Origami', title=f'{p["code"]}: {p["ids"][position]}')
        ax.legend()
    elif view in ('Template heatmaps', 'Barcode maps'):
        draw_template_heatmaps(figure, p, CompositeSettings(**result['report']['settings']),
            result['report']['gaussian_rendering'], show_grid=show_grid,
            show_values=show_values, show_target=view == 'Barcode maps', full_grid=full_grid)
    elif view == 'Target errors':
        rows = comparison_rows(result)
        axes = figure.subplots(1, 3)
        for method, marker in [('occupancy', 'o'), ('overlay', 'x')]:
            x = np.array([r['programmed'] for r in rows]); y = np.array([r[method] for r in rows])
            axes[0].scatter(x, y, s=8, marker=marker, label=method)
            axes[1].scatter(x, y-x, s=8, marker=marker)
            axes[2].scatter([r['origami_count'] for r in rows], abs(y-x), s=8, marker=marker)
        axes[0].plot([0, 1], [0, 1], color='gray'); axes[0].legend()
        axes[0].set(xlabel='Programmed fraction', ylabel='Measured value')
        axes[1].set(xlabel='Programmed fraction', ylabel='Signed error')
        axes[2].set(xlabel='Origami per barcode', ylabel='Absolute error')
        figure.suptitle('Overlay errors are display-scale diagnostics unless independently calibrated')
    else:
        ax = figure.add_subplot(111)
        image = result['rgb'] if view == 'Composite' else result['coverage'].sum(axis=2)
        if view == 'Site occupancy':
            image = result['site_rgb']
        if view in ('Red', 'Green', 'Blue'):
            image = result['gaussian_signal'][:, :, ('Red', 'Green', 'Blue').index(view)]
        im = ax.imshow(image, interpolation='nearest', vmin=0 if image.ndim == 2 and view != 'Coverage' else None,
                       vmax=1 if image.ndim == 2 and view != 'Coverage' else None)
        if image.ndim == 2:
            figure.colorbar(im, ax=ax)
        label = ('Gaussian spots · saved picklist display scale' if view in ('Composite', 'Red', 'Green', 'Blue')
                 else ('ON / valid · fixed 0–1' if view == 'Site occupancy' else 'Valid molecular observations'))
        ax.set_title(view + ' · ' + label + '; missing RGB = magenta')
    return figure


def control_rates(populations):
    """Empirical rates at programmed endpoints, when those controls are present.

    These are endpoint diagnostics, not proof that assembly realized the target.
    """
    totals = dict(off_valid=0, off_on=0, on_valid=0, on_off=0)
    for p in populations.values():
        off, on = p['targets'] == 0, p['targets'] == 1
        totals['off_valid'] += int(p['valid'][off].sum())
        totals['off_on'] += int(p['on'][off].sum())
        totals['on_valid'] += int(p['valid'][on].sum())
        totals['on_off'] += int((p['valid'][on] - p['on'][on]).sum())
    return dict(**totals,
        false_positive_fraction=totals['off_on'] / totals['off_valid'] if totals['off_valid'] else None,
        false_negative_fraction=totals['on_off'] / totals['on_valid'] if totals['on_valid'] else None,
        meaning='Programmed 0/1 endpoint diagnostics; interpretation as detection error requires independently validated OFF/ON controls.')


def _json_finite(value):
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def export_subsampling(result, path, sample_sizes=(10, 20, 50, 100, 200, 300), repeats=20, seed=0, progress=None):
    """Save all repeat vectors and a reconstructed image for every size/repeat."""
    from PIL import Image
    from matplotlib.figure import Figure
    path = Path(path)
    rows = subsample_occupancy(result, sample_sizes, repeats, seed)
    folder = path.with_name(path.stem + '_images')
    folder.mkdir(exist_ok=True)
    settings = CompositeSettings(**result['report']['settings'])
    all_metrics = []
    for size in sample_sizes:
        for repeat in range(repeats):
            sampled = {}
            selected = {row['barcode']: row for row in rows if row['sample_size'] == size and row['repeat'] == repeat}
            for key, population in result['populations'].items():
                row = selected[population['code']]
                valid = np.asarray(row.get('valid', np.zeros(len(population['ids']))), int)
                values = np.asarray(row.get('occupancy', np.full(len(population['ids']), np.nan)))
                sampled[key] = dict(population, valid=valid, on=np.nan_to_num(values) * valid, occupancy=values)
            density, valid, target, _ = _raster(sampled, settings)
            stem = folder / f'n{size}_repeat{repeat:03d}'
            rendered = gaussian_raster(sampled, settings, result['report']['gaussian_rendering'])
            Image.fromarray(rendered['rgb']).save(stem.with_suffix('.png'))
            np.savez_compressed(stem.with_suffix('.npz'), occupancy=density, valid=valid, missing=~np.isfinite(density),
                gaussian_density=rendered['gaussian_density'], gaussian_signal=rendered['gaussian_signal'],
                gaussian_missing=~np.isfinite(rendered['gaussian_density']))
            all_metrics.append(dict(sample_size=size, repeat=repeat,
                measured_barcodes=sum(row['status'] == 'ok' for row in selected.values()),
                **error_metrics(density, target)))
        if progress:
            progress(f'Sample-count study: N={size}, {repeats} seeded reconstructions saved')
    path.write_text(json.dumps(_json_finite(dict(seed=seed, repeats=repeats, sample_sizes=list(sample_sizes),
        policy='Sample whole accepted molecules without replacement. Classes with fewer than N molecules are missing; valid counts can be smaller than N due to uncertainty. Compare metrics alongside measured barcode coverage.',
        observations=rows, image_metrics=all_metrics, gaussian_rendering=result['report']['gaussian_rendering'])), indent=2, allow_nan=False)+'\n')
    figure = Figure(figsize=(10, 4), layout='constrained')
    axes = figure.subplots(1, 2)
    for axis, metric in zip(axes, ('mae', 'rmse')):
        for size in sample_sizes:
            values = [row[metric] for row in all_metrics if row['sample_size'] == size and row[metric] is not None]
            if values:
                axis.errorbar(size, np.mean(values), yerr=np.std(values), fmt='o', color='tab:blue')
        axis.set(xlabel='Origami sampled per barcode', ylabel=metric.upper(), title='Mean ± repeat SD; missing classes excluded')
    figure.savefig(path.with_suffix('.png'), dpi=160)
    figure.clear()
    return rows
