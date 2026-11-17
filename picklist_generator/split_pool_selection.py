"""Selection overlays and spatial views of one design's synthesis route."""
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib import patheffects


def selected_design(designs, color, row):
    group = [design for design in designs or [] if design['color'] == color]
    return group[max(0, min(len(group)-1, row))] if group else None


def highlight_image(ax, design, image):
    """Match the Gaussian renderer's field bounds and clockwise rotation."""
    from image_encoding import geometry, DEFAULT_SCHEME
    rotation = design.get('origami_rotation_degrees_clockwise', 0)
    info = geometry(design.get('encoding_scheme', DEFAULT_SCHEME), rotation, design.get('active_image_columns'))
    width, height = image.size
    upright_w, upright_h = (height, width) if rotation in (90, 270) else (width, height)
    dx = info['field_width_nm']/upright_w; dy = info['field_height_nm']/upright_h
    for site in design['image_sites']:
        x = (site['x_nm']+info['pixel_pitch_x_nm']/2)/dx-.5
        y = (site['y_nm']+info['pixel_pitch_y_nm']/2)/dy-.5
        w, h = info['pixel_pitch_x_nm']/dx, info['pixel_pitch_y_nm']/dy
        if rotation == 90: x, y, w, h = upright_h-1-y, x, h, w
        elif rotation == 180: x, y = upright_w-1-x, upright_h-1-y
        elif rotation == 270: x, y, w, h = y, upright_w-1-x, h, w
        patch = Rectangle((x-w/2, y-h/2), w, h, fill=False, edgecolor='white', linewidth=1)
        patch.set_path_effects([patheffects.Stroke(linewidth=3, foreground='black'), patheffects.Normal()])
        ax.add_patch(patch)


def draw_selected_origami(figure, design, result, target_row):
    from image_encoding import ROWS, site_name, rotate_point
    from split_pool_vectors import draw_site_canvas
    axes = figure.subplots(1, 2)
    if design is None:
        for ax in axes: ax.axis('off')
        axes[0].text(.5, .5, 'Select an RGB design to inspect its origami.', ha='center')
        return
    row = int(np.flatnonzero(np.asarray(result['assignment']) == target_row)[0])
    rgb = np.ones(3) if design.get('image_mode') == 'grayscale' else np.eye(3)[design['channel']]
    rotation = design.get('origami_rotation_degrees_clockwise', 0)
    ax = axes[0]; ax.set_facecolor('#15191d')
    for c in range(12):
        column = f'{12-c:02d}'; x = c*9.9+(10 if c >= 6 else 0)
        for r, letter in enumerate(ROWS):
            px, py = rotate_point(x, r*5.1, 118.9, 35.7, rotation)
            value = design['site_fractions'].get(site_name(letter, column), 0)
            shade = np.array([21, 25, 29])/255*(1-value)+rgb*value
            fiducial = [letter, column] in design.get('always_on_sites', [])
            ax.scatter(px, py, s=65, c=[shade], edgecolors='#ffd166' if fiducial else '#748087', linewidths=1.5 if fiducial else .6)
    for c in range(12):
        px, py = rotate_point(c*9.9+(10 if c >= 6 else 0), -6.5, 118.9, 35.7, rotation)
        ax.text(px, py, f'{12-c:02d}', color='white', fontsize=7, ha='center')
    for r, letter in enumerate(ROWS):
        px, py = rotate_point(-8, r*5.1, 118.9, 35.7, rotation)
        ax.text(px, py, letter, color='white', fontsize=7, ha='center')
    ax.margins(x=.2, y=.25); ax.set_aspect('equal'); ax.invert_yaxis(); ax.set(xticks=[], yticks=[])
    original_code = ''.join(map(str, design.get('bits', []))) or 'unspecified'
    ax.set_title(f"{design['name']}\nOriginal origami • barcode {original_code}", fontsize=10)
    draw_site_canvas(axes[1], result['prediction'][row], design['image_sites'], rotation, vmin=0, vmax=1, cmap='viridis')
    address = int(result['address_indices'][row]); code = format(address, f"0{result['address_bits']}b")
    axes[1].set(title=f'Fitted analog canvas\nSplit–pool address {code}', xticks=[], yticks=[])
    if result.get('synthesis_mode') == 'independent_wells':
        axes[1].set_title('Fitted analog canvas • independent well')


def draw_evolution(figure, result, target_row, sites=None, vector_index=1, rotation=0, *, round_specs=None, design=None):
    from split_pool_canvas_build import canvas_build_steps
    from split_pool_vectors import draw_mixed_canvas
    plan = canvas_build_steps(result, target_row)
    if plan is None:
        ax = figure.subplots(); ax.axis('off'); ax.text(.5, .5, 'Fit ON-only or ON/OFF to inspect the synthesis path.', ha='center')
        return
    mixture_scale = max((float(event['mixture'].sum(axis=-1).max()) for event in plan['events']), default=1.)
    captures = [event for event in plan['events'] if event['phase'] == 'capture']
    groups = [[event for event in plan['events'] if event['round'] == cap['round'] and event['phase'] == 'addition'] for cap in captures]
    addition_columns = min(4, max(1, max(map(len, groups))))
    columns = addition_columns+1
    grid = figure.add_gridspec(len(captures), columns) if round_specs is None else None
    for r, (cap, additions) in enumerate(zip(captures, groups)):
        subrows = max(1, (len(additions)+addition_columns-1)//addition_columns)
        row_grid = round_specs[r].subgridspec(subrows, columns) if round_specs is not None else grid[r, :].subgridspec(subrows, columns)
        prefix = plan['code'][:r+1]
        label = 'Single capture' if plan['independent'] else f"Round {r+1}: D{r+1}={plan['code'][r]} ({'ON' if plan['code'][r] == '1' else 'OFF'})\nBarcode so far: {prefix}"
        if not additions:
            empty = figure.add_subplot(row_grid[:, :addition_columns]); empty.axis('off')
            empty.text(.5, .5, 'No K additions\n0 nL transferred' if plan['physical'] else 'No K additions',
                       ha='center', va='center', fontsize=8)
        for index, event in enumerate([*additions, cap]):
            cumulative = index == len(additions)
            cell = row_grid[:, -1] if cumulative else row_grid[index//addition_columns, index%addition_columns]
            ax = figure.add_subplot(cell)
            on, off = (event['on_after'], event['off_after']) if cumulative else (event['added'][:, 0], event['added'][:, 1])
            draw_mixed_canvas(ax, on, off, sites, rotation, scale=1 if cumulative else mixture_scale)
            if cumulative and not plan['independent']:
                draw_barcode_progress(ax, design, plan['code'], r+1)
            amount = f"{event['amount']:g} nL" if plan['physical'] else f"weight {event['amount']:.3g}"
            title = 'Accumulated pattern\nafter capture' if cumulative else f"+ K{event['vector']}\n{amount}\nPremixed ON + OFF"
            if index == 0 and round_specs is None: title = label+'\n'+title
            ax.set(title=title, xticks=[], yticks=[]); ax.title.set_fontsize(8)
    if round_specs is not None:
        return
    heading = 'Independent well: mix K additions → capture once' if plan['independent'] else 'Selected route: K additions → accumulated ON/OFF pattern'
    digital_note = '' if plan['independent'] else 'Barcode overlay is a proposed site schedule; digital additions are not simulated.\n'
    figure.suptitle(heading+'\n'+digital_note+f"All nonzero K additions are shown for each round.\nCyan = ON; orange = OFF; dark = no dose / unoccupied. Mixed-dose scale: {mixture_scale:.3g} {plan['unit']}; occupancy scale: 1.", fontsize=9)


def draw_barcode_progress(ax, design, code, completed, *, caption='Planned barcode', progress=True):
    """Overlay a proposed digital schedule, not a simulated chemical result."""
    from image_encoding import ROWS, rotate_point
    groups = design.get('barcode_sites', []) if design else []
    if not groups or len(groups) != len(code) or any(not group for group in groups):
        ax.text(.5, -.12, f'Barcode site mapping unavailable for {len(code)} rounds',
                transform=ax.transAxes, ha='center', va='top', fontsize=7)
        return False
    analog = {(s.get('row'), str(s.get('column', '')).zfill(2)) for s in design['image_sites']}
    fiducials = {tuple(site) for site in design.get('always_on_sites', [])}
    used = set()
    for group in groups:
        for row, column in group:
            key = (row, str(column).zfill(2))
            if row not in ROWS or key[1] not in {f'{c:02d}' for c in range(1, 13)} or key in used or key in analog or key in fiducials:
                ax.text(.5, -.12, 'Barcode layout conflicts with analog / reserved sites',
                        transform=ax.transAxes, ha='center', fontsize=7)
                return False
            used.add(key)
    rotation = design.get('origami_rotation_degrees_clockwise', 0)
    for bit, group in enumerate(groups):
        positions = []
        for row, column in group:
            c = 12-int(column)
            positions.append(rotate_point(c*9.9+(10 if c >= 6 else 0), ROWS.index(row)*5.1,
                                          118.9, 35.7, rotation))
        if not positions:
            continue
        x, y = np.asarray(positions).T
        state = code[bit] if bit < completed else None
        fill = '#00ccff' if state == '1' else '#ff730d' if state == '0' else 'none'
        artist = ax.scatter(x, y, s=18, facecolors=fill,
                            edgecolors='#d48b00' if progress and bit == completed-1 else '#748087',
                            linewidths=1.4 if progress and bit == completed-1 else .6)
        artist.set_gid(f'barcode-bit-{bit+1}-{state if state is not None else "pending"}')
    note = 'Gold outline: this round; hollow: future bits' if progress else 'Cyan = 1 (ON); orange = 0 (OFF)'
    caption_text = caption+': '+code[:completed]+'·'*(len(code)-completed)
    if not progress:
        caption_text += '\n'+note
    ax.text(.5, -.12, caption_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=7)
    return True


def draw_selection_comparison(figure, image, design, result, target_row, *, result_image=None):
    """Keep the image context, spatial goal/result, and site-level comparison together."""
    from split_pool_vectors import draw_site_canvas
    has_result_image = result_image is not None
    transfer_lines = selected_transfer_lines(result, target_row) if has_result_image else []
    grid = figure.add_gridspec(3 if has_result_image else 2, 4 if has_result_image else 3,
                               height_ratios=[1, 1, max(.55, .12*len(transfer_lines))] if has_result_image else [1, 1],
                               width_ratios=[1.3, 1.3, 1, 1] if has_result_image else [1.4, 1, 1])
    detail_offset = int(has_result_image)
    overview = figure.add_subplot(grid[:2, 0])
    overview.imshow(image); overview.axis('off')
    overview.set_title(f"Current image targets\nSelected: {design['name']}", fontsize=10)
    highlight_image(overview, design, image)
    if has_result_image:
        whole_result = figure.add_subplot(grid[:2, 1], sharex=overview, sharey=overview)
        whole_result.imshow(result_image); whole_result.axis('off')
        title = ('Whole-image PCA reference' if result['model'] == 'pca' else
                 'Whole-image independent-well result' if result.get('synthesis_mode') == 'independent_wells' else
                 'Whole-image split–pool result')
        whole_result.set_title(title+'\nExpected reconstruction', fontsize=10)
        highlight_image(whole_result, design, result_image)
    sites = design['image_sites']
    target = np.array([design['site_fractions'][site['name']] for site in sites])
    assigned = int(np.flatnonzero(np.asarray(result['assignment']) == target_row)[0])
    prediction = np.asarray(result['prediction'])[assigned]
    rotation = design.get('origami_rotation_degrees_clockwise', 0)
    result_label = ('PCA reference' if result['model'] == 'pca' else
                    'Independent-well result' if result.get('synthesis_mode') == 'independent_wells' else
                    'Expected result after split pooling')
    groups = design.get('barcode_sites', [])
    original_code = ''.join(map(str, design.get('bits', [])))
    if not original_code and groups and 'barcode_code' in design:
        original_code = format(int(design['barcode_code']), f'0{len(groups)}b')
    assigned_code = format(int(result['address_indices'][assigned]), f"0{result['address_bits']}b")
    for col, values, title in ((1, target, 'Goal • ideal selected canvas'),
                               (2, prediction, result_label)):
        ax = figure.add_subplot(grid[0, col+detail_offset])
        draw_site_canvas(ax, values, sites, rotation, vmin=0, vmax=1, cmap='viridis')
        use_original = col == 1 or result['model'] == 'pca' or result.get('synthesis_mode') == 'independent_wells'
        code = original_code if use_original else assigned_code
        caption = 'Original barcode' if use_original else 'Planned barcode (not simulated)'
        if col == 2 and use_original:
            caption = 'Original barcode (not simulated)'
        draw_barcode_progress(ax, design, code, len(code), caption=caption, progress=False)
        ax.set_title(title+'\nAnalog ON occupancy: 0–1', fontsize=9)
    comparison = figure.add_subplot(grid[1:, 1+detail_offset:])
    x = np.arange(len(target))
    comparison.plot(x, target, 'o-', label='Goal', markersize=3)
    comparison.plot(x, prediction, 'o-', label='Result' if result['model'] != 'pca' else 'PCA reference', markersize=3)
    ticks = np.unique(np.linspace(0, len(target)-1, min(8, len(target)), dtype=int))
    comparison.set(xticks=ticks, xticklabels=[sites[i]['name'] for i in ticks],
                   xlabel='Analog site', ylabel='ON occupancy',
                   title=f"Selected analog canvas • RMSE {np.sqrt(np.mean((prediction-target)**2)):.4f}")
    comparison.tick_params(axis='x', labelsize=7, rotation=30)
    comparison.title.set_fontsize(10)
    if result['model'] != 'pca':
        comparison.set_ylim(-.05, 1.05)
    comparison.legend(fontsize=8); comparison.grid(alpha=.2)
    if has_result_image:
        transfers = figure.add_subplot(grid[2, :2]); transfers.axis('off')
        transfers.set_title('Selected segment • branch-well K transfers (nL)', loc='left', fontsize=9)
        transfers.text(0, 1, '\n'.join(transfer_lines), transform=transfers.transAxes,
                       ha='left', va='top', fontsize=8, linespacing=1.4)


def selected_transfer_lines(result, target_row):
    """Exact physical additions along the selected target's assigned route."""
    import textwrap
    recipe = result.get('recipe')
    if recipe is None:
        return ['No physical K-transfer recipe for this fit.']
    assigned = int(np.flatnonzero(np.asarray(result['assignment']) == target_row)[0])
    independent = result.get('synthesis_mode') == 'independent_wells'
    code = format(int(result['address_indices'][assigned]), f"0{result['address_bits']}b")
    destinations = [assigned] if independent else [2*r+int(bit) for r, bit in enumerate(code)]
    lines = []
    for r, destination in enumerate(destinations):
        amounts = np.asarray(recipe['branch_transfers_nl'])[destination]
        additions = '; '.join(f'K{k+1}: {volume:,.0f} nL' for k, volume in enumerate(amounts) if volume > 0)
        label = 'Destination well' if independent else f'Round {r+1} · branch {code[r]}'
        lines.extend(textwrap.wrap(label+' — '+(additions or 'No K additions (0 nL)'),
                                   width=70, subsequent_indent='    ', break_long_words=False))
    return lines
