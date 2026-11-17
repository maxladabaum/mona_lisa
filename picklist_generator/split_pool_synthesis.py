"""Logical split/pool workflow; liquid carryover is deliberately not inferred."""
import numpy as np
from matplotlib.patches import FancyBboxPatch


def draw_synthesis(figure, color_fits, selected_result=None, target_row=0, *, subplot_ax=None, round_index=None, show_title=True, vertical=False):
    """Show one shared split–pool population, or independent destination wells."""
    selected_code = None
    if selected_result is not None and selected_result.get('synthesis_mode') != 'independent_wells' and selected_result.get('model') != 'pca':
        assigned = int(np.flatnonzero(np.asarray(selected_result['assignment']) == target_row)[0])
        selected_code = format(int(selected_result['address_indices'][assigned]), f"0{selected_result['address_bits']}b")
    lanes = [(color, fit) for color, fit in color_fits.items() if fit.get('recipe') is not None]
    if not lanes:
        ax = figure.subplots(); ax.axis('off')
        ax.text(.5, .5, 'Fit ON-only or ON/OFF with 25 nL recipes enabled\nto see the full synthesis workflow.', ha='center', va='center')
        return
    independent = lanes[0][1].get('synthesis_mode') == 'independent_wells'
    if not independent:
        shared = lanes[0][1].get('shared_pool_fit')
        if shared is not None and all(fit.get('shared_pool_fit') is shared for _, fit in lanes):
            lanes = [('All RGB designs', shared)]
        elif len(lanes) > 1:
            ax = figure.subplots(); ax.axis('off')
            ax.text(.5, .5, 'These results used separate color fits.\nClick Visualize analysis to fit one shared RGB population.',
                    ha='center', va='center')
            return
    axes = [subplot_ax] if subplot_ax is not None else figure.subplots(len(lanes), 1, squeeze=False)[:, 0]
    for ax, (color, fit) in zip(axes, lanes):
        recipe = fit['recipe']; rounds = fit['address_bits']
        amounts = np.asarray(recipe['branch_transfers_nl'])
        if fit.get('synthesis_mode') == 'independent_wells':
            im = ax.imshow(amounts, aspect='auto', interpolation='nearest', vmin=0)
            ax.set(title=f'{color}: prepare K stocks → {len(amounts)} separate design wells → capture once → pool completed origami',
                   xlabel='Source K-vector index (0-based)', ylabel='Destination design index')
            figure.colorbar(im, ax=ax, label='Volume added (nL)')
            def coordinates(x, y, amounts=amounts, color=color):
                k, row = int(np.floor(x+.5)), int(np.floor(y+.5))
                if 0 <= row < len(amounts) and 0 <= k < amounts.shape[1]:
                    return f'{color}: K{k+1} → design {row}: {amounts[row,k]:g} nL; mix all additions before capture'
                return ''
            ax.format_coord = coordinates
            continue
        nodes = []
        def point(x, y):
            return (y, -x) if vertical else (x, y)
        def line(start, end):
            a, b = point(*start), point(*end)
            ax.plot([a[0], b[0]], [a[1], b[1]], '--', color='#7b8fa2', linewidth=.8)
        def box(x, y, text, detail, fill, width=.84, height=.65, selected=False):
            x, y = point(x, y)
            if vertical:
                width, height = max(height, 1.15), width
                if fill in ('#dcefe3', '#fce5ce'):
                    height = max(height, .18*len(text.splitlines()))
            ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                boxstyle='round,pad=0.03', facecolor=fill, edgecolor='#d48b00' if selected else '#425466', linewidth=2.5 if selected else .7))
            ax.text(x, y, text, ha='center', va='center', fontsize=6.5 if vertical else 7, linespacing=1.0)
            nodes.append((x-width/2, x+width/2, y-height/2, y+height/2, detail))
        def arrow(start, end, dashed=False, selected=False):
            ax.annotate('', xy=point(*end), xytext=point(*start), arrowprops=dict(arrowstyle='->', color='#d48b00' if selected else '#526578',
                        linewidth=2.5 if selected else .85, linestyle='--' if dashed else '-'))
        k = amounts.shape[1]; settings = recipe['settings']
        max_additions = int(np.count_nonzero(amounts, axis=1).max())
        branch_height = max(1.15 if vertical else .9, .22*(max_additions+3))
        branch_y = branch_height/2 + .4
        stock_y = 2*branch_y + .6
        prep = []
        for index, transfers in enumerate(recipe['source_transfers_nl']):
            prep.append(f"K{index+1}: ON {transfers[:,0].sum():g} + OFF {transfers[:,1].sum():g} + buffer {recipe['vector_buffer_nl'][index]:g} nL")
        prep_label = f'ON/OFF\nstocks\n{k} K wells\n+ buffer' if vertical else f'ON + OFF stocks\n→ {k} shared K wells\n+ dilution buffer'
        box(-1, stock_y, prep_label,
            f'{color} preparation: '+ '; '.join(prep), '#d9eaf7', width=1.8, height=.9)
        displayed_rounds = rounds if round_index is None else 1
        line((0, stock_y), (2*displayed_rounds-1, stock_y))
        if not vertical:
            ax.text(displayed_rounds, stock_y+.25, 'Prepared K stocks supply each branch (dashed arrows)', ha='center', fontsize=7)
        start_label = f'Starting pool\n{len(fit["assignment"])} designs' if not round_index else f'Pool {round_index}\nresplit'
        if vertical and not round_index:
            start_label = f'Starting\npool\n{len(fit["assignment"])} designs'
        box(-1, 0, start_label, f'{color}: one shared population; initial population volume is not specified', '#eeeeee', width=1.5)
        for r in (range(rounds) if round_index is None else [round_index]):
            x = 2*r+1 if round_index is None else 1
            previous = (-1 if r == 0 else 2*r) if round_index is None else -1
            for bit, y in ((0, branch_y), (1, -branch_y)):
                volumes = amounts[2*r+bit]
                detail = ', '.join(f'K{i+1}: {v:g} nL' for i, v in enumerate(volumes) if v > 0) or 'No K additions'
                remaining = recipe['branch_remaining_nl'][2*r+bit]
                additions = '\n'.join(f'K{i+1}: {v:g} nL' for i, v in enumerate(volumes) if v > 0) or 'No K additions'
                chosen = selected_code is not None and int(selected_code[r]) == bit
                digital = f'D{r+1}={bit} ('+('ON' if bit else 'OFF')+')'
                prefix = f'\nCode: {selected_code[:r+1]}' if chosen else ''
                branch_label = f'D{r+1}={bit}'+'\n'+additions if vertical else f'R{r+1} · bit {bit}\nDigital {digital}{prefix}\n{additions}'
                box(x, y, branch_label,
                    f'{color} Round {r+1} branch {bit} | {detail} | remaining {remaining:g} nL | final recipe {settings["branch_volume_nl"]:g} nL',
                    '#dcefe3' if bit == 0 else '#fce5ce', height=branch_height, selected=chosen)
                arrow((previous+.43, 0), (x-.45, y), selected=chosen)
                arrow((x+.45, y), (x+.78, 0), selected=chosen)
                # Route stock feeds outside the boxes so they do not cross recipe labels.
                feed_x = x-.55
                line((feed_x, stock_y), (feed_x, y))
                arrow((feed_x, y), (x-.45, y), dashed=True)
            final = r == rounds-1
            box(x+1, 0, 'Final\npool' if final else f'Pool {r+1}',
                f'{color}: combine both branches after round {r+1}'+('' if final else '; mix, then resplit into both branches next round')+
                '. Transfer recovery, wash/removal, carryover, and split volumes are not specified.', '#e6dff4', width=.4, height=.55)
            if not final and round_index is None:
                ax.text(x+1.5, -.12, 'resplit', fontsize=6, ha='center')
        ax.set(xlim=(-2.05, 2*displayed_rounds+.65), ylim=(-branch_y-branch_height/2-.25, stock_y+.65),
               title=f'{color}: {len(fit["assignment"])} assigned designs in one pool • {rounds} rounds • {2*rounds} shared branch wells'+(f' • selected route {selected_code} (gold)' if selected_code else ''))
        if vertical:
            ax.set(xlim=(-branch_y-branch_height/2-.25, stock_y+.65),
                   ylim=(-2*displayed_rounds-.65, 2.05))
        if round_index is not None:
            title = f'Round {round_index+1}'
            if selected_code:
                title += ('\n' if vertical else ' • ')+f'selected branch {selected_code[round_index]}'
            ax.set_title('')
            title_x = .5
            if vertical:
                left, right = ax.get_xlim()
                # The inter-round connector stays at data x=0. Put the
                # left-aligned title to its right, including a fixed gap.
                title_x = (0-left)/(right-left)+.06
            ax.set_title(title, fontsize=9 if vertical else 10,
                         loc='left' if vertical else 'center', x=title_x)
        ax.axis('off')
        def coordinates(x, y, nodes=nodes):
            for left, right, bottom, top, detail in nodes:
                if left <= x <= right and bottom <= y <= top:
                    return detail
            return 'Hover a well for its recipe. Solid arrows: population flow. Dashed arrows: K-stock additions.'
        ax.format_coord = coordinates
    independent = lanes[0][1].get('synthesis_mode') == 'independent_wells'
    if not show_title:
        return
    figure.suptitle('Independent wells • barcode each design separately; pool only after assembly/capture' if independent else
                   'Full synthesis • split → add K stocks + planned digital bit → pool → resplit\nDigital labels identify address bits; DNA sequences are not assigned.', fontsize=12)


def draw_combined_synthesis(figure, color_fits, selected_result, target_row, sites=None, vector_index=1, rotation=0, *, design=None):
    """Align each split/pool stage with its K contributions and captured pattern."""
    from split_pool_canvas_build import canvas_build_steps
    from split_pool_selection import draw_evolution
    plan = canvas_build_steps(selected_result, target_row)
    if plan is None or selected_result.get('recipe') is None:
        draw_synthesis(figure, color_fits, selected_result, target_row)
        return
    if plan['independent']:
        top, bottom, volumes = figure.subfigures(3, 1, height_ratios=[1, 2, 1])
        draw_synthesis(top, color_fits, selected_result, target_row)
        draw_evolution(bottom, selected_result, target_row, sites, vector_index, rotation)
        draw_selected_addition_histogram(volumes.subplots(), selected_result, target_row)
        return
    rounds = selected_result['address_bits']
    grid = figure.add_gridspec(rounds, 3, width_ratios=[1.35, 3.1, .55])
    from matplotlib.patches import ConnectionPatch
    previous_ax = None
    for r in range(rounds):
        ax = figure.add_subplot(grid[r, 0])
        draw_synthesis(figure, color_fits, selected_result, target_row,
                       subplot_ax=ax, round_index=r, show_title=False, vertical=True)
        if previous_ax is not None:
            connection = ConnectionPatch((0, -2.23), (0, 1.78), 'data', 'data',
                                         axesA=previous_ax, axesB=ax, arrowstyle='->',
                                         color='#d48b00', linewidth=2.5, clip_on=False)
            connection.set_in_layout(False)
            figure.add_artist(connection)
        previous_ax = ax
    draw_evolution(figure, selected_result, target_row, sites, vector_index, rotation,
                   round_specs=[grid[r, 1] for r in range(rounds)], design=design)
    draw_selected_addition_histogram(figure.add_subplot(grid[:, 2]), selected_result, target_row, vertical_timeline=True)
    mixture_scale = max(float(event['mixture'].sum(axis=-1).max()) for event in plan['events'])
    figure.suptitle(f"Synthesis and pooling • route {plan['code']} • all active K additions\n"
                   "Cyan: ON · Orange: OFF · Dark: empty · Gold: selected route / new barcode bit\n"
                   f"Premix dose scale: {mixture_scale:.3g} {'nM' if plan['physical'] else 'weight'}; occupancy: 0–1. Captured patterns include all K.\n"
                   "Barcode overlay is a proposed schedule; digital chemistry and volumes are not simulated.", fontsize=9)


def draw_addition_histogram(ax, amounts, remaining, labels, branch_volume, *, vertical_timeline=False):
    """Stack every delivered K stock, independent of the canvas page."""
    from matplotlib import colormaps
    x = np.arange(len(amounts)); bottom = np.zeros(len(amounts))
    active = np.flatnonzero(amounts.sum(axis=0) > 0)
    colors = colormaps['turbo']
    def bars(values, offset, **style):
        if vertical_timeline:
            return ax.barh(x, values, left=offset, height=.14, **style)
        return ax.bar(x, values, bottom=offset, **style)
    for k in active:
        bars(amounts[:, k], bottom, color=colors(k/max(amounts.shape[1]-1, 1)), label=f'K{k+1}')
        bottom += amounts[:, k]
    bars(remaining, bottom, color='#dddddd', hatch='//', label='Remaining' if vertical_timeline else 'Remaining volume')
    for r, volume in enumerate(bottom):
        ax.annotate(f'{volume:g} nL' if vertical_timeline else f'{volume:g} nL K mix', (branch_volume/2, r-.10) if vertical_timeline else (r, branch_volume),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)
    if vertical_timeline:
        ax.set(yticks=x, yticklabels=labels, xlabel='nL',
               title='K additions',
               xlim=(0, branch_volume*1.05), ylim=(len(amounts)-.5, -.5))
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(2))
        ax.tick_params(axis='both', labelsize=7)
        ax.set_yticklabels([label.replace('Round ', 'R').replace('\nbranch ', ' · ') for label in labels])
        ax.title.set_fontsize(9)
        ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    else:
        ax.set(xticks=x, xticklabels=labels, ylabel='Destination recipe volume (nL)',
               title='Prepared K-vector additions at each destination well', ylim=(0, branch_volume*1.2))
        ax.set_xlabel('Gray: remaining volume (buffer / incoming fluid); each bar is one destination recipe, not cumulative liquid.', fontsize=8)
    if len(active) <= 12:
        if vertical_timeline:
            ax.legend(fontsize=7, ncol=1, loc='upper center', frameon=False)
        else:
            ax.legend(fontsize=8, ncol=min(len(active)+1, 5))


def draw_selected_addition_histogram(ax, result, target_row, *, vertical_timeline=False):
    assigned = int(np.flatnonzero(np.asarray(result['assignment']) == target_row)[0])
    independent = result.get('synthesis_mode') == 'independent_wells'
    code = format(int(result['address_indices'][assigned]), f"0{result['address_bits']}b")
    branches = [assigned] if independent else [2*r+int(bit) for r, bit in enumerate(code)]
    recipe = result['recipe']
    labels = ['Mix all K stocks\nCapture once'] if independent else [f'Round {r+1}\nbranch {bit}' for r, bit in enumerate(code)]
    draw_addition_histogram(ax, np.asarray(recipe['branch_transfers_nl'])[branches],
                           np.asarray(recipe['branch_remaining_nl'])[branches], labels,
                           recipe['settings']['branch_volume_nl'], vertical_timeline=vertical_timeline)
