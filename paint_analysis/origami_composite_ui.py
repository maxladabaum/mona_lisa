"""Independent reconstruction editor attached to the PAINT Origami tab."""
from crash_tracking import record as record_diagnostic
from dataclasses import asdict
import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from binary_occupancy import (MODES, OccupancySettings, reconstruct_occupancy, export_occupancy,
                              occupancy_figure, export_subsampling)
from reconstruction_analysis import load_reconstruction_from_analysis
from composite_session import REGION_KEYS, save_reconstruction_session, load_reconstruction_session
from origami_composite import image_region_weights, image_region_preview_points, RegionPreviewCancelled, _progress_heartbeat, CompositeSettings, encoding_codebook, reconstruct_composite, export_composite, reconstruction_mapping_issues, reorient_composite, saved_rgb_batch_mapping, image_region_defaults


class CompositeWindow(tk.Toplevel):
    def __init__(self, owner, grid_points, alignment_points):
        super().__init__(owner)
        self.owner, self.grid_points, self.alignment_points = owner, grid_points, alignment_points
        self.title('Origami · Reconstruct composite image')
        self.geometry('1180x850')
        self.result = None
        self.reconstruction_owner = None
        self.batch_manifest = None
        self._closed = False
        self.busy = False
        self.preview_cancel = None
        self.codebook = {}
        self.geometry_settings = {}
        self.rows = {}
        self.row_widgets = {}
        self.events = queue.Queue()
        self.status = tk.StringVar(value='Assign each template a color and XY shift, then check the image region.')
        self.variables = {}
        self.region_widgets = {}
        self.source_scope = tk.StringVar(value='')
        # Keep progress visible even when the controls/preview are scrolled.
        ttk.Label(self, textvariable=self.status, wraplength=1120, anchor='w', relief='sunken').pack(side='bottom', fill='x', padx=6, pady=4)
        # Keep the controls and a full-size preview in one scrollable page.
        viewport = ttk.Frame(self)
        viewport.pack(fill='both', expand=True)
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)
        self.page_canvas = tk.Canvas(viewport, highlightthickness=0, yscrollincrement=24)
        self.page_canvas.grid(row=0, column=0, sticky='nsew')
        page_vertical = ttk.Scrollbar(viewport, orient='vertical', command=self.page_canvas.yview)
        page_vertical.grid(row=0, column=1, sticky='ns')
        page_horizontal = ttk.Scrollbar(viewport, orient='horizontal', command=self.page_canvas.xview)
        page_horizontal.grid(row=1, column=0, sticky='ew')
        self.page_canvas.configure(yscrollcommand=page_vertical.set, xscrollcommand=page_horizontal.set)
        self.page = ttk.Frame(self.page_canvas)
        self.page_window = self.page_canvas.create_window((0, 0), window=self.page, anchor='nw')
        self.page.bind('<Configure>', self.resize_page)
        self.page_canvas.bind('<Configure>', self.resize_page)
        self.bind('<MouseWheel>', self.scroll_page)
        self.bind('<Button-4>', self.scroll_page)
        self.bind('<Button-5>', self.scroll_page)
        ttk.Label(self.page, textvariable=self.source_scope, wraplength=1120).pack(fill='x', padx=12, pady=3)
        controls = ttk.Frame(self.page, padding=8)
        controls.pack(fill='x')
        for label, command in [('Load barcode templates…', self.load_templates),
                               ('Load saved RGB picklist batch…', self.load_picklist_batch),
                               ('Load image encoding settings…', self.load_encoding),
                               ('Load mappings…', self.load_mappings), ('Save mappings…', self.save_mappings)]:
            ttk.Button(controls, text=label, command=command).pack(side='left', padx=3)
        sessions = ttk.Frame(self.page, padding=8)
        sessions.pack(fill='x')
        ttk.Button(sessions, text='Save reconstruction session…', command=self.save_session).pack(side='left', padx=3)
        ttk.Button(sessions, text='Open reconstruction session…', command=self.load_session).pack(side='left', padx=3)
        ttk.Label(sessions, text='Reopen directly without loading the full analysis.').pack(side='left', padx=8)
        ttk.Label(self.page, text='Load image_manifest.json from a saved RGB picklist batch to fill color, XY placement, and image region. Unmatched templates need manual mapping.').pack(anchor='w', padx=12)
        container = ttk.Frame(self.page)
        container.pack(fill='x', padx=10)
        horizontal = ttk.Scrollbar(container, orient='horizontal')
        horizontal.pack(side='bottom', fill='x')
        canvas = tk.Canvas(container, height=150, highlightthickness=0)
        self.table_canvas = canvas
        self.mapping_summary = tk.StringVar(value='')
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
        self.table = ttk.Frame(canvas)
        item = canvas.create_window((0, 0), window=self.table, anchor='nw')
        horizontal.configure(command=canvas.xview)
        canvas.configure(yscrollcommand=scrollbar.set, xscrollcommand=horizontal.set)
        self.table_window = item
        canvas.pack(side='left', fill='x', expand=True)
        scrollbar.pack(side='right', fill='y')
        self.table.bind('<Configure>', self.resize_mapping_table)
        canvas.bind('<Configure>', self.resize_mapping_table)
        ttk.Label(self.page, textvariable=self.mapping_summary, wraplength=1120).pack(fill='x', padx=12, pady=3)
        self.refresh_templates()
        bulk_mirrors = ttk.Frame(self.page)
        bulk_mirrors.pack(fill='x', padx=12, pady=3)
        ttk.Label(bulk_mirrors, text='Mirror all sections before XY placement:').pack(side='left', padx=(0, 6))
        for label, horizontal, vertical in [('None', False, False), ('Left ↔ right', True, False),
                                            ('Top ↔ bottom', False, True), ('Both axes', True, True)]:
            ttk.Button(bulk_mirrors, text=label,
                       command=lambda x=horizontal, y=vertical: self.set_all_section_mirrors(x, y)).pack(side='left', padx=3)
        ttk.Label(self.page, text='Each barcode: rotate clockwise, then mirror left/right or top/bottom; XY shifts place its transformed upper-left corner. Rebuild after changes.').pack(anchor='w', padx=12)
        ttk.Label(self.page, text='Brightness ×: 1 = unchanged, 0.5 = half, 2 = double, 0 = no signal. RGB scales all sections together to the brightest pixel.').pack(anchor='w', padx=12)
        ttk.Label(self.page, text='Excluded fill %: uncheck Include and enter 50 for a uniform half-brightness replacement in that section’s color. Blank leaves it excluded. Fills are synthetic and recorded in exports.', wraplength=1120).pack(anchor='w', padx=12)
        geometry_box = ttk.LabelFrame(self.page, text='Image-side rectangle in aligned coordinates (X right, Y up)', padding=6)
        geometry_box.pack(fill='x', padx=10, pady=5)
        params = self.first_params()
        grid = self.full_grid(params)
        left, right = float(grid[:, 0].min()), float(grid[:, 0].max())
        bottom, top = float(grid[:, 1].min()), float(grid[:, 1].max())
        sx, sy = float(params.get('spacing_x_nm', 9.9)), float(params.get('spacing_y_nm', 5.1))
        defaults = dict(left_nm=(left + right) / 2, top_nm=top + sy / 2,
                        width_nm=(right - left + sx) / 2, height_nm=top - bottom + sy,
                        pixel_nm=1, sigma_nm=1.5, global_sigma_nm=0, exclusion_radius_nm=2, rotation_deg=0,
                        blend_width_nm=0, blend_height_nm=0)
        labels = [('left_nm', 'Left X (nm)'), ('top_nm', 'Top Y (nm)'), ('width_nm', 'Width (nm)'),
                  ('height_nm', 'Height (nm)'), ('pixel_nm', 'Output pixel (nm)'), ('sigma_nm', 'Per-section Gaussian σ (nm)'),
                  ('exclusion_radius_nm', 'Overlap marker exclusion σ (nm)'), ('rotation_deg', 'Whole image rotation (°)'),
                  ('global_sigma_nm', 'Whole-image Gaussian σ (nm)'),
                  ('blend_width_nm', 'Overlap total width (nm)'), ('blend_height_nm', 'Overlap total height (nm)')]
        for i, (key, label) in enumerate(labels):
            var = tk.StringVar(value=f'{defaults[key]:g}')
            self.variables[key] = var
            ttk.Label(geometry_box, text=label).grid(row=i // 4 * 2, column=i % 4, sticky='w', padx=5)
            if key == 'rotation_deg':
                widget = ttk.Combobox(geometry_box, textvariable=var, values=(0,90,180,270), state='readonly', width=16)
                var.trace_add('write', self.update_display_orientation)
            else:
                widget = ttk.Entry(geometry_box, textvariable=var, width=18)
                var.trace_add('write', self.invalidate)
            if key in REGION_KEYS:
                self.region_widgets[key] = widget
            widget.grid(row=i // 4 * 2 + 1, column=i % 4, sticky='ew', padx=5)
        self.average = tk.BooleanVar(value=True)
        self.average.trace_add('write', self.invalidate)
        ttk.Checkbutton(geometry_box, text='Average signal per origami within each species', variable=self.average).grid(row=6, column=0, columnspan=3, sticky='w', pady=5)
        ttk.Label(geometry_box, text='Positive decoded shifts move right/down in the composite. Preview the rectangle before building; exclude barcode and fiducial signal.').grid(row=7, column=0, columnspan=4, sticky='w')
        ttk.Label(geometry_box, text='Whole-image smoothing blends across section boundaries after assembly. Set σ to 0 to disable; rebuild after changes.').grid(row=8, column=0, columnspan=4, sticky='w')
        ttk.Label(geometry_box, text='Overlap uses total dimensions, not added margins: height 45 → total height 55 adds 5 nm per side. 0 follows normal size; Gaussian tails reach 3σ at the edges.').grid(row=9, column=0, columnspan=4, sticky='w')
        self.normalize_overlap = tk.BooleanVar(value=False)
        self.normalize_overlap.trace_add('write', self.invalidate)
        ttk.Checkbutton(geometry_box, text='Normalize overlapping sections (optional; off adds Gaussian-weighted signals)',
                        variable=self.normalize_overlap).grid(row=10, column=0, columnspan=4, sticky='w', pady=5)
        occupancy_box = ttk.LabelFrame(self.page, text='Reconstruction method', padding=6)
        occupancy_box.pack(fill='x', padx=10, pady=5)
        self.reconstruction_mode = tk.StringVar(value=MODES[0])
        ttk.Combobox(occupancy_box, textvariable=self.reconstruction_mode, values=MODES,
                     state='readonly', width=32).grid(row=0, column=0, columnspan=2, sticky='w')
        self.reconstruction_mode.trace_add('write', self.invalidate)
        self.occupancy_variables = {}
        self.occupancy_advanced = {}
        for i, (key, label, default) in enumerate((('radius_nm', 'ROI radius (nm)', '2'),
                ('min_localizations', 'ON count threshold', '3'),
                ('uncertainty_margin', 'Uncertain count margin (0 = off)', '0'),
                ('overlay_density_per_on', 'Calibrated overlay density per ON (optional)', ''))):
            var = tk.StringVar(value=default)
            self.occupancy_variables[key] = var
            var.trace_add('write', self.invalidate)
            ttk.Label(occupancy_box, text=label).grid(row=1, column=i, sticky='w', padx=4)
            ttk.Entry(occupancy_box, textvariable=var, width=20).grid(row=2, column=i, sticky='w', padx=4)
        self.occupancy_variables['gaussian_fwhm_nm'] = tk.StringVar(value='')
        self.occupancy_variables['gaussian_fwhm_nm'].trace_add('write', self.invalidate)
        ttk.Label(occupancy_box, text='Occupancy Gaussian FWHM (nm; blank = saved picklist)').grid(row=3, column=0, columnspan=2, sticky='w', padx=4)
        ttk.Entry(occupancy_box, textvariable=self.occupancy_variables['gaussian_fwhm_nm'], width=20).grid(row=3, column=2, sticky='w', padx=4)
        ttk.Button(occupancy_box, text='Load occupancy settings JSON…', command=self.load_occupancy_settings).grid(row=0, column=2)
        ttk.Label(occupancy_box, text='Occupancy images use Gaussian spots and saved picklist display settings. Site occupancy retains the measured ON/valid fractions.').grid(row=4, column=0, columnspan=4, sticky='w')
        diagnostics = ttk.Frame(self.page)
        diagnostics.pack(fill='x', padx=10)
        self.occupancy_barcode = tk.StringVar(value='')
        self.occupancy_position = tk.StringVar(value='1')
        ttk.Label(diagnostics, text='Diagnostic template:').pack(side='left')
        self.occupancy_barcode_selector = ttk.Combobox(diagnostics, textvariable=self.occupancy_barcode, state='readonly', width=35)
        self.occupancy_barcode_selector.pack(side='left', padx=4)
        self.occupancy_barcode_selector.bind('<<ComboboxSelected>>', lambda e: self.plot_result())
        ttk.Label(diagnostics, text='Site number (manifest order):').pack(side='left')
        ttk.Spinbox(diagnostics, from_=1, to=96, textvariable=self.occupancy_position, width=5, command=self.plot_result).pack(side='left')
        ttk.Button(diagnostics, text='Export sample-count study…', command=self.export_sample_study).pack(side='left', padx=5)
        heatmap_controls = ttk.Frame(self.page)
        heatmap_controls.pack(fill='x', padx=10, pady=3)
        ttk.Button(heatmap_controls, text='Template heatmaps', command=self.show_template_heatmaps).pack(side='left', padx=3)
        self.occupancy_show_grid = tk.BooleanVar(value=True)
        self.occupancy_show_values = tk.BooleanVar(value=True)
        self.occupancy_full_grid = tk.BooleanVar(value=False)
        ttk.Checkbutton(heatmap_controls, text='Overlay origami grid', variable=self.occupancy_show_grid,
                        command=self.plot_result).pack(side='left', padx=6)
        ttk.Checkbutton(heatmap_controls, text='Show site occupancy values', variable=self.occupancy_show_values,
                        command=self.plot_result).pack(side='left', padx=6)
        ttk.Checkbutton(heatmap_controls, text='Full origami context', variable=self.occupancy_full_grid,
                        command=self.plot_result).pack(side='left', padx=6)
        ttk.Label(heatmap_controls, text='Uses the Diagnostic template selected above.').pack(side='left', padx=6)
        actions = ttk.Frame(self.page)
        actions.pack(fill='x', padx=10)
        ttk.Button(actions, text='Preview image region', command=self.preview_region).pack(side='left', padx=3)
        ttk.Button(actions, text='Cancel preview', command=self.cancel_region_preview).pack(side='left', padx=3)
        ttk.Button(actions, text='Build composite', command=self.build).pack(side='left', padx=3)
        ttk.Button(actions, text='Export PNGs + densities + report…', command=self.export).pack(side='left', padx=3)
        self.view = tk.StringVar(value='Composite')
        selector = ttk.Combobox(actions, textvariable=self.view, values=('Composite', 'Red', 'Green', 'Blue', 'Coverage', 'Comparison', 'Individual sites', 'Count distributions', 'Template heatmaps', 'Barcode maps', 'Target errors', 'Site occupancy'), state='readonly', width=14)
        selector.pack(side='left', padx=8)
        selector.bind('<<ComboboxSelected>>', lambda e: self.plot_result())
        orientation_bar = ttk.Frame(self.page)
        orientation_bar.pack(fill='x', padx=10, pady=(5,0))
        ttk.Label(orientation_bar, text='Whole image:').pack(side='left', padx=3)
        ttk.Button(orientation_bar, text='Rotate left 90°', command=lambda: self.rotate_display(-90)).pack(side='left', padx=3)
        ttk.Button(orientation_bar, text='Rotate right 90°', command=lambda: self.rotate_display(90)).pack(side='left', padx=3)
        self.mirror_x = tk.BooleanVar(value=False)
        self.mirror_y = tk.BooleanVar(value=False)
        for text, variable in (('Mirror left ↔ right', self.mirror_x), ('Mirror top ↔ bottom', self.mirror_y)):
            ttk.Checkbutton(orientation_bar, text=text, variable=variable).pack(side='left', padx=6)
            variable.trace_add('write', self.update_display_orientation)
        ttk.Button(orientation_bar, text='Reset orientation', command=self.reset_display_orientation).pack(side='left', padx=3)
        self.figure = Figure(figsize=(10, 7.6), dpi=100, layout='constrained')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.page)
        self.canvas.get_tk_widget().configure(height=760)
        self.canvas.get_tk_widget().pack(fill='x')
        NavigationToolbar2Tk(self.canvas, self.page, pack_toolbar=False).pack(fill='x')
        saved = getattr(owner, 'origami_composite_configuration', None)
        if saved:
            self.apply_configuration(saved)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_after_id = self.after(100, self.poll)

    @property
    def data_owner(self):
        return self.reconstruction_owner or self.owner

    def resize_page(self, _event=None):
        self.page_canvas.itemconfigure(
            self.page_window, width=max(self.page_canvas.winfo_width(), self.page.winfo_reqwidth()))
        self.page_canvas.configure(scrollregion=self.page_canvas.bbox('all'))

    def scroll_page(self, event):
        if getattr(event, 'num', None) in (4, 5):
            amount = -1 if event.num == 4 else 1
        elif event.delta:
            amount = int(-event.delta / 120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1)
        else:
            return
        self.page_canvas.yview_scroll(amount, 'units')
        return 'break'

    def reveal_preview(self):
        self.page.update_idletasks()
        self.resize_page()
        self.page_canvas.yview_moveto(
            self.canvas.get_tk_widget().winfo_y() / max(1, self.page.winfo_height()))

    def first_params(self):
        results = self.data_owner.origami_multi_template_results
        if results:
            return next(iter(results.values()))['params']
        templates = self.data_owner.origami_custom_templates
        return templates[0] if templates else dict(rows=8, columns=12, spacing_x_nm=9.9, spacing_y_nm=5.1)

    def full_grid(self, params):
        if '_reconstruction_grid_nm' in params:
            return np.asarray(params['_reconstruction_grid_nm'], dtype=float)
        return self.grid_points(int(params.get('rows', 8)), int(params.get('columns', 12)),
                                float(params.get('spacing_x_nm', 9.9)), float(params.get('spacing_y_nm', 5.1)), params)

    def invalidate(self, *args):
        if self.__dict__.get('_bulk_mapping_update', False):
            return
        self.result = None
        self.update_mapping_summary()
        if hasattr(self, 'status'):
            self.status.set('Mapping or geometry changed. Build the composite to update the preview and exports.')

    def refresh_templates(self):
        preview = self.first_params().get('_checkpoint_preview')
        self.source_scope.set(
            f"Partial analysis: {preview['completed']}/{preview['planned']} tiles. Composite uses completed tiles only."
            if preview else '')
        if self.reconstruction_owner is not None:
            self.source_scope.set('Standalone reconstruction: ' + self.reconstruction_owner.reconstruction_path + '\n' + self.source_scope.get())
        previous = self.mapping_values() if self.rows else {}
        for child in self.table.winfo_children():
            child.destroy()
        self.rows = {}
        self.row_widgets = {}
        for col, label in enumerate(('Include', 'Barcode template', 'Code / binary codebook entry', 'Color', 'X shift (nm)', 'Y shift (nm)', 'Rotate CW (°)', 'Mirror ↔', 'Mirror ↕', 'Brightness ×', 'Excluded fill %', 'Accepted')):
            ttk.Label(self.table, text=label).grid(row=0, column=col, padx=5, sticky='w')
        templates = self.data_owner.origami_custom_templates
        names = list(dict.fromkeys([str(t['name']) for t in templates] + list(self.data_owner.origami_multi_template_results)))
        for row, name in enumerate(names, 1):
            saved = previous.get(name, {})
            values = dict(include=tk.BooleanVar(value=saved.get('include', True)),
                          code=tk.StringVar(value=saved.get('code', '')),
                          color=tk.StringVar(value=saved.get('color', '')),
                          x_shift_nm=tk.StringVar(value=saved.get('x_shift_nm', '0')),
                          y_shift_nm=tk.StringVar(value=saved.get('y_shift_nm', '0')),
                          rotation_deg=tk.StringVar(value=saved.get('rotation_deg', '0')),
                          mirror_x=tk.BooleanVar(value=saved.get('mirror_x', False)),
                          mirror_y=tk.BooleanVar(value=saved.get('mirror_y', False)),
                          brightness=tk.StringVar(value=saved.get('brightness', '1')),
                          fill_percent=tk.StringVar(value=saved.get('fill_percent', '')))
            self.rows[name] = values
            values['code']._applied_code = values['code'].get()
            ttk.Checkbutton(self.table, variable=values['include']).grid(row=row, column=0)
            ttk.Label(self.table, text=name).grid(row=row, column=1, sticky='w', padx=5)
            codes = ttk.Combobox(self.table, textvariable=values['code'], values=tuple(self.codebook), width=16)
            codes.grid(row=row, column=2, padx=5)
            codes.bind('<<ComboboxSelected>>', lambda e, n=name: self.apply_code(n))
            codes.bind('<FocusOut>', lambda e, n=name: self.apply_code(n, only_if_new=True))
            color_picker = ttk.Combobox(self.table, textvariable=values['color'], values=('Red', 'Green', 'Blue'), state='readonly', width=9)
            color_picker.grid(row=row, column=3)
            self.row_widgets[name] = color_picker
            for col, key in ((4, 'x_shift_nm'), (5, 'y_shift_nm')):
                ttk.Entry(self.table, textvariable=values[key], width=8).grid(row=row, column=col, padx=5)
            ttk.Combobox(self.table, textvariable=values['rotation_deg'], values=(0, 90, 180, 270),
                         state='readonly', width=5).grid(row=row, column=6, padx=5)
            for col, key in ((7, 'mirror_x'), (8, 'mirror_y')):
                ttk.Checkbutton(self.table, variable=values[key]).grid(row=row, column=col)
            ttk.Entry(self.table, textvariable=values['brightness'], width=8).grid(row=row, column=9, padx=5)
            ttk.Entry(self.table, textvariable=values['fill_percent'], width=8).grid(row=row, column=10, padx=5)
            payload = self.data_owner.origami_multi_template_results.get(name)
            count = int(np.count_nonzero(payload['picks'].accepted_mask)) if payload else 'Run Step 4'
            ttk.Label(self.table, text=str(count)).grid(row=row, column=11, padx=5)
            for var in values.values():
                var.trace_add('write', self.invalidate)
        for widget in (self.table_canvas, self.table, *self.table.winfo_children()):
            widget.bind('<MouseWheel>', self.scroll_mapping_table)
            widget.bind('<Button-4>', lambda e: self.table_canvas.yview_scroll(-1, 'units'))
            widget.bind('<Button-5>', lambda e: self.table_canvas.yview_scroll(1, 'units'))
        self.invalidate()

    def resize_mapping_table(self, _event=None):
        self.table_canvas.itemconfigure(self.table_window,
                                        width=max(self.table_canvas.winfo_width(), self.table.winfo_reqwidth()))
        self.table_canvas.configure(scrollregion=self.table_canvas.bbox('all'),
                                    height=min(260, max(80, self.table.winfo_reqheight())))

    def scroll_mapping_table(self, event):
        if event.delta:
            amount = int(-event.delta / 120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1)
            self.table_canvas.yview_scroll(amount, 'units')
        return 'break'

    def update_mapping_summary(self):
        mappings = self.mapping_values()
        included = sum(bool(row.get('include', True)) for row in mappings.values())
        issues = reconstruction_mapping_issues(self.rows, mappings)
        names = list(dict.fromkeys(name for name, _ in issues))
        text = f'{len(self.rows)} templates · {included} included'
        text += (' · Needs attention: ' + ', '.join(names)) if names else ' · All included mappings are ready'
        self.mapping_summary.set(text)

    def show_mapping_issues(self, issues):
        message = '\n'.join(f'Template "{name}": {reason}' for name, reason in issues)
        self.status.set('Check barcode mappings: ' + '; '.join(dict.fromkeys(name for name, _ in issues)))
        widget = self.row_widgets.get(issues[0][0])
        if widget is not None:
            self.table.update_idletasks()
            self.table_canvas.yview_moveto(max(0, widget.winfo_y()-5) / max(1, self.table.winfo_reqheight()))
            widget.focus_set()
        messagebox.showerror('Incomplete barcode mappings', message, parent=self)

    def apply_code(self, name, only_if_new=False):
        row = self.rows[name]
        code = row['code'].get().strip()
        if only_if_new and getattr(row['code'], '_applied_code', None) == code:
            return
        if code in self.codebook:
            for key in ('color', 'x_shift_nm', 'y_shift_nm'):
                row[key].set(self.codebook[code][key])
            row['code']._applied_code = code

    def set_all_section_mirrors(self, horizontal, vertical):
        """Set every row's existing mirror flags, preserving barcode placement."""
        self._bulk_mapping_update = True
        try:
            for row in self.rows.values():
                row['mirror_x'].set(horizontal)
                row['mirror_y'].set(vertical)
        finally:
            self._bulk_mapping_update = False
            self.invalidate()
        self.status.set(f'Section mirrors updated for {len(self.rows)} templates. Click Build composite to apply before XY placement.')

    def rotate_display(self, degrees):
        # Keep the buttons visually clockwise/counterclockwise even with one
        # reflection enabled; reflections reverse the sense of rotation.
        if self.mirror_x.get() != self.mirror_y.get():
            degrees = -degrees
        self.variables['rotation_deg'].set((int(float(self.variables['rotation_deg'].get())) + degrees) % 360)

    def reset_display_orientation(self):
        self.variables['rotation_deg'].set(0)
        self.mirror_x.set(False)
        self.mirror_y.set(False)

    def update_display_orientation(self, *_args):
        if self.result is None:
            return
        try:
            rotation = float(self.variables['rotation_deg'].get())
            self.result = reorient_composite(self.result, rotation, self.mirror_x.get(), self.mirror_y.get())
            self.data_owner.origami_composite_configuration = self.configuration()
            self.plot_result()
            self.status.set('Composite orientation updated. PNG and numerical exports use this orientation; no reconstruction rerun is needed.')
        except (ValueError, tk.TclError) as exc:
            self.status.set(str(exc))

    def mapping_values(self):
        return {name: {key: var.get() for key, var in row.items()} for name, row in self.rows.items()}

    def settings(self):
        values = {key: float(var.get()) for key, var in self.variables.items()}
        settings = CompositeSettings(**values, normalize_overlap=self.normalize_overlap.get(), average_species=self.average.get(), mirror_x=self.mirror_x.get(), mirror_y=self.mirror_y.get())
        settings.validate()
        if self.reconstruction_owner is not None and self.reconstruction_owner.reconstruction_region is not None:
            region = self.reconstruction_owner.reconstruction_region
            if any(getattr(settings, key) != region[key] for key in REGION_KEYS):
                raise ValueError('This reconstruction session has a fixed image region. Save a new session from the full analysis to change it.')
        return settings

    def configuration(self):
        return dict(format='paint-composite-mappings-v1', mappings=self.mapping_values(),
                    settings=asdict(self.settings()), reconstruction_mode=self.reconstruction_mode.get(),
                    occupancy_settings=asdict(self.occupancy_settings()), geometry=self.geometry_settings,
                    codebook=self.codebook, rgb_batch_manifest=self.batch_manifest)

    def apply_configuration(self, document):
        if document.get('format') != 'paint-composite-mappings-v1':
            raise ValueError('This is not a saved composite mapping file.')
        settings = CompositeSettings(**document['settings'])
        settings.validate()
        if self.reconstruction_owner is not None and self.reconstruction_owner.reconstruction_region is not None:
            region = self.reconstruction_owner.reconstruction_region
            if any(getattr(settings, key) != region[key] for key in REGION_KEYS):
                raise ValueError('These mappings use a different image region than the reconstruction session.')
        mode = document.get('reconstruction_mode', MODES[0])
        if mode not in MODES:
            raise ValueError('Unknown reconstruction mode in mapping settings.')
        self.reconstruction_mode.set(mode)
        self.apply_occupancy_settings(document.get('occupancy_settings', {}))
        self.batch_manifest = document.get('rgb_batch_manifest')
        self.codebook = document.get('codebook', {})
        self.geometry_settings = document.get('geometry', {})
        self.refresh_templates()
        for name, values in document['mappings'].items():
            if name in self.rows:
                for key, var in self.rows[name].items():
                    var.set(values.get(key, {'rotation_deg': 0, 'mirror_x': False, 'mirror_y': False, 'brightness': 1, 'fill_percent': ''}.get(key, var.get())))
                self.rows[name]['code']._applied_code = values['code']
        for key, var in self.variables.items():
            var.set(getattr(settings, key))
        self.average.set(settings.average_species)
        self.normalize_overlap.set(settings.normalize_overlap)
        self.mirror_x.set(settings.mirror_x)
        self.mirror_y.set(settings.mirror_y)
        missing = set(document['mappings']) - set(self.rows)
        for widget in self.region_widgets.values():
            widget.configure(state='readonly' if self.reconstruction_owner is not None and self.reconstruction_owner.reconstruction_region is not None else 'normal')
        self.status.set('Mappings loaded.' + (f' {len(missing)} saved templates are not currently loaded.' if missing else ''))

    def load_templates(self):
        if self.busy:
            return
        if self.reconstruction_owner is not None:
            self.status.set('This session contains saved identifications. Close this window to return to the full analysis.')
            return
        self.owner._load_multiple_origami_custom_templates()
        self.refresh_templates()
        self.status.set('Templates loaded. Run Steps 1–4 in the Origami tab before reconstruction.')

    def load_picklist_batch(self):
        path = filedialog.askopenfilename(parent=self, title='Load saved RGB picklist batch (image_manifest.json)', filetypes=[('RGB batch manifest', '*.json')])
        if not path:
            return
        try:
            document=json.loads(Path(path).read_text())
            params=self.first_params()
            geometry,codebook,mappings,defaults=saved_rgb_batch_mapping(document,self.rows,self.full_grid(params),params)
            if not self.rows:
                raise ValueError('Load the barcode classification templates first, then load the saved RGB batch.')
            self.batch_manifest = document
            self.geometry_settings,self.codebook=geometry,codebook
            self.refresh_templates()
            for name,row in self.rows.items():
                # Prevent unrelated control origami from inheriting old placements.
                row['fill_percent'].set('')
                row['include'].set(name in mappings)
                if name in mappings:
                    for key,var in row.items():
                        var.set(mappings[name].get(key, '' if key == 'fill_percent' else var.get()))
                    row['code']._applied_code=mappings[name]['code']
                else:
                    row['code'].set('')
                    row['color'].set('')
            for key,value in defaults.items():
                if self.reconstruction_owner is None or self.reconstruction_owner.reconstruction_region is None or key not in REGION_KEYS:
                    self.variables[key].set(value)
            self.mirror_x.set(False)
            self.mirror_y.set(False)
            self.invalidate()
            missing=len(self.rows)-len(mappings)
            self.status.set(f'Loaded {Path(path).name}: mapped {len(mappings)} templates; {missing} unmatched templates excluded. '
                            f'{len(codebook)-len(mappings)} batch designs not matched. Check the image-region preview; renamed templates can select a code manually.')
        except Exception as exc:
            messagebox.showerror('Cannot load RGB picklist batch',str(exc),parent=self)

    def load_encoding(self):
        path = filedialog.askopenfilename(parent=self, title='Load image encoding settings', filetypes=[('JSON', '*.json')])
        if not path:
            return
        try:
            geometry, codebook = encoding_codebook(json.loads(Path(path).read_text()))
            sx, sy = float(geometry['site_pitch_x_nm']), float(geometry['site_pitch_y_nm'])
            params = self.first_params()
            grid = self.full_grid(params)
            if not (np.isclose(float(params.get('spacing_x_nm', sx)), sx) and np.isclose(float(params.get('spacing_y_nm', sy)), sy)):
                raise ValueError('Encoding site pitch differs from the loaded template. Load compatible templates first.')
            origin = geometry['image_origin_from_C12_H01_nm']
            expected_span = [11 * sx + 10, 7 * sy]
            if not np.allclose(np.ptp(grid, axis=0), expected_span, atol=.05):
                raise ValueError('Encoding geometry requires the full 12-column, 8-row origami lattice, including its central gap. Load compatible templates, or enter the image rectangle and shifts manually.')
            defaults = image_region_defaults(geometry, grid, params)
            self.geometry_settings, self.codebook = geometry, codebook
            self.refresh_templates()
            for key, value in defaults.items():
                if self.reconstruction_owner is None or self.reconstruction_owner.reconstruction_region is None or key not in REGION_KEYS:
                    self.variables[key].set(value)
            for name in self.rows:
                self.apply_code(name)
            self.status.set(f'Loaded {len(codebook)} codes. Select the matching code for each template; color and shifts remain editable. Check the image-region preview.')
        except Exception as exc:
            messagebox.showerror('Cannot load encoding settings', str(exc), parent=self)

    def load_mappings(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[('JSON', '*.json')])
        if path:
            try:
                self.apply_configuration(json.loads(Path(path).read_text()))
            except Exception as exc:
                messagebox.showerror('Cannot load mappings', str(exc), parent=self)

    def save_mappings(self):
        try:
            document = self.configuration()
            path = filedialog.asksaveasfilename(parent=self, defaultextension='.json', initialfile='barcode_image_mappings.json')
            if path:
                Path(path).write_text(json.dumps(document, indent=2) + '\n')
                self.data_owner.origami_composite_configuration = document
        except Exception as exc:
            messagebox.showerror('Cannot save mappings', str(exc), parent=self)

    def exclusions(self, results, progress=None):
        excluded = {}
        for index, (name, payload) in enumerate(results.items(), 1):
            if progress:
                progress(f'Preparing exclusion sites: {name} ({index}/{len(results)} templates)')
            params = payload['params']
            if '_reconstruction_exclusions_nm' in params:
                excluded[name] = np.asarray(params['_reconstruction_exclusions_nm'], dtype=float).reshape(-1, 2)
                continue
            sites = [self.alignment_points(np.empty((0, 2)), params)]
            model = params.get('logical_model') or {}
            cells = sorted({int(i) for group in model.get('bit_physical_cells', ()) for i in group})
            if cells:
                sites.append(self.full_grid(params)[cells])
            excluded[name] = np.concatenate(sites).reshape(-1, 2)
        return excluded

    def cancel_region_preview(self):
        if self.preview_cancel is not None:
            self.preview_cancel.set()
            self.status.set('Stopping image-region preview…')

    def preview_region(self):
        if self.busy:
            return
        try:
            settings = self.settings()
            results = self.data_owner.origami_multi_template_results
            source = self.data_owner.origami_source_points_nm
            fallback_params = self.first_params()
            cancel = threading.Event()
            self.preview_cancel = cancel
            self.busy = True
            self.status.set('Preparing image-region preview…')
            def work():
                try:
                    with _progress_heartbeat(lambda text: self.events.put(('progress', text))) as report:
                        report('Selecting an accepted origami for the image-region preview…')
                        selected = None
                        for name, payload in results.items():
                            if cancel.is_set():
                                raise RegionPreviewCancelled()
                            mask = payload['picks'].accepted_mask
                            if np.any(mask):
                                selected = (name, payload, int(np.argmax(mask)))
                                break
                        # Match grid and marker sites to the selected example.
                        name, payload = (selected[:2] if selected else
                                         next(iter(results.items()), (None, None)))
                        params = payload['params'] if payload is not None else fallback_params
                        grid = self.full_grid(params)
                        sites = (self.exclusions({name: payload}, report)[name]
                                 if payload is not None else np.empty((0, 2)))
                        points, count = np.empty((0, 2)), 0
                        if selected:
                            points, count = image_region_preview_points(
                                source, payload['picks'].rectangle_corners_nm[selected[2]], grid, settings,
                                report, cancel.is_set)
                        if cancel.is_set():
                            raise RegionPreviewCancelled()
                        preview = dict(settings=settings, results=results, source=source, grid=grid,
                                       sites=sites, points=points, count=count, example=name if selected else None)
                    if not self._closed:
                        self.events.put(('region_preview', preview))
                except RegionPreviewCancelled:
                    if not self._closed:
                        self.events.put(('preview_cancelled', None))
                except Exception as exc:
                    if not self._closed:
                        self.events.put(('preview_error', str(exc)))
            threading.Thread(target=work, daemon=True).start()
        except Exception as exc:
            self.busy = False
            self.preview_cancel = None
            self.status.set('Preview failed: ' + str(exc))
            messagebox.showerror('Cannot preview region', str(exc), parent=self)

    def show_region_preview(self, preview):
        try:
            current_settings = self.settings()
        except (ValueError, tk.TclError):
            current_settings = None
        if (preview['settings'] != current_settings
                or preview['results'] is not self.data_owner.origami_multi_template_results
                or preview['source'] is not self.data_owner.origami_source_points_nm):
            self.status.set('Inputs changed during preview. Click Preview image region again.')
            return
        self.status.set('Drawing image-region preview…')
        self.update_idletasks()
        settings = preview['settings']
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(*preview['grid'].T, s=12, color='gray', label='Physical lattice')
        if preview['example'] is not None:
            weights = image_region_weights(preview['points'], preview['sites'], settings)
            colors = np.zeros((len(weights), 4))
            colors[:, :3] = [.12, .47, .71]
            colors[:, 3] = .25 * weights
            ax.scatter(*preview['points'].T, s=2, color=colors, label='Example (Gaussian weighted): ' + preview['example'])
        if len(preview['sites']):
            ax.scatter(*preview['sites'].T, marker='x', color='red', label='Marker sites (excluded only in overlap)')
        ax.add_patch(Rectangle((settings.left_nm, settings.top_nm - settings.height_nm), settings.width_nm,
                               settings.height_nm, fill=False, edgecolor='limegreen', linewidth=2,
                               label='Image extraction region'))
        mx, my = settings.blend_margin
        if mx or my:
            ax.add_patch(Rectangle((settings.left_nm - mx, settings.top_nm - settings.height_nm - my),
                                   *settings.blend_size, fill=False, edgecolor='orange', linestyle='--',
                                   label='Overlap blend rectangle (3σ)'))
        ax.set(xlabel='Aligned X (nm)', ylabel='Aligned Y (nm)', title=f'Image-side extraction · marker exclusion σ = {settings.exclusion_radius_nm:g} nm')
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.autoscale_view()
        self.canvas.draw_idle()
        self.reveal_preview()
        if preview['example'] is None:
            self.status.set('No accepted origami; showing image-region geometry only.')
        else:
            self.status.set(f"Preview ready: {preview['example']}; showing {len(preview['points']):,} "
                            f"of {preview['count']:,} nearby localizations. Display sampling does not affect reconstruction.")

    def occupancy_settings(self):
        values = dict(self.occupancy_advanced)
        for key, var in self.occupancy_variables.items():
            text = var.get().strip()
            values[key] = None if key in ('overlay_density_per_on', 'gaussian_fwhm_nm') and not text else float(text)
        settings = OccupancySettings(**values)
        settings.validate()
        return settings

    def apply_occupancy_settings(self, values):
        settings = OccupancySettings(**values)
        settings.validate()
        values = asdict(settings)
        self.occupancy_advanced = {key: value for key, value in values.items() if key not in self.occupancy_variables}
        for key, var in self.occupancy_variables.items():
            var.set('' if values[key] is None else str(values[key]))

    def load_occupancy_settings(self):
        path = filedialog.askopenfilename(parent=self, title='Occupancy classification settings', filetypes=[('JSON', '*.json')])
        if not path:
            return
        try:
            self.apply_occupancy_settings(json.loads(Path(path).read_text()))
            self.invalidate()
        except Exception as exc:
            messagebox.showerror('Cannot load occupancy settings', str(exc), parent=self)

    def export_sample_study(self):
        if self.busy or self.result is None or 'populations' not in self.result:
            self.status.set('Build Binary Occupancy or Compare Both before exporting a sample-count study.')
            return
        path = filedialog.asksaveasfilename(parent=self, title='Sample-count study', defaultextension='.json', filetypes=[('JSON', '*.json')])
        if not path:
            return
        result = self.result
        self.busy = True
        def work():
            try:
                export_subsampling(result, path, progress=lambda text: self.events.put(('progress', text)))
                self.events.put(('sample_exported', path))
            except Exception as exc:
                self.events.put(('export_error', str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def build(self):
        record_diagnostic("composite_build_requested")
        if self.busy:
            return
        self.status.set('Validating classification and capturing composite settings…')
        self.update_idletasks()
        try:
            if self.data_owner.origami_identification_running:
                raise ValueError('Wait for classification to finish before reconstruction.')
            baseline = self.data_owner.origami_identification_baseline
            if baseline is not None and self.data_owner._origami_identification_snapshot() != baseline:
                raise ValueError('Identification settings changed. Rerun classification before reconstruction.')
            settings = self.settings()
            results = self.data_owner.origami_multi_template_results
            if not results:
                raise ValueError('Run Step 4 to classify templates before reconstruction.')
            if set(results) - set(self.rows):
                self.refresh_templates()
                raise ValueError('Classification templates changed. Review their mappings before building.')
            mappings = self.mapping_values()
            issues = reconstruction_mapping_issues(results, mappings)
            if issues:
                self.show_mapping_issues(issues)
                return
            source = self.data_owner.origami_source_points_nm
            configuration = self.configuration()
            reconstruction_mode = self.reconstruction_mode.get()
            classifier = self.occupancy_settings() if reconstruction_mode != MODES[0] else None
            manifest = self.batch_manifest
            grids = {name: self.full_grid(payload['params']) for name, payload in results.items()} if classifier else None
            unclassified_count = int(self.data_owner.origami_multi_template_unclassified_count)
            source_path = str(self.data_owner.origami_loaded_source_path)
            self.data_owner.origami_composite_configuration = configuration
            self.busy = True
            self.result = None
            self.status.set('Extracting image-side localizations from accepted origami…')
            def work():
                try:
                    progress = lambda text: self.events.put(('progress', text))
                    with _progress_heartbeat(progress) as report:
                        exclusions = self.exclusions(results, report)
                    if reconstruction_mode == MODES[0]:
                        result = reconstruct_composite(source, results, mappings, settings, exclusions, progress)
                    else:
                        result = reconstruct_occupancy(source, results, mappings, settings, manifest, grids,
                            classifier, exclusions, compare=reconstruction_mode == MODES[2], progress=progress)
                    progress('Finalizing source and mapping metadata…')
                    result['report']['unclassified_origami'] = unclassified_count
                    result['report']['source'] = source_path
                    result['report']['encoding_codes_without_observed_species'] = sorted(
                        set(configuration['codebook']) - {s.get('code', '') for s in result['report']['species'] if not s['missing']})
                    if not self._closed:
                        self.events.put(('result', (result, configuration, results, source)))
                except Exception as exc:
                    if not self._closed:
                        self.events.put(('error', str(exc)))
            threading.Thread(target=work, daemon=True).start()
        except Exception as exc:
            self.busy = False
            self.status.set('Reconstruction failed: ' + str(exc))
            messagebox.showerror('Cannot reconstruct', str(exc), parent=self)

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                record_diagnostic('composite_event', kind=kind, message=payload if kind in ('progress', 'error') else 'result ready')
                if kind == 'progress':
                    self.status.set(payload)
                elif kind in ('region_preview', 'preview_cancelled', 'preview_error'):
                    cancelled = self.preview_cancel is not None and self.preview_cancel.is_set()
                    self.busy = False
                    self.preview_cancel = None
                    if kind == 'preview_cancelled' or cancelled:
                        self.status.set('Image-region preview cancelled.')
                    elif kind == 'preview_error':
                        self.status.set('Preview failed: ' + payload)
                        messagebox.showerror('Cannot preview region', payload, parent=self)
                    else:
                        try:
                            self.show_region_preview(payload)
                        except Exception as exc:
                            self.status.set('Preview failed: ' + str(exc))
                            messagebox.showerror('Cannot preview region', str(exc), parent=self)
                elif kind == 'analysis_reconstruction_loaded':
                    self.busy = False
                    try:
                        self.install_analysis_reconstruction(payload)
                    except Exception as exc:
                        self.status.set('Cannot apply reconstruction inputs: ' + str(exc))
                        messagebox.showerror('Cannot apply reconstruction inputs', str(exc), parent=self)
                elif kind == 'session_loaded':
                    self.busy = False
                    self.result = None
                    self.result_inputs = None
                    self.reconstruction_owner = payload
                    self._bulk_mapping_update = True
                    try:
                        self.status.set('Applying saved RGB mappings and reconstruction settings…')
                        self.update_idletasks()
                        self.apply_configuration(payload.origami_composite_configuration)
                    except Exception as exc:
                        self.finish_session_dialog()
                        self.status.set('Cannot apply reconstruction session: ' + str(exc))
                        messagebox.showerror('Cannot apply reconstruction session', str(exc), parent=self)
                    else:
                        self.finish_session_dialog()
                        self.status.set('Reconstruction session loaded. RGB batch mappings applied; saved image region is fixed. Click Build composite.')
                    finally:
                        self._bulk_mapping_update = False
                elif kind == 'session_saved':
                    self.finish_session_dialog()
                    self.busy = False
                    self.status.set('Reconstruction session saved: ' + payload + '. Reopen it directly after restarting; no full analysis load is needed.')
                elif kind == 'session_error':
                    self.finish_session_dialog()
                    self.busy = False
                    self.status.set('Reconstruction session failed: ' + payload)
                    messagebox.showerror('Cannot open/save reconstruction session', payload, parent=self)
                elif kind == 'sample_exported':
                    self.busy = False
                    self.status.set('Saved reproducible sample-count reconstructions, metrics and plot: ' + payload)
                elif kind == 'exported':
                    self.busy = False
                    self.status.set('Saved PNG images, numerical arrays and report; occupancy exports also include individual, summary and error CSV tables.')
                elif kind == 'export_error':
                    self.busy = False
                    self.status.set('Export failed: ' + payload)
                    messagebox.showerror('Cannot export composite', payload, parent=self)
                elif kind == 'error':
                    self.busy = False
                    self.status.set('Reconstruction failed: ' + payload)
                    messagebox.showerror('Cannot reconstruct', payload, parent=self)
                else:
                    self.busy = False
                    result, configuration, results, source = payload
                    try:
                        current_configuration = self.configuration()
                    except (ValueError, tk.TclError):
                        current_configuration = None
                    if (configuration != current_configuration or results is not self.data_owner.origami_multi_template_results
                            or source is not self.data_owner.origami_source_points_nm):
                        self.status.set('Inputs changed during reconstruction. Build again to use the updated inputs.')
                        continue
                    result['report']['mapping_configuration'] = configuration
                    self.result = result
                    self.result_inputs = (results, source)
                    species = result['report']['species']
                    self.status.set('Drawing composite preview…')
                    self.update_idletasks()
                    self.plot_result()
                    self.reveal_preview()
                    self.status.set(f"{sum(s['accepted_origami'] for s in species)} origami; {sum(s['missing'] for s in species)} missing species. "
                                    f"{len(result['report'].get('synthetic_replacements', []))} synthetic replacements. Per-template counts are included in the exported report.")
        except queue.Empty:
            pass
        self.poll_after_id = self.after(100, self.poll)

    def show_template_heatmaps(self):
        if self.result is None or 'populations' not in self.result:
            self.status.set('Build Binary Occupancy or Compare Both, then select a Diagnostic template to inspect its heatmaps.')
            return
        self.view.set('Template heatmaps')
        self.plot_result()
        self.reveal_preview()

    def plot_result(self):
        if self.result is None:
            return
        if 'populations' in self.result:
            keys = list(self.result['populations'])
            self.occupancy_barcode_selector.configure(values=keys)
            if self.occupancy_barcode.get() not in keys:
                self.occupancy_barcode.set(keys[0])
            try:
                position = max(0, int(self.occupancy_position.get()) - 1)
            except ValueError:
                position = 0
            occupancy_figure(self.result, self.view.get(), self.occupancy_barcode.get(), position, self.figure,
                             show_grid=self.occupancy_show_grid.get(), show_values=self.occupancy_show_values.get(),
                             full_grid=self.occupancy_full_grid.get())
            self.canvas.draw_idle()
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        mode = self.view.get()
        if mode not in ('Composite', 'Red', 'Green', 'Blue', 'Coverage'):
            ax.text(.1, .5, 'Select Binary Occupancy or Compare Both and rebuild for diagnostics.')
            self.canvas.draw_idle()
            return
        image = self.result['rgb']
        if mode == 'Coverage':
            image = self.result['coverage'].sum(axis=2)
        elif mode != 'Composite':
            channel = ('Red', 'Green', 'Blue').index(mode)
            image = np.zeros_like(image)
            image[:, :, channel] = self.result['rgb'][:, :, channel]
        px = self.result['report']['output_pixel_nm']
        ax.imshow(image, origin='upper', extent=(0, image.shape[1] * px, image.shape[0] * px, 0))
        ax.set(xlabel='Display X (nm)', ylabel='Display Y (nm)', title=mode + (' · extraction-region coverage' if mode == 'Coverage' else ' · shared RGB brightness scale'))
        self.canvas.draw_idle()

    def load_analysis_reconstruction(self, path, filter_series=None):
        if self.busy:
            self.status.set('Wait for the current reconstruction operation to finish.')
            return
        self.busy = True
        self.status.set('Loading only reconstruction inputs from analysis…')
        def work():
            try:
                inputs = load_reconstruction_from_analysis(
                    path, lambda text: self.events.put(('progress', text)), filter_series)
                if not self._closed:
                    self.events.put(('analysis_reconstruction_loaded', inputs))
            except Exception as exc:
                if not self._closed:
                    self.events.put(('session_error', str(exc)))
        threading.Thread(target=work, daemon=True).start()

    def install_analysis_reconstruction(self, inputs):
        self.result = None
        self.result_inputs = None
        self.reconstruction_owner = inputs
        self._bulk_mapping_update = True
        try:
            self.rows = {}
            self.codebook = {}
            self.geometry_settings = {}
            self.batch_manifest = None
            configuration = inputs.origami_composite_configuration
            if configuration:
                self.apply_configuration(configuration)
            else:
                self.refresh_templates()
                params = self.first_params()
                grid = self.full_grid(params)
                sx, sy = float(params.get('spacing_x_nm', 9.9)), float(params.get('spacing_y_nm', 5.1))
                left, right = grid[:, 0].min(), grid[:, 0].max()
                bottom, top = grid[:, 1].min(), grid[:, 1].max()
                defaults = CompositeSettings(left_nm=(left + right) / 2, top_nm=top + sy / 2,
                                             width_nm=(right - left + sx) / 2, height_nm=top - bottom + sy)
                for key, variable in self.variables.items():
                    variable.set(getattr(defaults, key))
                self.average.set(True)
                self.normalize_overlap.set(False)
                self.mirror_x.set(False)
                self.mirror_y.set(False)
                for widget in self.region_widgets.values():
                    widget.configure(state='normal')
            self.figure.clear()
            self.canvas.draw_idle()
        finally:
            self._bulk_mapping_update = False
        self.status.set('Reconstruction inputs loaded without restoring the full analysis. '
                        + ('Saved RGB mappings restored. ' if configuration else 'Load the RGB picklist batch manifest. ')
                        + 'Define the image region, then Build composite or Save reconstruction session for faster reopening.')

    def load_session(self, path=None):
        if self.busy:
            return
        if path is None:
            path = filedialog.askopenfilename(parent=self, title='Open standalone reconstruction session',
                                              filetypes=[('Reconstruction session', '*.paintrecon')])
        if not path:
            return
        self.busy = True
        self.status.set('Opening reconstruction session…')
        try:
            self.show_session_dialog(loading=True)
        except Exception as exc:
            self.busy = False
            self.finish_session_dialog()
            self.status.set('Cannot open reconstruction session: ' + str(exc))
            messagebox.showerror('Cannot open reconstruction session', str(exc), parent=self)
            return
        def work():
            try:
                loaded = load_reconstruction_session(path, lambda text: self.events.put(('progress', text)))
                if not self._closed:
                    self.events.put(('session_loaded', loaded))
            except Exception as exc:
                if not self._closed:
                    self.events.put(('session_error', str(exc)))
        try:
            threading.Thread(target=work, daemon=True).start()
        except Exception as exc:
            self.busy = False
            self.finish_session_dialog()
            self.status.set('Cannot open reconstruction session: ' + str(exc))
            messagebox.showerror('Cannot open reconstruction session', str(exc), parent=self)

    def show_session_dialog(self, loading=False):
        self.session_io_busy = True
        dialog = tk.Toplevel(self)
        self.session_dialog = dialog
        dialog.title('Loading reconstruction session' if loading else 'Saving reconstruction session')
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.protocol('WM_DELETE_WINDOW', self.prevent_close_during_session_io)
        ttk.Label(dialog, text='Loading...' if loading else 'Saving...', font=('TkDefaultFont', 14)).pack(padx=25, pady=(20, 10))
        ttk.Label(dialog, text='Please wait. This window will close when loading is complete.' if loading else 'Please wait. This window will close when saving is complete.').pack(padx=25, pady=(0, 10))
        ttk.Label(dialog, textvariable=self.status, wraplength=460).pack(padx=25, pady=(0, 15))
        progress = ttk.Progressbar(dialog, mode='indeterminate', length=400)
        self.session_progress = progress
        progress.pack(padx=25, pady=(0, 20))
        progress.start()
        dialog.grab_set()
        dialog.focus_set()

    def prevent_close_during_session_io(self):
        if not self.__dict__.get('session_io_busy', False):
            return False
        dialog = self.__dict__.get('session_dialog')
        if dialog is not None:
            dialog.lift()
            dialog.bell()
        return True

    def finish_session_dialog(self):
        self.session_io_busy = False
        progress = self.__dict__.pop('session_progress', None)
        if progress is not None:
            progress.stop()
        dialog = self.__dict__.pop('session_dialog', None)
        if dialog is not None:
            dialog.grab_release()
            dialog.destroy()

    def save_session(self):
        if self.busy:
            return
        try:
            data = self.data_owner
            if data.origami_identification_running:
                raise ValueError('Wait for classification to finish before saving a reconstruction session.')
            if (data.origami_identification_baseline is not None
                    and data._origami_identification_snapshot() != data.origami_identification_baseline):
                raise ValueError('Identification settings changed. Rerun classification before saving.')
            results = data.origami_multi_template_results
            if not results:
                raise ValueError('Run Step 4 to classify templates before saving a reconstruction session.')
            if not self.batch_manifest:
                region_values = {key: self.variables[key].get() for key in REGION_KEYS}
                self.load_picklist_batch()
                # Saving uses the region the user defined, even when importing
                # the required batch manifest as part of this action.
                for key, value in region_values.items():
                    self.variables[key].set(value)
                if not self.batch_manifest:
                    self.status.set('Load the RGB picklist batch to include it in the reconstruction session.')
                    return
            configuration = self.configuration()
            issues = reconstruction_mapping_issues(results, configuration['mappings'])
            if issues:
                self.show_mapping_issues(issues)
                return
            path = filedialog.asksaveasfilename(parent=self, title='Save standalone reconstruction session',
                                               defaultextension='.paintrecon', initialfile='image_reconstruction.paintrecon',
                                               filetypes=[('Reconstruction session', '*.paintrecon')])
            if not path:
                return
            source = data.origami_source_points_nm
            source_path = str(data.origami_loaded_source_path)
            unclassified_count = int(data.origami_multi_template_unclassified_count)
            self.busy = True
            self.status.set('Preparing reconstruction session…')
            self.show_session_dialog()
            def work():
                try:
                    progress = lambda text: self.events.put(('progress', text))
                    with _progress_heartbeat(progress) as report:
                        exclusions = self.exclusions(results, report)
                        report('Preparing saved template geometry…')
                        grids = {name: self.full_grid(result['params']) for name, result in results.items()}
                    save_reconstruction_session(path, source, results, configuration, exclusions, grids,
                                                source_path, unclassified_count, progress)
                    if not self._closed:
                        self.events.put(('session_saved', path))
                except Exception as exc:
                    if not self._closed:
                        self.events.put(('session_error', str(exc)))
            threading.Thread(target=work, daemon=True).start()
        except Exception as exc:
            self.busy = False
            self.finish_session_dialog()
            self.status.set('Cannot save reconstruction session: ' + str(exc))
            messagebox.showerror('Cannot save reconstruction session', str(exc), parent=self)

    def export(self):
        if self.busy:
            return
        try:
            if self.result is None:
                raise ValueError('Build the composite before exporting.')
            if (self.result_inputs[0] is not self.data_owner.origami_multi_template_results
                    or self.result_inputs[1] is not self.data_owner.origami_source_points_nm):
                raise ValueError('Classification or source changed. Build the composite again.')
            baseline = self.data_owner.origami_identification_baseline
            if baseline is not None and self.data_owner._origami_identification_snapshot() != baseline:
                raise ValueError('Identification settings changed. Rerun classification and rebuild before export.')
            path = filedialog.asksaveasfilename(parent=self, defaultextension='.png', initialfile='origami_composite.png', filetypes=[('PNG', '*.png')])
            if path:
                result = self.result
                self.busy = True
                self.status.set('Preparing composite export…')
                def work():
                    try:
                        (export_occupancy if 'populations' in result else export_composite)(result, path, lambda text: self.events.put(('progress', text)))
                        if not self._closed:
                            self.events.put(('exported', None))
                    except Exception as exc:
                        if not self._closed:
                            self.events.put(('export_error', str(exc)))
                threading.Thread(target=work, daemon=True).start()
        except Exception as exc:
            self.busy = False
            self.status.set('Export failed: ' + str(exc))
            messagebox.showerror('Cannot export composite', str(exc), parent=self)

    def close(self):
        if self.__dict__.get('session_io_busy', False):
            self.prevent_close_during_session_io()
            return
        try:
            self.data_owner.origami_composite_configuration = self.configuration()
        except (ValueError, tk.TclError):
            pass
        self._closed = True
        cancel = self.__dict__.get('preview_cancel')
        if cancel is not None:
            cancel.set()
        self.after_cancel(self.poll_after_id)
        self.result = None
        self.result_inputs = None
        self.reconstruction_owner = None
        while True:
            try:
                self.events.get_nowait()
            except queue.Empty:
                break
        if getattr(self.owner, 'origami_composite_window', None) is self:
            self.owner.origami_composite_window = None
        self.destroy()
