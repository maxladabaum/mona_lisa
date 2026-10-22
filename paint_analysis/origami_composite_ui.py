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

from origami_composite import CompositeSettings, encoding_codebook, reconstruct_composite, export_composite, reconstruction_mapping_issues, reorient_composite, saved_rgb_batch_mapping, image_region_defaults


class CompositeWindow(tk.Toplevel):
    def __init__(self, owner, grid_points, alignment_points):
        super().__init__(owner)
        self.owner, self.grid_points, self.alignment_points = owner, grid_points, alignment_points
        self.title('Origami · Reconstruct composite image')
        self.geometry('1180x850')
        self.result = None
        self._closed = False
        self.busy = False
        self.codebook = {}
        self.geometry_settings = {}
        self.rows = {}
        self.row_widgets = {}
        self.events = queue.Queue()
        self.status = tk.StringVar(value='Assign each template a color and XY shift, then check the image region.')
        self.variables = {}
        self.source_scope = tk.StringVar(value='')
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
        geometry_box = ttk.LabelFrame(self.page, text='Image-side rectangle in aligned coordinates (X right, Y up)', padding=6)
        geometry_box.pack(fill='x', padx=10, pady=5)
        params = self.first_params()
        grid = self.full_grid(params)
        left, right = float(grid[:, 0].min()), float(grid[:, 0].max())
        bottom, top = float(grid[:, 1].min()), float(grid[:, 1].max())
        sx, sy = float(params.get('spacing_x_nm', 9.9)), float(params.get('spacing_y_nm', 5.1))
        defaults = dict(left_nm=(left + right) / 2, top_nm=top + sy / 2,
                        width_nm=(right - left + sx) / 2, height_nm=top - bottom + sy,
                        pixel_nm=1, sigma_nm=1.5, global_sigma_nm=0, exclusion_radius_nm=2, rotation_deg=0)
        labels = [('left_nm', 'Left X (nm)'), ('top_nm', 'Top Y (nm)'), ('width_nm', 'Width (nm)'),
                  ('height_nm', 'Height (nm)'), ('pixel_nm', 'Output pixel (nm)'), ('sigma_nm', 'Per-section Gaussian σ (nm)'),
                  ('exclusion_radius_nm', 'Marker exclusion (nm)'), ('rotation_deg', 'Whole image rotation (°)'),
                  ('global_sigma_nm', 'Whole-image Gaussian σ (nm)')]
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
            widget.grid(row=i // 4 * 2 + 1, column=i % 4, sticky='ew', padx=5)
        self.average = tk.BooleanVar(value=True)
        self.average.trace_add('write', self.invalidate)
        ttk.Checkbutton(geometry_box, text='Average signal per origami within each species', variable=self.average).grid(row=6, column=0, columnspan=3, sticky='w', pady=5)
        ttk.Label(geometry_box, text='Positive decoded shifts move right/down in the composite. Preview the rectangle before building; exclude barcode and fiducial signal.').grid(row=7, column=0, columnspan=4, sticky='w')
        ttk.Label(geometry_box, text='Whole-image smoothing blends across section boundaries after assembly. Set σ to 0 to disable; rebuild after changes.').grid(row=8, column=0, columnspan=4, sticky='w')
        actions = ttk.Frame(self.page)
        actions.pack(fill='x', padx=10)
        ttk.Button(actions, text='Preview image region', command=self.preview_region).pack(side='left', padx=3)
        ttk.Button(actions, text='Build composite', command=self.build).pack(side='left', padx=3)
        ttk.Button(actions, text='Export PNGs + densities + report…', command=self.export).pack(side='left', padx=3)
        self.view = tk.StringVar(value='Composite')
        selector = ttk.Combobox(actions, textvariable=self.view, values=('Composite', 'Red', 'Green', 'Blue', 'Coverage'), state='readonly', width=14)
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
        ttk.Label(self.page, textvariable=self.status, wraplength=1120).pack(fill='x', padx=12, pady=5)
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
        results = self.owner.origami_multi_template_results
        if results:
            return next(iter(results.values()))['params']
        templates = self.owner.origami_custom_templates
        return templates[0] if templates else dict(rows=8, columns=12, spacing_x_nm=9.9, spacing_y_nm=5.1)

    def full_grid(self, params):
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
        previous = self.mapping_values() if self.rows else {}
        for child in self.table.winfo_children():
            child.destroy()
        self.rows = {}
        self.row_widgets = {}
        for col, label in enumerate(('Include', 'Barcode template', 'Code / binary codebook entry', 'Color', 'X shift (nm)', 'Y shift (nm)', 'Rotate CW (°)', 'Mirror ↔', 'Mirror ↕', 'Brightness ×', 'Accepted')):
            ttk.Label(self.table, text=label).grid(row=0, column=col, padx=5, sticky='w')
        templates = self.owner.origami_custom_templates
        names = list(dict.fromkeys([str(t['name']) for t in templates] + list(self.owner.origami_multi_template_results)))
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
                          brightness=tk.StringVar(value=saved.get('brightness', '1')))
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
            payload = self.owner.origami_multi_template_results.get(name)
            count = int(np.count_nonzero(payload['picks'].accepted_mask)) if payload else 'Run Step 4'
            ttk.Label(self.table, text=str(count)).grid(row=row, column=10, padx=5)
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
            self.owner.origami_composite_configuration = self.configuration()
            self.plot_result()
            self.status.set('Composite orientation updated. PNG and numerical exports use this orientation; no reconstruction rerun is needed.')
        except (ValueError, tk.TclError) as exc:
            self.status.set(str(exc))

    def mapping_values(self):
        return {name: {key: var.get() for key, var in row.items()} for name, row in self.rows.items()}

    def settings(self):
        values = {key: float(var.get()) for key, var in self.variables.items()}
        settings = CompositeSettings(**values, average_species=self.average.get(), mirror_x=self.mirror_x.get(), mirror_y=self.mirror_y.get())
        settings.validate()
        return settings

    def configuration(self):
        return dict(format='paint-composite-mappings-v1', mappings=self.mapping_values(),
                    settings=asdict(self.settings()), geometry=self.geometry_settings,
                    codebook=self.codebook)

    def apply_configuration(self, document):
        if document.get('format') != 'paint-composite-mappings-v1':
            raise ValueError('This is not a saved composite mapping file.')
        settings = CompositeSettings(**document['settings'])
        settings.validate()
        self.codebook = document.get('codebook', {})
        self.geometry_settings = document.get('geometry', {})
        self.refresh_templates()
        for name, values in document['mappings'].items():
            if name in self.rows:
                for key, var in self.rows[name].items():
                    var.set(values.get(key, {'rotation_deg': 0, 'mirror_x': False, 'mirror_y': False, 'brightness': 1}.get(key, var.get())))
                self.rows[name]['code']._applied_code = values['code']
        for key, var in self.variables.items():
            var.set(getattr(settings, key))
        self.average.set(settings.average_species)
        self.mirror_x.set(settings.mirror_x)
        self.mirror_y.set(settings.mirror_y)
        missing = set(document['mappings']) - set(self.rows)
        self.status.set('Mappings loaded.' + (f' {len(missing)} saved templates are not currently loaded.' if missing else ''))

    def load_templates(self):
        if self.busy:
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
            self.geometry_settings,self.codebook=geometry,codebook
            self.refresh_templates()
            for name,row in self.rows.items():
                # Prevent unrelated control origami from inheriting old placements.
                row['include'].set(name in mappings)
                if name in mappings:
                    for key,var in row.items():
                        var.set(mappings[name][key])
                    row['code']._applied_code=mappings[name]['code']
                else:
                    row['code'].set('')
                    row['color'].set('')
            for key,value in defaults.items():
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
                self.owner.origami_composite_configuration = document
        except Exception as exc:
            messagebox.showerror('Cannot save mappings', str(exc), parent=self)

    def exclusions(self, results):
        excluded = {}
        for name, payload in results.items():
            params = payload['params']
            sites = [self.alignment_points(np.empty((0, 2)), params)]
            model = params.get('logical_model') or {}
            cells = sorted({int(i) for group in model.get('bit_physical_cells', ()) for i in group})
            if cells:
                sites.append(self.full_grid(params)[cells])
            excluded[name] = np.concatenate(sites).reshape(-1, 2)
        return excluded

    def preview_region(self):
        try:
            settings = self.settings()
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            params = self.first_params()
            grid = self.full_grid(params)
            ax.scatter(*grid.T, s=12, color='gray', label='Physical lattice')
            results = self.owner.origami_multi_template_results
            for payload in results.values():
                picks = payload['picks']
                selected = np.flatnonzero(picks.accepted_mask)
                if len(selected):
                    from origami_composite import pose_basis
                    from scipy.spatial import cKDTree
                    center, basis = pose_basis(picks.rectangle_corners_nm[selected[0]])
                    source = self.owner.origami_source_points_nm
                    radius = max(np.linalg.norm(grid, axis=1).max(), np.hypot(settings.left_nm, settings.top_nm) + np.hypot(settings.width_nm, settings.height_nm))
                    ids = cKDTree(source).query_ball_point(center, radius)
                    points = (source[ids] - center) @ basis
                    ax.scatter(*points.T, s=2, alpha=.25, label='Example accepted origami')
                    break
            if results:
                sites = next(iter(self.exclusions(results).values()))
                ax.scatter(*sites.T, marker='x', color='red', label='Excluded barcode / fiducial sites')
            ax.add_patch(Rectangle((settings.left_nm, settings.top_nm - settings.height_nm), settings.width_nm, settings.height_nm, fill=False, edgecolor='limegreen', linewidth=2, label='Image extraction region'))
            ax.set(xlabel='Aligned X (nm)', ylabel='Aligned Y (nm)', title='Image-side extraction before decoded shifts')
            ax.set_aspect('equal'); ax.legend(fontsize=8); ax.autoscale_view()
            self.canvas.draw_idle()
        except Exception as exc:
            messagebox.showerror('Cannot preview region', str(exc), parent=self)

    def build(self):
        record_diagnostic("composite_build_requested")
        if self.busy:
            return
        try:
            if self.owner.origami_identification_running:
                raise ValueError('Wait for classification to finish before reconstruction.')
            baseline = self.owner.origami_identification_baseline
            if baseline is not None and self.owner._origami_identification_snapshot() != baseline:
                raise ValueError('Identification settings changed. Rerun classification before reconstruction.')
            settings = self.settings()
            results = self.owner.origami_multi_template_results
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
            exclusions = self.exclusions(results)
            source = self.owner.origami_source_points_nm
            configuration = self.configuration()
            unclassified_count = int(self.owner.origami_multi_template_unclassified_count)
            source_path = str(self.owner.origami_loaded_source_path)
            self.owner.origami_composite_configuration = configuration
            self.busy = True
            self.result = None
            self.status.set('Extracting image-side localizations from accepted origami…')
            def work():
                try:
                    result = reconstruct_composite(source, results, mappings, settings, exclusions,
                                                   lambda text: self.events.put(('progress', text)))
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
            messagebox.showerror('Cannot reconstruct', str(exc), parent=self)

    def poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                record_diagnostic('composite_event', kind=kind, message=payload if kind in ('progress', 'error') else 'result ready')
                if kind == 'progress':
                    self.status.set(payload)
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
                    if (configuration != current_configuration or results is not self.owner.origami_multi_template_results
                            or source is not self.owner.origami_source_points_nm):
                        self.status.set('Inputs changed during reconstruction. Build again to use the updated inputs.')
                        continue
                    result['report']['mapping_configuration'] = configuration
                    self.result = result
                    self.result_inputs = (results, source)
                    species = result['report']['species']
                    self.status.set(f"{sum(s['accepted_origami'] for s in species)} origami; {sum(s['missing'] for s in species)} missing species. "
                                    'Per-template counts are included in the exported report.')
                    self.plot_result()
                    self.reveal_preview()
        except queue.Empty:
            pass
        self.poll_after_id = self.after(100, self.poll)

    def plot_result(self):
        if self.result is None:
            return
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        mode = self.view.get()
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

    def export(self):
        try:
            if self.result is None:
                raise ValueError('Build the composite before exporting.')
            if (self.result_inputs[0] is not self.owner.origami_multi_template_results
                    or self.result_inputs[1] is not self.owner.origami_source_points_nm):
                raise ValueError('Classification or source changed. Build the composite again.')
            baseline = self.owner.origami_identification_baseline
            if baseline is not None and self.owner._origami_identification_snapshot() != baseline:
                raise ValueError('Identification settings changed. Rerun classification and rebuild before export.')
            path = filedialog.asksaveasfilename(parent=self, defaultextension='.png', initialfile='origami_composite.png', filetypes=[('PNG', '*.png')])
            if path:
                export_composite(self.result, path)
                self.status.set('Saved plain PNG and _axes.png with nanometre axes, numerical RGB densities and coverage (.npz), and reconstruction report (.json).')
        except Exception as exc:
            messagebox.showerror('Cannot export composite', str(exc), parent=self)

    def close(self):
        try:
            self.owner.origami_composite_configuration = self.configuration()
        except (ValueError, tk.TclError):
            pass
        self._closed = True
        self.after_cancel(self.poll_after_id)
        self.result = None
        self.result_inputs = None
        while True:
            try:
                self.events.get_nowait()
            except queue.Empty:
                break
        if getattr(self.owner, 'origami_composite_window', None) is self:
            self.owner.origami_composite_window = None
        self.destroy()
