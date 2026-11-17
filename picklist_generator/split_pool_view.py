"""Interactive, in-memory split–pool analysis of RGB targets and analog matrices."""
from copy import deepcopy
from queue import Queue, Empty
from threading import Thread, Event
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class SplitPoolWindow(ttk.Frame):
    """Persistent RGB analysis subtab (historical class name kept for callers)."""
    def __init__(self, owner):
        super().__init__(getattr(owner, '_split_pool_parent', owner))
        self.owner = owner; self.cancel = Event(); self.events = Queue(); self.running = False
        self.k_results = {}; self.basis_k = tk.StringVar(value="4"); self.k_sweep = tk.StringVar(value="1,2,4,8")
        self.results = {}; self.data = None; self._poll_id = None
        self.image_designs = None; self.image_cache = {}; self._preview_busy = False; self._revision = 0
        self.source = tk.StringVar(value='Current RGB designs')
        colors = list(dict.fromkeys(d['color'] for d in owner.designs))
        self.color = tk.StringVar(value=colors[0] if colors else '')
        self.location = tk.StringVar(); self.detail_datasets = {}; self.detail_metadata = {}
        self.path = tk.StringVar(); self.seed = tk.StringVar(value='17'); self.restarts = tk.StringVar(value='3')
        self.family = tk.StringVar(value='random'); self.model = tk.StringVar(value='additive')
        self.plot = tk.StringVar(value='Dashboard'); self.address = tk.IntVar(value=0)
        self.flexible = tk.BooleanVar(value=True)
        self.pca_seed = tk.BooleanVar(value=True)
        self.extra_rounds = tk.StringVar(value='0')
        self.round_info = tk.StringVar()
        self.synthesis_mode = tk.StringVar(value='Split-pool')
        self.independent_response = tk.StringVar(value='RGB ratio (complete occupancy)')
        self.split_response = tk.StringVar(value='Concentration-dependent capture (C50)')
        self.response_explanation = tk.StringVar(value='C50 applies to each round. Occupied ON/OFF sites cannot change in later rounds.')
        self.sample_count = tk.StringVar(value='300'); self.example_seed = tk.StringVar(value='42')
        self.status = tk.StringVar(value='Choose targets and a model, then visualize the selected model.')
        self.summary = tk.StringVar()
        self.vector_index = tk.IntVar(value=1); self._vector_axes = []; self._vector_count = 0
        self._vector_cells = {}; self._dashboard_key = None; self.dashboard_sections = {}
        self._dashboard_rendered = {}
        self._primary_dashboard_key = None; self._build_artists = {}
        self.build_step = tk.IntVar(value=1); self.build_status = tk.StringVar(value='Fit a model to explore canvas construction.')
        self._build_key = None; self._build_plan = None; self._build_count = 1; self._build_pickers = []
        self.physical_enabled = tk.BooleanVar(value=True)
        # Stock concentrations come from the existing RGB source settings.
        volume = float(owner.final_volume.get())*1000
        amount_fmol = float(owner.base_target.get())*volume/1000
        self.recipe_variables = {key: tk.StringVar(value=str(value)) for key, value in dict(
            vector_volume_nl=volume, vector_amount_fmol=amount_fmol,
            branch_volume_nl=volume, branch_budget_nl=volume,
            capture_c50_nm=float(owner.base_target.get())).items()}
        form = ttk.Frame(self, padding=12); form.pack(fill='both', expand=True)
        ttk.Label(form, text='Compare PCA, additive, irreversible ON-only, and ON/OFF encoding.').pack(anchor='w')
        bar = ttk.Frame(form); bar.pack(fill='x', pady=6)
        ttk.Combobox(bar, textvariable=self.source, values=('Current RGB designs', 'CSV / NumPy matrix', 'Synthetic benchmarks'), state='readonly', width=24).pack(side='left')
        ttk.Button(bar, text='Choose CSV / NPY…', command=self.browse).pack(side='left', padx=8)
        ttk.Label(bar, text='Synthesis mode').pack(side='left', padx=(12, 4))
        self.mode_picker = ttk.Combobox(bar, textvariable=self.synthesis_mode,
            values=('Split-pool', 'Independent wells'), state='readonly', width=20)
        self.mode_picker.pack(side='left'); self.mode_picker.bind('<<ComboboxSelected>>', self._mode_changed)
        ttk.Label(form, textvariable=self.path, wraplength=1000).pack(anchor='w')
        response_bar = ttk.Frame(form); response_bar.pack(fill='x', pady=4)
        ttk.Label(response_bar, text='Recipe capture response').pack(side='left')
        self.response_picker = ttk.Combobox(response_bar, textvariable=self.split_response,
            values=('RGB ratio (complete occupancy)', 'Concentration-dependent capture (C50)'), state='readonly', width=38)
        self.response_picker.pack(side='left', padx=6)
        self.response_picker.bind('<<ComboboxSelected>>', self._response_changed)
        ttk.Label(response_bar, textvariable=self.response_explanation, wraplength=550).pack(side='left')
        bar = ttk.Frame(form); bar.pack(fill='x', pady=4)
        from split_pool_encoding.datasets import FAMILIES
        ttk.Label(bar, text='Synthetic dataset').pack(side='left')
        ttk.Combobox(bar, textvariable=self.family, values=FAMILIES, state='readonly', width=16).pack(side='left', padx=6)
        for label, variable in (('Seed', self.seed), ('Restarts', self.restarts)):
            ttk.Label(bar, text=label).pack(side='left', padx=(12, 4)); ttk.Entry(bar, textvariable=variable, width=6).pack(side='left')
        self.assignment_check = ttk.Checkbutton(bar, text='Optimize address assignment', variable=self.flexible)
        self.assignment_check.pack(side='left', padx=12)
        ttk.Label(form, text='Split–pool fits all RGB designs together with one shared K library; color selects detail plots. Synthetic targets use 256 × 40.').pack(anchor='w', pady=4)
        basis_bar = ttk.Frame(form); basis_bar.pack(fill='x', pady=4)
        ttk.Label(basis_bar, text='Shared library size K').pack(side='left')
        self.k_picker = ttk.Combobox(basis_bar, textvariable=self.basis_k, values=('Unrestricted', '1', '2', '4', '8', '16', '32', '64', '128', '256'), width=13)
        self.k_picker.pack(side='left', padx=6); self.k_picker.bind('<<ComboboxSelected>>', self._select_k)
        self.k_picker.bind('<Return>', self._select_k)
        ttk.Label(basis_bar, text='K sweep').pack(side='left', padx=(12, 4))
        ttk.Entry(basis_bar, textvariable=self.k_sweep, width=20).pack(side='left')
        self.sweep_button = ttk.Button(basis_bar, text='Fit RMSE vs K', command=lambda: self.start(sweep=True))
        self.sweep_button.pack(side='left', padx=8)
        ttk.Checkbutton(basis_bar, text='Try PCA-seeded library (+1 fit)', variable=self.pca_seed).pack(side='left', padx=6)
        round_bar = ttk.Frame(form); round_bar.pack(fill='x', pady=4)
        ttk.Label(round_bar, text='Extra split–pool rounds').pack(side='left')
        self.round_picker = ttk.Spinbox(round_bar, from_=0, to=11, textvariable=self.extra_rounds, width=5)
        self.round_picker.pack(side='left', padx=6)
        ttk.Label(round_bar, textvariable=self.round_info).pack(side='left')
        self.extra_rounds.trace_add('write', self._update_round_info)
        self.source.trace_add('write', self._update_round_info)
        self._update_round_info()
        physical_bar = ttk.Frame(form); physical_bar.pack(fill='x', pady=4)
        self.physical_check = ttk.Checkbutton(physical_bar, text='25 nL stock preparation + destination transfers', variable=self.physical_enabled)
        self.physical_check.pack(side='left')
        ttk.Button(physical_bar, text='Stocks, dilution & capture…', command=self._recipe_dialog).pack(side='left', padx=8)
        ttk.Label(physical_bar, text='ON-only / ON/OFF recipes; additive and PCA are ideal references.').pack(side='left')
        actions = ttk.Frame(form); actions.pack(fill='x', pady=4)
        self.run_button = ttk.Button(actions, text='Visualize analysis', command=self.start); self.run_button.pack(side='left')
        ttk.Button(actions, text='Cancel', command=self.cancel.set).pack(side='left', padx=8)
        ttk.Label(actions, text='Plots stay in this tab. Use the plot toolbar to save an image if needed.').pack(side='left')
        ttk.Label(form, textvariable=self.status, wraplength=1000).pack(anchor='w', pady=6)
        progress_bar = ttk.Frame(form); progress_bar.pack(fill='x', pady=(0, 6))
        self.progress_value = tk.DoubleVar(value=0)
        self.progress_text = tk.StringVar(value='Ready')
        self.progress_bar = ttk.Progressbar(progress_bar, variable=self.progress_value, maximum=100, mode='determinate')
        self.progress_bar.pack(side='left', fill='x', expand=True)
        ttk.Label(progress_bar, textvariable=self.progress_text, width=24).pack(side='left', padx=8)

        controls = ttk.Frame(form); controls.pack(fill='x', pady=4)
        self.plot_picker = ttk.Combobox(controls, textvariable=self.plot, state='readonly', width=24,
            values=('Dashboard', 'Selected origami', 'Selected route evolution', 'Canvas construction', 'PCA seed comparison', 'Full synthesis', 'Well trajectory', 'Branch walkthrough', 'K vectors', 'K stock preparation', 'Transfer recipe', 'RMSE vs K', 'Image Monte Carlo', 'Model comparison', 'Target and reconstruction', 'Analog canvas', 'Sampling error', 'Convergence'))
        self.plot_picker.pack(side='left'); self.plot_picker.bind('<<ComboboxSelected>>', self._plot_changed)
        ttk.Label(controls, text='Model').pack(side='left', padx=(12, 4))
        self.model_picker = ttk.Combobox(controls, textvariable=self.model, state='readonly', width=14, values=('additive', 'on_only', 'on_off', 'pca'))
        self.model_picker.pack(side='left'); self.model_picker.bind('<<ComboboxSelected>>', lambda e: self.draw())
        ttk.Label(controls, text='Design index').pack(side='left', padx=(12, 4))
        self.address_picker = ttk.Spinbox(controls, from_=0, to=0, textvariable=self.address, width=6, command=self.draw)
        self.address_picker.pack(side='left'); self.address_picker.bind('<Return>', lambda e: self.draw())
        self.address_picker.bind('<FocusOut>', lambda e: self.draw())
        ttk.Label(controls, text='Vector').pack(side='left', padx=(12, 4))
        self.vector_picker = ttk.Spinbox(controls, from_=1, to=1, textvariable=self.vector_index,
                                         width=6, state='disabled', command=self.draw)
        self.vector_picker.pack(side='left')
        self.vector_picker.bind('<Return>', lambda e: self.draw())
        self.vector_picker.bind('<FocusOut>', lambda e: self.draw())
        ttk.Button(controls, text='Previous 8', command=lambda: self._vector_page(-1)).pack(side='left', padx=(6, 0))
        ttk.Button(controls, text='Next 8', command=lambda: self._vector_page(1)).pack(side='left')

        location_bar = ttk.Frame(form); location_bar.pack(fill='x', pady=4)
        ttk.Label(location_bar, text='Channel for detail plots').pack(side='left', padx=(0, 4))
        self.color_picker = ttk.Combobox(location_bar, textvariable=self.color, values=colors, state='readonly', width=12)
        self.color_picker.pack(side='left'); self.color_picker.bind('<<ComboboxSelected>>', self._channel_changed)
        ttk.Label(location_bar, text='X/Y design location').pack(side='left', padx=(12, 0))
        self.location_picker = ttk.Combobox(location_bar, textvariable=self.location, state='readonly', width=44)
        self.location_picker.pack(side='left', padx=6)
        self.location_picker.bind('<<ComboboxSelected>>', self._location_changed)
        self._update_locations()
        self._build_controls(form)
        sample_bar = ttk.Frame(form); sample_bar.pack(fill='x', pady=4)
        ttk.Label(sample_bar, text='Origami / barcode / color').pack(side='left')
        ttk.Entry(sample_bar, textvariable=self.sample_count, width=7).pack(side='left', padx=5)
        ttk.Label(sample_bar, text='Example seed').pack(side='left', padx=(12, 4))
        ttk.Entry(sample_bar, textvariable=self.example_seed, width=8).pack(side='left')
        ttk.Button(sample_bar, text='Update image', command=self.draw).pack(side='left', padx=8)
        ttk.Button(sample_bar, text='New finite-count example', command=self.new_example).pack(side='left')
        ttk.Label(form, textvariable=self.summary, wraplength=1050).pack(anchor='w', pady=4)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        from split_pool_plot_canvas import plot_canvas
        self.figure = Figure(figsize=(10, 6), constrained_layout=True)
        primary_plot = ttk.Frame(form); primary_plot.pack(fill='both', expand=True)
        self.canvas = plot_canvas(self.figure, master=primary_plot)
        self.canvas.mpl_connect('button_press_event', self._pick_vector)
        self.toolbar = NavigationToolbar2Tk(self.canvas, primary_plot, pack_toolbar=False)
        self.toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.dashboard_frame = ttk.Frame(form)
        self.draw()

    def _recipe_dialog(self):
        window = tk.Toplevel(self); window.title('Split–pool stocks, dilution and capture')
        form = ttk.Frame(window, padding=12); form.pack(fill='both', expand=True)
        ttk.Label(form, text=f"ON stock: {self.owner.replacement_stock.get()} nM • OFF stock: {self.owner.base_stock.get()} nM\n"
            'Reused from the RGB generator. Both preparation and branch transfers use 25 nL droplets.').grid(row=0, column=0, columnspan=2, sticky='w')
        fields = [('vector_volume_nl', 'Final diluted volume of each vector well (nL)'),
                  ('vector_amount_fmol', 'Full-site reference staple amount (fmol)'),
                  ('branch_volume_nl', 'Final volume of each destination well (nL)'),
                  ('branch_budget_nl', 'Maximum vector volume added to each destination (nL)'),
                  ('capture_c50_nm', 'Assumed C50 for one capture step (nM; uncalibrated)')]
        for row, (key, label) in enumerate(fields, 1):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky='w', pady=5)
            ttk.Entry(form, textvariable=self.recipe_variables[key], width=16).grid(row=row, column=1, padx=8)
        ttk.Label(form, text='The optimized ON/OFF doses are dispensed into one well per vector, then diluted to its final volume.\n'
            'At a fixed reference amount, increasing that volume dilutes the vector; it does not add more staple.\n'
            'Remaining branch volume includes origami and buffer. Source wells cannot be overdrawn.\n'
            'Assumption: equal ON/OFF capture rates, q = 1 − exp(−ln(2) × total concentration / C50).\n'
            'C50 is adjustable and has NOT been measured. All changes require a new fit.',
            wraplength=720).grid(row=6, column=0, columnspan=2, sticky='w', pady=10)
        ttk.Button(form, text='Close', command=window.destroy).grid(row=7, column=1, sticky='e')

    def _read_recipe_settings(self, color=None):
        from split_pool_encoding.quantization import RecipeSettings
        values = {key: float(var.get()) for key, var in self.recipe_variables.items()}
        amount = values.pop('vector_amount_fmol')
        if not 0 < amount < float('inf'):
            raise ValueError('Reference staple amount must be finite and positive.')
        if values['vector_volume_nl'] <= 0:
            raise ValueError('Vector well volume must be positive.')
        return RecipeSettings(on_stock_nm=float(self.owner.replacement_stock.get()),
            off_stock_nm=float(self.owner.base_stock.get()),
            vector_max_nm=amount*1000/values['vector_volume_nl'], **values).validate()

    def _update_locations(self):
        designs = self.image_designs if self.image_designs is not None else self.owner.designs
        if self.data is not None and self.image_designs is None:
            designs = []
        group = [d for d in designs if d['color'] == self.color.get()]
        labels = []
        for index, design in enumerate(group):
            if 'x_shift' in design and 'y_shift' in design:
                label = f"X {design['x_shift']}, Y {design['y_shift']}"
                if 'quadrant' in design:
                    label += f" • tile {design['quadrant']}"
            else:
                label = f"Design {index}"
            labels.append(label)
        previous = self.location.get()
        self.location_picker.configure(values=labels, state='readonly' if labels else 'disabled')
        if labels:
            row = labels.index(previous) if previous in labels else 0
            self.location_picker.current(row); self.address.set(row)
            self.address_picker.configure(to=len(labels)-1)
        else:
            self.location.set('')

    def _channel_changed(self, _event=None):
        self._update_locations()
        self.draw()

    def _location_changed(self, _event=None):
        row = self.location_picker.current()
        if row >= 0:
            self.address.set(row)
            if self.plot.get() not in ('Dashboard', 'Selected origami', 'Selected route evolution', 'Canvas construction', 'Branch walkthrough', 'Well trajectory', 'Full synthesis'):
                self.plot.set('Analog canvas')
            self.draw()

    def _detail_result(self, result):
        if self.image_designs and result.get('color_fits'):
            return {**result['color_fits'][self.color.get()], 'color_fits': result['color_fits'], 'basis_label': result.get('basis_label', 'Unrestricted'), 'physical_mode': result.get('physical_mode', False)}
        return result

    @staticmethod
    def _k_label(k):
        return 'Unrestricted' if k is None else str(k)

    def _update_round_info(self, *_args):
        if self.synthesis_mode.get() == 'Independent wells':
            self.round_info.set('One well per design; mix K stocks, then capture once. K is per color. No split–pool rounds.')
            return
        count = len(self.owner.designs) if self.source.get() == 'Current RGB designs' else 256 if self.source.get() == 'Synthetic benchmarks' else None
        try:
            extra = int(self.extra_rounds.get())
            if extra < 0:
                raise ValueError
            if count:
                minimum = max(1, (count-1).bit_length())
                total = minimum+extra
                self.round_info.set(f'{minimum} minimum + {extra} extra = {total} rounds; {2*total} shared branch mixtures across all colors. '
                                    + ('Maximum is 12 rounds.' if total > 12 else 'Click a fit button to apply.'))
            else:
                self.round_info.set('Adds rounds with both branches; minimum determined from targets. Maximum 12 total.')
        except ValueError:
            self.round_info.set('Enter a nonnegative integer. Maximum 12 total rounds.')

    def _mode_changed(self, _event=None):
        independent = self.synthesis_mode.get() == 'Independent wells'
        for widget in (self.round_picker, self.assignment_check, self.physical_check):
            widget.configure(state='disabled' if independent else 'normal')
        self.model_picker.configure(values=('on_only', 'on_off', 'pca') if independent else ('additive', 'on_only', 'on_off', 'pca'))
        if independent:
            self.physical_enabled.set(True)
            if self.model.get() == 'additive':
                self.model.set('on_off')
        self.response_picker.configure(state='readonly', textvariable=self.independent_response if independent else self.split_response)
        self._response_changed()
        self._update_round_info()
        self.status.set('Click Visualize analysis or Fit RMSE vs K to fit the selected synthesis mode. Existing plots remain from the last fit.')

    def _response_changed(self, _event=None):
        independent = self.synthesis_mode.get() == 'Independent wells'
        response = self.independent_response.get() if independent else self.split_response.get()
        if response == 'RGB ratio (complete occupancy)':
            self.response_explanation.set('Ideal ON/(ON+OFF) ratio; complete occupancy wherever staples are present. '+
                ('Matches normal RGB preview assumptions.' if independent else 'Each round fills remaining sites; later rounds cannot overwrite them.'))
        else:
            self.response_explanation.set('Adjustable, uncalibrated C50 '+('for one capture step.' if independent else 'per round. Already occupied sites remain ON/OFF.'))

    def _select_k(self, _event=None):
        label = self.basis_k.get().strip()
        if label in self.k_results:
            self.results = self.k_results[label]
            self._revision += 1; self.image_cache.clear()
            self.draw()
        else:
            self.status.set('This K has not been fitted yet. Click Visualize analysis or Fit RMSE vs K.')

    @staticmethod
    def _parse_k_values(text):
        from split_pool_encoding.library import BasisLibrary
        values = []
        for raw in text.split(','):
            k = None if raw.strip().lower() == 'unrestricted' else int(raw.strip())
            if k is not None:
                BasisLibrary('additive', 1, 1, k)
            if k not in values:
                values.append(k)
        if len(values) > 16:
            raise ValueError('Use at most 16 K values per sweep.')
        return values

    def _plot_changed(self, _event=None):
        # Plot selection only displays results; fitting requires an explicit button.
        self.draw()

    def draw_k_curve(self):
        import numpy as np
        ax = self.figure.subplots()
        def rmse(result):
            fits = list(result['color_fits'].values()) if result.get('color_fits') else [result]
            return np.sqrt(sum(f['mse']*f['prediction'].size for f in fits)/sum(f['prediction'].size for f in fits))
        for model in ('additive', 'on_only', 'on_off', 'pca'):
            points = sorted((int(k), rmse(bucket[model])) for k, bucket in self.k_results.items()
                            if k != 'Unrestricted' and model in bucket)
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], 'o-', label=model+(' (25 nL)' if model in ('on_only', 'on_off') and any(b.get(model, {}).get('physical_mode') for b in self.k_results.values()) else ' (reference)' if model in ('pca', 'additive') else ''))
            reference = self.k_results.get('Unrestricted', {}).get(model)
            if reference and model != 'pca':
                ax.axhline(rmse(reference), linestyle='--', label=model+' unrestricted')
        ax.set(xlabel='Learned shared basis-library size K', ylabel='Encoding RMSE',
               title='All image colors combined • best fitted candidate at each K')
        if ax.lines:
            ax.legend()
        try:
            requested = self._parse_k_values(self.k_sweep.get())
        except ValueError:
            requested = []
        ticks = sorted({k for k in requested if k is not None} | {int(k) for k in self.k_results if k != 'Unrestricted'})
        if ticks:
            ax.set_xticks(ticks)
            padding = max(.5, (ticks[-1]-ticks[0])*.05)
            ax.set_xlim(ticks[0]-padding, ticks[-1]+padding)
        independent = any(f.get('synthesis_mode') == 'independent_wells' for bucket in self.k_results.values() for f in bucket.values())
        models = {'on_only', 'on_off', 'pca'} if independent else {'additive', 'on_only', 'on_off', 'pca'}
        complete = [self._k_label(k) for k in requested
                    if models.issubset(self.k_results.get(self._k_label(k), {}))]
        progress = f"Completed {len(complete)}/{len(requested)} K values: {', '.join(complete) or 'none yet'}. "
        if not ax.lines:
            ax.text(.5, .5, 'Click Fit RMSE vs K to run the K sweep.', transform=ax.transAxes, ha='center')
        self.summary.set(progress + 'Click Fit RMSE vs K to run or refresh the sweep. RMSE vs K: each K learns its own basis vectors, branch coefficients, and assignment. Rounds are unchanged. '
                         'PCA is a fixed reference. Local optimization can produce non-monotonic curves.')

    def browse(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[('Analog matrix', '*.csv *.npy')])
        if path:
            self.path.set(path); self.source.set('CSV / NumPy matrix')

    def start(self, sweep=False):
        if self.running:
            return
        try:
            from split_pool_encoding.datasets import load, from_designs, generate
            from split_pool_encoding.experiment import DEFAULTS
            cfg = {**DEFAULTS, 'seed': int(self.seed.get()), 'restarts': int(self.restarts.get())}
            if cfg['seed'] < 0 or not 1 <= cfg['restarts'] <= 100:
                raise ValueError('Use a nonnegative seed and 1–100 restarts.')
            source = self.source.get(); self.metadata = None; self.synthetic_truth = None
            image_designs = None; datasets = {}; metadata_by_color = {}
            if source == 'Current RGB designs':
                image_designs = deepcopy(self.owner.designs)
                for color in dict.fromkeys(d['color'] for d in image_designs):
                    datasets[color], metadata_by_color[color] = from_designs(image_designs, color)
                self.color_picker.configure(values=list(datasets))
                if datasets and self.color.get() not in datasets:
                    self.color.set(next(iter(datasets)))
                data, self.metadata = from_designs(image_designs, self.color.get())
                render_options = dict(fwhm_nm=float(self.owner.gaussian_fwhm.get()),
                    exposure=float(self.owner.gaussian_exposure.get()), linear_light=self.owner.linear_light.get(),
                    normalize_designs=self.owner.normalize_designs.get(), compensate_area=self.owner.compensate_area.get())
            elif source == 'CSV / NumPy matrix':
                data = load(self.path.get())
            else:
                data, self.synthetic_truth = generate(self.family.get(), seed=cfg['seed'])
            ks = self._parse_k_values(self.k_sweep.get() if sweep else self.basis_k.get())
            independent = self.synthesis_mode.get() == 'Independent wells'
            synthesis_mode = 'independent_wells' if independent else 'split_pool'
            shared_pool = bool(image_designs) and not independent
            if shared_pool:
                from split_pool_encoding.shared_pool import combined_targets
                fit_datasets = {'All RGB designs': combined_targets(datasets, metadata_by_color)}
            else:
                fit_datasets = datasets or {'input': data}
            response = self.independent_response.get() if independent else self.split_response.get()
            response_model = 'rgb_ratio' if response == 'RGB ratio (complete occupancy)' else 'capture'
            extra_rounds = 0 if independent else int(self.extra_rounds.get())
            target_count = len(image_designs) if image_designs else len(data)
            minimum_rounds = max(1, (target_count-1).bit_length())
            bits = minimum_rounds+extra_rounds
            if extra_rounds < 0 or bits > 12:
                raise ValueError(f'Use 0–{12-minimum_rounds} extra rounds for these targets (maximum 12 total rounds).')
            physical = independent or self.physical_enabled.get()
            recipe_settings = self._read_recipe_settings() if physical else None
            if physical and None in ks:
                raise ValueError('Choose finite K values for discrete vector recipes, or disable the 25 nL recipe option for unrestricted references.')
            if self.synthetic_truth is not None and 'k' in self.synthetic_truth:
                if independent or not physical:
                    raise ValueError('Exact-K datasets require Split pooling with 25 nL recipes enabled.')
                data, self.synthetic_truth = generate(self.family.get(), seed=cfg['seed'],
                    recipe_settings=recipe_settings, response_model=response_model, extra_rounds=extra_rounds)
                fit_datasets = {'input': data}
            flexible = self.flexible.get()
            pca_seed = self.pca_seed.get()
            selected_model = self.model.get()
            available_models = ('on_off', 'on_only', 'pca') if independent else ('additive', 'on_only', 'on_off', 'pca')
            if selected_model not in available_models:
                raise ValueError('Choose a supported model for the selected synthesis mode.')
        except (ValueError, OSError, ImportError) as exc:
            messagebox.showerror('Split–pool analysis', str(exc), parent=self); return
        self.image_designs = image_designs; self.image_cache.clear(); self._revision += 1
        self.k_results.clear(); self.basis_k.set(self._k_label(ks[0]))
        if image_designs:
            self.render_options = render_options
        self.plot.set('Dashboard')
        self.detail_datasets = datasets; self.detail_metadata = metadata_by_color
        if sweep:
            self.plot.set('RMSE vs K')
        selected_color = self.color.get()
        self.data = data; self.results.clear()
        self.address.set(0); self.address_picker.configure(to=len(data)-1)
        self._update_locations()
        self.cancel.clear(); self.running = True; self.run_button.configure(state='disabled'); self.sweep_button.configure(state='disabled')
        self.progress_value.set(0); self.progress_text.set('0% • fitting budget')
        if independent:
            self.round_info.set(f'{target_count} destination wells, one capture step each. K is per color; barcodes are separate identifiers.')
        else:
            self.round_info.set(f'{minimum_rounds} minimum + {extra_rounds} extra = {bits} rounds; {2*bits} shared branch mixtures across all colors.')
        self.status.set(f'Starting independent-well analysis for {target_count} designs…' if independent else f'Starting analysis with {bits} rounds…'); self.draw()
        def worker():
            try:
                from time import monotonic
                from split_pool_encoding.models import MODELS
                from split_pool_encoding.optimization import fit
                from split_pool_encoding.evaluation import metrics, sampling
                order = ('on_off', 'on_only', 'pca') if independent else ('on_off', 'on_only', 'additive', 'pca') if physical else (('additive', 'on_only', 'on_off', 'pca') if image_designs else MODELS)
                if not sweep:
                    order = (selected_model,)
                total_jobs = len(ks)*len(order)*len(fit_datasets)
                completed_jobs = 0
                for k in ks:
                    for model in order:
                        color_fits = {}
                        # Reserve addresses globally across all color libraries.
                        available = list(range(2**bits))
                        for color, target_data in fit_datasets.items():
                            if self.cancel.is_set():
                                raise InterruptedError('Analysis cancelled. Completed models remain available.')
                            self.events.put(('progress', f'K={self._k_label(k)} • fitting {model} • {color}…'))
                            last_update = 0.
                            def on_progress(fraction, detail):
                                nonlocal last_update
                                now = monotonic()
                                if fraction in (0., 1.) or now-last_update >= .1:
                                    last_update = now
                                    percentage = 98*(completed_jobs+fraction)/total_jobs
                                    self.events.put(('fit_progress', (percentage,
                                        f'K={self._k_label(k)} • {model} • {color} • {detail}')))
                            direct_source_nl = None
                            if independent and image_designs and model == 'on_off':
                                group = [d for d in image_designs if d['color'] == color]
                                if all('replacement_nl' in site and 'original_nl' in site for d in group for site in d['image_sites']):
                                    direct_source_nl = [[[site['replacement_nl'], site['original_nl']] for site in d['image_sites']] for d in group]
                            color_fits[color] = fit(target_data, model, seed=cfg['seed'], restarts=cfg['restarts'],
                                alternations=cfg['alternations'], maxiter=cfg['maxiter'], flexible=flexible, cancel=self.cancel, allow_partial=bool(image_designs) or extra_rounds > 0, basis_k=k,
                                recipe_settings=recipe_settings if model in ('on_only', 'on_off') else None, progress=on_progress,
                                address_bits=bits, allowed_addresses=available, pca_seed=pca_seed, synthesis_mode=synthesis_mode,
                                response_model=response_model, direct_source_nl=direct_source_nl)
                            if available is not None:
                                used = set(color_fits[color]['address_indices'])
                                available = [address for address in available if address not in used]
                            color_fits[color].update(minimum_rounds=minimum_rounds, extra_rounds=extra_rounds)
                            completed_jobs += 1
                        if shared_pool:
                            from split_pool_encoding.shared_pool import color_views
                            color_fits = color_views(color_fits['All RGB designs'], datasets)
                        for color, fitted in color_fits.items():
                            target_data = datasets[color] if image_designs else data
                            target = target_data[fitted['assignment']]
                            fitted['metrics'] = metrics(target, fitted['prediction'])
                            fitted['noise'] = [] if model == 'pca' else sampling(target, fitted['prediction'],
                                counts=cfg['sample_counts'], repeats=cfg['measurement_repeats'], seed=cfg['seed'])
                        result = dict(color_fits[selected_color if image_designs else 'input'])
                        result['color_fits'] = color_fits if image_designs else None
                        result['basis_label'] = self._k_label(k)
                        result['physical_mode'] = physical
                        self.events.put(('result', (self._k_label(k), result)))
                self.events.put(('fit_progress', (99., 'Finalizing plots…')))
                self.events.put(('done', 'Analysis ready. Choose a plot, model, or address to explore the results.'))
            except Exception as exc:
                self.events.put(('error', str(exc)))
        Thread(target=worker, daemon=True, name='split-pool-analysis').start()
        if self._poll_id is None:
            self._poll_id = self.after(100, self.poll)

    def poll(self):
        self._poll_id = None
        # Drain queued updates before drawing. A second pass handles a preview
        # that completed during rendering, without rebuilding for every model.
        for _pass in range(2):
            needs_draw = False
            try:
                while True:
                    kind, value = self.events.get_nowait()
                    if kind == 'fit_progress':
                        percentage, detail = value
                        self.progress_value.set(max(self.progress_value.get(), percentage))
                        self.progress_text.set(f'{self.progress_value.get():.0f}% • fitting budget')
                        self.status.set(detail)
                    elif kind == 'preview':
                        key, preview, error = value
                        self._preview_busy = False
                        if key[0] == self._revision:
                            if len(self.image_cache) >= 6:
                                self.image_cache.clear()
                            self.image_cache[key] = (preview, error)
                        needs_draw = True
                    elif kind == 'result':
                        label, value = value
                        bucket = self.k_results.setdefault(label, {})
                        bucket[value['model']] = value
                        if label == self.basis_k.get():
                            self.results = bucket
                        needs_draw = True
                    else:
                        self.status.set(value)
                    if kind == 'done':
                        self.progress_value.set(100); self.progress_text.set('100% • fits complete')
                    if kind == 'error':
                        self.progress_text.set(f"{self.progress_value.get():.0f}% • {'cancelled' if self.cancel.is_set() else 'stopped'}")
                    if kind in ('done', 'error'):
                        self.running = False; self.run_button.configure(state='normal'); self.sweep_button.configure(state='normal')
            except Empty:
                pass
            if needs_draw:
                self.draw()
            if self.events.empty():
                break
        if (self.running or self._preview_busy) and self._poll_id is None:
            self._poll_id = self.after(100, self.poll)

    def draw(self):
        import numpy as np
        if self.model.get() in self.results:
            selected = self._detail_result(self.results[self.model.get()])
            try:
                row = max(0, min(len(selected['assignment'])-1, self.address.get()))
            except (ValueError, tk.TclError):
                row = 0
            self.address.set(row)
            if row < len(self.location_picker['values']):
                self.location_picker.current(row)
        dashboard = self.plot.get() == 'Dashboard'
        primary_key = None
        if dashboard and self.model.get() in self.results:
            current = self.results.get(self.model.get(), next(iter(self.results.values())))
            result = self._detail_result(current)
            data = self.detail_datasets[self.color.get()] if self.image_designs else self.data
            metadata = self.detail_metadata[self.color.get()] if self.image_designs else self.metadata
            primary_key = (self._revision, id(current), self.color.get(), self.address.get(), self.sample_count.get(), self.example_seed.get(),
                           tuple((k, id(v)) for k, v in self.image_cache.items()))
            if primary_key == self._primary_dashboard_key:
                self._draw_dashboard_sections(result, data, metadata)
                return
        if not dashboard or self.model.get() not in self.results:
            self.dashboard_frame.pack_forget()
            self._dashboard_key = None
        self._primary_dashboard_key = None
        self.canvas.reuse_layout = self.plot.get() == 'Canvas construction'
        build_key = (self._revision, id(selected['prediction']), self.color.get(), self.address.get()) if self.model.get() in self.results else None
        cached_build = self._build_artists.get(self.figure)
        reuse_build = (self.canvas.reuse_layout and cached_build is not None
                       and cached_build[0] == build_key and cached_build[1][0, 0] in self.figure.axes)
        if not reuse_build:
            self.figure.clear()
        self.summary.set('')
        self._vector_axes = []; self._vector_cells = {}; self.vector_picker.configure(state='disabled')
        if self.plot.get() == 'RMSE vs K':
            self.draw_k_curve()
        elif self.model.get() not in self.results:
            ax = self.figure.subplots(); ax.axis('off')
            ax.text(.5, .5, 'Results appear here as fits finish.\nProgress is shown above.' if self.running else 'Click Visualize analysis to fit the selected model.', ha='center', va='center', fontsize=16)
        else:
            result = self._detail_result(self.results.get(self.model.get(), next(iter(self.results.values()))))
            data = self.detail_datasets[self.color.get()] if self.image_designs else self.data
            metadata = self.detail_metadata[self.color.get()] if self.image_designs else self.metadata
            m = result['metrics']; target = data[result['assignment']]; p = result['prediction']
            self.summary.set(f"{result['model']} • K={result.get('basis_label', 'Unrestricted')}  •  RMSE {m['rmse']:.4f}  •  Max error {m['max_absolute_error']:.4f}  •  Within ±0.05: {m['fraction_within_0.05']:.1%}  •  Fit {result['runtime']:.2f}s")
            view = self.plot.get()
            if view == 'Dashboard':
                view = 'Image Monte Carlo'
            if view == 'Canvas construction':
                self.draw_canvas_build(result, data, metadata)
            elif view == 'PCA seed comparison':
                ax = self.figure.subplots()
                trials = result.get('trials', [])
                points = [(label, min(t['mse'] for t in trials if t.get('initialization') == kind)**.5)
                for kind, label in (('random', 'Best random start'), ('pca', 'PCA-seeded start'), ('target', 'One vector per design'), ('original_rgb_recipe', 'Original RGB recipe'))
                          if any(t.get('initialization') == kind for t in trials)]
                if result.get('pca_seed_tested') or result.get('exact_recipe_shortcut'):
                    ax.bar([label for label, _ in points], [value for _, value in points])
                    ax.set(ylabel='Final encoding RMSE', title='Same synthesis constraints • selected channel')
                    for i, (_, value) in enumerate(points):
                        ax.annotate(f'{value:.5f}', (i, value), xytext=(0, 4), textcoords='offset points', ha='center')
                    ax.margins(y=.2)
                else:
                    ax.axis('off'); ax.text(.5, .5, 'Enable Try PCA-seeded library and fit a finite K.\nSelect additive, ON-only, or ON/OFF to compare starts.', ha='center', va='center')
            elif view == 'Selected origami':
                self.draw_selected_origami(result)
            elif view == 'Selected route evolution':
                self.draw_route_evolution(result)
            elif view == 'Full synthesis':
                self.draw_complete_synthesis(result)
                if result.get('synthesis_mode') == 'independent_wells':
                    self.summary.set(self.summary.get()+' • Each row is a separate design well. Mix its K stocks, assemble/capture once, then pool completed populations. Barcode chemistry and final pooling volumes are not simulated.')
                else:
                    self.summary.set(self.summary.get()+' • One shared RGB population; gold traces the selected origami through its assigned branches. '
                    'Solid arrows show logical population flow, dashed arrows show K additions. Hover wells for volumes; use toolbar zoom. '
                    'Pooling, resplitting, washing, and carryover volumes are not specified by the current model; this is not a complete liquid-handling protocol.')
            elif view == 'Well trajectory':
                self.draw_well_trajectory(result, data, metadata)
            elif view == 'Branch walkthrough':
                if result.get('synthesis_mode') == 'independent_wells':
                    self.draw_well_trajectory(result, data, metadata)
                else:
                    self.draw_branch(result, data)
            elif view == 'K vectors':
                self.draw_vectors(result)
            elif view == 'K stock preparation':
                self.draw_preparation(result)
            elif view == 'Transfer recipe':
                self.draw_recipe(result)
            elif view == 'RMSE vs K':
                self.draw_k_curve()
            elif view == 'Image Monte Carlo':
                self.draw_image(result)
            elif view == 'Model comparison':
                ax = self.figure.subplots(); names = list(self.results)
                ax.bar(names, [self._detail_result(self.results[n])['metrics']['rmse'] for n in names], color=['#607d8b', '#e5a333', '#30a594', '#7864b5'][:len(names)])
                ax.set(ylabel='Reconstruction RMSE (lower is better)', title='Encoding accuracy • PCA is a linear reference')
            elif view == 'Target and reconstruction':
                for ax, values, title in zip(self.figure.subplots(3, 1), (target, p, abs(p-target)), ('Target in address order', 'Reconstruction', 'Absolute error')):
                    im = ax.imshow(values, aspect='auto', vmin=0, vmax=1, cmap='viridis')
                    ax.set(title=title, xlabel='Analog position', ylabel='Assigned row'); self.figure.colorbar(im, ax=ax)
            elif view == 'Analog canvas':
                try:
                    address = max(0, min(len(target)-1, self.address.get()))
                except (ValueError, tk.TclError):
                    address = 0
                # The selector identifies a target design, independent of the fitted permutation.
                row = address; self.address.set(row)
                if self.image_designs:
                    self.location_picker.current(row)
                assigned_row = int(np.flatnonzero(result['assignment'] == row)[0])
                ax = self.figure.subplots()
                ax.plot(data[row], 'o-', label='Target'); ax.plot(p[assigned_row], 'o-', label='Reconstruction')
                label = f'target row {row}'
                if metadata:
                    label = metadata['target_names'][row]
                actual_address = result.get('address_indices', np.arange(len(target)))[assigned_row]
                ax.set(title=f'Address {actual_address} → {label}', xlabel='Analog position', ylabel='ON probability'); ax.legend()
            elif view == 'Sampling error':
                ax = self.figure.subplots(); noise = result['noise']
                if noise:
                    counts = sorted({r['samples'] for r in noise})
                    for key, label in (('encoding_rmse', 'Encoding'), ('sampling_rmse', 'Sampling'), ('total_rmse', 'Total')):
                        values = [[r[key] for r in noise if r['samples'] == n] for n in counts]
                        ax.errorbar(counts, np.mean(values, axis=1), yerr=np.std(values, axis=1), marker='o', label=label)
                    ax.set(xlabel='Origami measured per barcode', ylabel='RMSE', title='Mean ± SD across repeated measurements'); ax.legend()
                else:
                    ax.axis('off'); ax.text(.5, .5, 'PCA is not a physical probability model.\nSelect another model to view sampling error.', ha='center', va='center')
            else:
                ax = self.figure.subplots()
                for name, item in self.results.items():
                    item = self._detail_result(item)
                    if item['history']:
                        ax.plot([h['mse'] for h in item['history']], label=name)
                ax.set(xlabel='Logged optimization step', ylabel='MSE', title='Learning curves for selected restarts')
                if ax.lines:
                    ax.legend()
                else:
                    ax.text(.5, .5, 'PCA uses direct SVD and has no learning curve.', transform=ax.transAxes, ha='center')
            if result.get('pca_seed_tested'):
                self.summary.set(self.summary.get()+f" • PCA-seeded candidate tested; retained {result['initialization']} start after constrained fitting.")
            if result.get('synthesis_mode') == 'independent_wells':
                self.summary.set(self.summary.get()+' • Independent wells: one design per destination, one capture step; K stocks are shared within each color. Barcode assembly is not simulated.')
                if result.get('model') != 'pca':
                    self.summary.set(self.summary.get()+(' Response: RGB ON/(ON+OFF), assuming complete occupancy.' if result.get('response_model') == 'rgb_ratio' else ' Response: concentration-dependent C50 capture.'))
                if 'original_recipe_mse' in result:
                    self.summary.set(self.summary.get()+f" Original RGB recipe RMSE under this response: {result['original_recipe_mse']**.5:.5f}.")
                if result.get('original_recipe_unavailable'):
                    self.summary.set(self.summary.get()+' Original recipe cannot fit these stock settings: '+result['original_recipe_unavailable'])
            elif 'extra_rounds' in result:
                self.summary.set(self.summary.get()+f" • {result['address_bits']} rounds ({result['extra_rounds']} extra), each with branches 0 and 1.")
            if result.get('recipe', {}).get('settings') and 'address_bits' in result and result.get('synthesis_mode') != 'independent_wells':
                if result.get('response_model') == 'rgb_ratio':
                    self.summary.set(self.summary.get()+' • Saturating ratio at each round: first nonzero exposure fills a site; later rounds cannot overwrite it.')
                else:
                    import math
                    s = result['recipe']['settings']
                    single = -math.expm1(-math.log(2)*s['vector_max_nm']*s['branch_budget_nl']/s['branch_volume_nl']/s['capture_c50_nm'])
                    self.summary.set(self.summary.get()+f' • C50 per round: continuous-reference maximum capture {single:.1%} of empty sites per round; optimistic {result["address_bits"]}-round ON ceiling {1-(1-single)**result["address_bits"]:.1%} (ignores OFF blocking, stock sharing, and quantization).')
            if result.get('retained_initial_recipe'):
                self.summary.set(self.summary.get()+' Retained the initial feasible recipe because optimization followed by rounding gave higher error.')
            if result.get('exact_recipe_shortcut'):
                self.summary.set(self.summary.get()+' Original recipe already matches the targets; optimization was skipped.')
        self.toolbar.update(); self.canvas.draw_idle()
        if dashboard and self.model.get() in self.results:
            self._primary_dashboard_key = primary_key
            self._draw_dashboard_sections(result, data, metadata)
        elif self.model.get() not in self.results:
            self.dashboard_frame.pack_forget(); self._dashboard_key = None

    def _pick_vector(self, event):
        toolbar = getattr(getattr(event, 'canvas', None), 'toolbar', self.toolbar)
        if toolbar.mode:
            return
        if event.inaxes not in self._vector_cells:
            return
        self.vector_index.set(self._vector_cells[event.inaxes])
        self.draw()

    def _vector_sites(self):
        if self.image_designs:
            return next(d['image_sites'] for d in self.image_designs if d['color'] == self.color.get())
        return None

    def _canvas_rotation(self):
        design = self._selected_design()
        return design.get('origami_rotation_degrees_clockwise', 0) if design else 0

    def _vector_page(self, direction):
        count = max(1, self._vector_count)
        try:
            current = self.vector_index.get()
        except (ValueError, tk.TclError):
            current = 1
        self.vector_index.set(max(1, min(count, current+8*direction)))
        self.draw()

    def draw_vectors(self, result):
        import numpy as np
        from split_pool_vectors import vector_panels, draw_site_canvas
        display = vector_panels(result)
        if display is None:
            ax = self.figure.subplots(); ax.axis('off')
            ax.text(.5, .5, 'Choose a fitted finite K to inspect its shared vectors.', ha='center', va='center')
            return
        panels = display['panels']; count = panels[0][1].shape[0]
        self._vector_count = count
        try:
            selected = max(1, min(count, self.vector_index.get()))
        except (ValueError, tk.TclError):
            selected = 1
        self.vector_index.set(selected); self.vector_picker.configure(to=count, state='normal')
        first = ((selected-1)//8)*8; last = min(first+8, count)
        maximum = max(float(np.max(np.abs(values))) for _, values in panels)
        low = -max(maximum, 1e-12) if display['kind'] == 'signed' else 0
        high = 1 if display['kind'] == 'probability' else max(maximum, 1e-12)
        cmap = 'coolwarm' if display['kind'] == 'signed' else 'viridis'
        grid = self.figure.add_gridspec(1+(last-first+3)//4, 1, height_ratios=[1.6]+[1]*((last-first+3)//4))
        detail_grid = grid[0].subgridspec(1, len(panels))
        sites = self._vector_sites()
        note = 'Physical origami frame' if sites else 'Schematic matrix (no spatial metadata)'
        for column, (title, values) in enumerate(panels):
            ax = self.figure.add_subplot(detail_grid[column])
            im = draw_site_canvas(ax, values[selected-1], sites, self._canvas_rotation(), vmin=low, vmax=high, cmap=cmap)
            ax.set(title=f'K{selected}: {title}', xlabel='', ylabel='')
            self.figure.colorbar(im, ax=ax, label=display['unit'])
        for offset, index in enumerate(range(first, last)):
            row = offset//4+1
            # Four cards per row, with the ON/OFF (or signed) components side by side.
            subgrid = grid[row].subgridspec(1, 4*len(panels))
            for component, (title, values) in enumerate(panels):
                ax = self.figure.add_subplot(subgrid[offset%4*len(panels)+component])
                draw_site_canvas(ax, values[index], sites, self._canvas_rotation(), vmin=low, vmax=high, cmap=cmap)
                ax.set(title=f'K{index+1} '+title.split()[0], xticks=[], yticks=[])
                ax.title.set_fontsize(8)
                if index == selected-1:
                    for spine in ax.spines.values():
                        spine.set_color('#e59d00'); spine.set_linewidth(2)
                self._vector_cells[ax] = index+1; self._vector_axes.append(ax)
        self.summary.set(f"{result['model']} • K{first+1}–K{last} of {count}; selected K{selected}. "
                         +note+'. Click a canvas or use Vector / Previous 8 / Next 8. '+display['note']
                         +f" Fit RMSE {result['mse']**.5:.4f}.")

    def _selected_design(self):
        from split_pool_selection import selected_design
        return selected_design(self.image_designs, self.color.get(), self.address.get())

    def draw_selected_origami(self, result):
        from split_pool_selection import draw_selected_origami
        draw_selected_origami(self.figure, self._selected_design(), result, self.address.get())

    def draw_complete_synthesis(self, result):
        from split_pool_synthesis import draw_combined_synthesis
        from split_pool_vectors import vector_panels
        display = vector_panels(result)
        if display is not None:
            self._vector_count = len(display['panels'][0][1])
            self.vector_index.set(max(1, min(self._vector_count, self.vector_index.get())))
            self.vector_picker.configure(state='normal', to=self._vector_count)
        draw_combined_synthesis(self.figure, result.get('color_fits') or {'Input dataset': result},
                                result, self.address.get(), self._vector_sites(), self.vector_index.get(), self._canvas_rotation(), design=self._selected_design())

    def draw_route_evolution(self, result):
        from split_pool_selection import draw_evolution
        from split_pool_vectors import vector_panels
        display = vector_panels(result)
        if display is not None and result['model'] in ('on_only', 'on_off'):
            self._vector_count = len(display['panels'][0][1])
            self.vector_index.set(max(1, min(self._vector_count, self.vector_index.get())))
            self.vector_picker.configure(state='normal', to=self._vector_count)
        draw_evolution(self.figure, result, self.address.get(), self._vector_sites(), self.vector_index.get(), self._canvas_rotation(), design=self._selected_design())

    def draw_preparation(self, result):
        from split_pool_vectors import draw_site_canvas, update_site_canvas
        recipe = result.get('recipe')
        if recipe is None:
            ax = self.figure.subplots(); ax.axis('off')
            ax.text(.5, .5, 'No physical stock recipe for this fit.\nEnable 25 nL recipes and fit ON-only or ON/OFF.', ha='center', va='center')
            return
        source = recipe['source_transfers_nl']
        selected = max(1, min(len(source), self.vector_index.get()))-1
        axes = self.figure.subplots(1, 2)
        for component, name in enumerate(('ON', 'OFF')):
            note = 'Physical origami frame' if self._vector_sites() else 'Schematic matrix (no spatial metadata)'
            im = draw_site_canvas(axes[component], source[selected, :, component], self._vector_sites(), self._canvas_rotation(), vmin=0, vmax=max(float(source[selected].max()), 25))
            axes[component].set(title=f'K{selected+1}: {name} stock added', xlabel='', ylabel='')
            self.figure.colorbar(im, ax=axes[component], label='Source transfer (nL)')
        self.summary.set(f"K{selected+1}: ON {source[selected,:,0].sum():g} nL + OFF {source[selected,:,1].sum():g} nL "
                         +f"+ buffer {recipe['vector_buffer_nl'][selected]:g} nL = {recipe['settings']['vector_volume_nl']:g} nL prepared. "
                         +'Each cell is the dose for that site, not its brightness. '+note+'.')

    def _build_controls(self, parent):
        bar = ttk.Frame(parent); bar.pack(fill='x', pady=4)
        ttk.Label(bar, text='Canvas construction step').pack(side='left')
        ttk.Button(bar, text='Previous', command=lambda: self._move_build(-1)).pack(side='left', padx=4)
        picker = ttk.Spinbox(bar, from_=1, to=1, textvariable=self.build_step, width=6, command=self.draw, state='disabled')
        picker.pack(side='left'); picker.bind('<Return>', lambda e: self.draw())
        picker.bind('<FocusOut>', lambda e: self.draw()); self._build_pickers.append(picker)
        ttk.Button(bar, text='Next', command=lambda: self._move_build(1)).pack(side='left', padx=4)
        ttk.Button(bar, text='Final canvas', command=lambda: self._move_build(self._build_count)).pack(side='left')
        ttk.Label(parent, textvariable=self.build_status, wraplength=1000).pack(anchor='w')

    def _move_build(self, delta):
        try:
            current = self.build_step.get()
        except (ValueError, tk.TclError):
            current = 1
        self.build_step.set(max(1, min(self._build_count, current+delta)))
        if self.plot.get() != 'Canvas construction':
            self.plot.set('Canvas construction')
        self.draw()

    def draw_canvas_build(self, result, data, metadata):
        import numpy as np
        from split_pool_canvas_build import canvas_build_steps
        from split_pool_vectors import draw_site_canvas, update_site_canvas
        try:
            row = max(0, min(len(data)-1, self.address.get()))
        except (ValueError, tk.TclError):
            row = 0
        key = (self._revision, id(result['prediction']), self.color.get(), row)
        if key != self._build_key:
            self._build_plan = canvas_build_steps(result, row); self._build_key = key; self.build_step.set(1)
        plan = self._build_plan
        if plan is None:
            self._build_count = 1; self.build_step.set(1)
            for picker in self._build_pickers:
                picker.configure(state='disabled')
            self.build_status.set('Select a fitted ON-only or ON/OFF library to see K-vector additions.')
            ax = self.figure.subplots(); ax.axis('off')
            ax.text(.5, .5, self.build_status.get(), ha='center', va='center')
            return
        self._build_count = len(plan['events'])
        try:
            step = max(1, min(self._build_count, self.build_step.get()))
        except (ValueError, tk.TclError):
            step = 1
        self.build_step.set(step)
        for picker in self._build_pickers:
            picker.configure(state='normal', to=self._build_count)
        event = plan['events'][step-1]
        self.build_status.set(f"Step {step}/{self._build_count} • {event['label']}")
        target_label = metadata['target_names'][row] if metadata and metadata.get('target_names') else f'Design {row}'
        title = f"{self.color.get() if self.image_designs else 'Dataset'} • {target_label} • code {plan['code']}\n{event['label']}"
        cached = self._build_artists.get(self.figure)
        reuse = cached is not None and cached[0] == key and cached[1][0, 0] in self.figure.axes
        self.figure.suptitle(title, fontsize=11)
        axes = cached[1] if reuse else self.figure.subplots(3, 3)
        sites = self._vector_sites()
        addition = f"K{event['vector']}" if event['vector'] is not None else 'No additional vector'
        if event['vector'] is not None and plan['physical']:
            addition += f" × {event['amount']:g} nL"
        for component, name in enumerate(('ON', 'OFF')):
            for col, (values, title) in enumerate(((event['before'][:, component], f'Mixture before · {name}'),
                    (event['added'][:, component], f'+ {addition} · {name}'),
                    (event['mixture'][:, component], f'= Combined mixture · {name}'))):
                if reuse:
                    im = update_site_canvas(axes[component, col], values)
                else:
                    im = draw_site_canvas(axes[component, col], values, sites, self._canvas_rotation(), vmin=0, vmax=plan['maximum'])
                axes[component, col].set(title=title, xticks=[], yticks=[]); axes[component, col].title.set_fontsize(9)
            if not reuse:
                self.figure.colorbar(im, ax=list(axes[component]), label=plan['unit'], shrink=.8)
        expected_title = 'ON after capture' if event['phase'] == 'capture' else 'ON if this mixture were captured now'
        for col, (values, title) in enumerate(((event['on_before'], 'ON before this round'),
                (event['on_after'], expected_title), (data[row], 'Target canvas'))):
            if reuse:
                im = update_site_canvas(axes[2, col], values)
            else:
                im = draw_site_canvas(axes[2, col], values, sites, self._canvas_rotation(), vmin=0, vmax=1)
            axes[2, col].set(title=title, xticks=[], yticks=[]); axes[2, col].title.set_fontsize(9)
        if not reuse:
            self.figure.colorbar(im, ax=list(axes[2]), label='ON occupancy', shrink=.8)
        self._build_artists[self.figure] = (key, axes)
        self.summary.set('Top rows show spatial addition: previous mixture + delivered K-vector contribution = combined mixture. '
            'K vectors are listed in numerical order for display; all additions within a well mix before one capture step. '
            'Partial-mixture ON previews are hypothetical, not extra capture steps. Earlier ON/OFF occupancy is retained across rounds. '
            f"Final selected-design RMSE {np.sqrt(np.mean((plan['final_on']-data[row])**2)):.4f}.")

    def _draw_dashboard_sections(self, result, data, metadata):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        from split_pool_plot_canvas import plot_canvas
        self.dashboard_frame.pack(fill='both', expand=True)
        name = 'Complete synthesis and pooling'
        if name not in self.dashboard_sections:
            frame = ttk.LabelFrame(self.dashboard_frame, text=name, padding=8)
            frame.pack(fill='x', pady=8)
            label = tk.StringVar(); ttk.Label(frame, textvariable=label, wraplength=1050).pack(fill='x')
            figure = Figure(figsize=(10, max(8, result.get('address_bits', 1)*3)), constrained_layout=True)
            canvas = plot_canvas(figure, master=frame)
            toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
            canvas.get_tk_widget().pack(fill='both', expand=True); toolbar.pack(fill='x')
            self.dashboard_sections[name] = (figure, canvas, toolbar, label)
        figure, canvas, toolbar, label = self.dashboard_sections[name]
        key = (self._revision, result['model'], id(result['prediction']), self.color.get(),
               self.address.get())
        if self._dashboard_rendered.get(name) == key:
            self.vector_picker.configure(state='normal', to=max(1, self._vector_count))
            return
        original_figure, original_summary = self.figure, self.summary.get()
        try:
            self.figure = figure; figure.clear(); self.summary.set('')
            self.draw_complete_synthesis(result)
            label.set(self.summary.get()); toolbar.update(); canvas.draw_idle()
            self._dashboard_rendered[name] = key
        finally:
            self.figure = original_figure; self.summary.set(original_summary)

    def draw_branch(self, result, data):
        import numpy as np
        from split_pool_branch import branch_trace
        try:
            row = max(0, min(len(data)-1, self.address.get()))
        except (ValueError, tk.TclError):
            row = 0
        trace = branch_trace(result, row)
        if trace is None:
            ax = self.figure.subplots(); ax.axis('off')
            ax.text(.5, .5, 'Select a fitted finite-K additive, ON-only, or ON/OFF model.\nPCA has no split–pool branch history.', ha='center', va='center')
            return
        rounds, count = trace['amounts'].shape
        grid = self.figure.add_gridspec(2, 2)
        ax = self.figure.add_subplot(grid[0, 0])
        amounts = trace['amounts']; signed = trace['unit'] == 'Signed coefficient'
        limit = max(float(np.abs(amounts).max()), 1e-12)
        im = ax.imshow(amounts, aspect='auto', cmap='coolwarm' if signed else 'viridis',
                       vmin=-limit if signed else 0, vmax=limit)
        ax.set(title='K vectors combined in each selected branch', xlabel='Vector (1-based)',
               yticks=range(rounds), yticklabels=[f'R{r+1} → {bit}' for r, bit in enumerate(trace['bits'])])
        ticks = np.unique(np.linspace(0, count-1, min(count, 12), dtype=int))
        ax.set_xticks(ticks, [str(k+1) for k in ticks])
        if amounts.size <= 80:
            for r in range(rounds):
                for k in range(count):
                    ax.text(k, r, f'{amounts[r,k]:.2g}', ha='center', va='center', fontsize=7,
                            color='black' if amounts[r,k] > .6*limit else 'white')
        self.figure.colorbar(im, ax=ax, label=trace['unit'])
        ax = self.figure.add_subplot(grid[0, 1])
        im = ax.imshow(trace['on'], aspect='auto', vmin=0, vmax=1)
        ax.set(title='Analog ON signal after each round', xlabel='Analog position',
               yticks=range(rounds+1), yticklabels=['Initial']+[f'After R{r+1}' for r in range(rounds)])
        self.figure.colorbar(im, ax=ax, label='ON probability')
        sites = None
        if self.image_designs:
            sites = [d for d in self.image_designs if d['color'] == self.color.get()][row]['image_sites']
        for col, values, title in ((0, data[row], 'Target analog canvas'), (1, trace['on'][-1], 'Final expected analog canvas')):
            ax = self.figure.add_subplot(grid[1, col])
            from split_pool_vectors import draw_site_canvas
            im = draw_site_canvas(ax, values, sites, self._canvas_rotation(), vmin=0, vmax=1)
            ax.set_title(title); self.figure.colorbar(im, ax=ax, label='ON probability')
        self.figure.suptitle(f"Digital code {trace['code']} • target design {row}" + (f' • {self.color.get()}' if sites else ''))
        error = float(np.sqrt(np.mean((trace['on'][-1]-data[row])**2)))
        self.summary.set(trace['note']+' Each row combines its K vectors in one branch; the code follows the selected 0/1 branches. '
                         f'Canvases show expected signal, before finite-count sampling. Selected design RMSE {error:.4f}.')

    def draw_well_trajectory(self, result, data, metadata):
        import numpy as np
        recipe = result.get('recipe')
        if recipe is None:
            ax = self.figure.subplots(); ax.axis('off')
            ax.text(.5, .5, 'Fit ON-only or ON/OFF with 25 nL recipes enabled\nto view liquid volumes along a design’s path.', ha='center', va='center')
            return
        try:
            row = max(0, min(len(data)-1, self.address.get()))
        except (ValueError, tk.TclError):
            row = 0
        self.address.set(row)
        assigned = int(np.flatnonzero(np.asarray(result['assignment']) == row)[0])
        address = int(result['address_indices'][assigned])
        code = format(address, f"0{result['address_bits']}b")
        independent = result.get('synthesis_mode') == 'independent_wells'
        branches = np.array([assigned]) if independent else 2*np.arange(len(code))+np.array([int(bit) for bit in code])
        amounts = np.asarray(recipe['branch_transfers_nl'])[branches]
        remaining = np.asarray(recipe['branch_remaining_nl'])[branches]
        active = np.flatnonzero(amounts.sum(axis=0) > 0)
        label = metadata['target_names'][row] if metadata and metadata.get('target_names') else f'Target row {row}'
        if self.image_designs:
            self.location_picker.current(row)
            label = f'{self.color.get()} • {self.location.get()} • {label}'
        self.figure.suptitle(f'{label}\nDigital code {code} • '+('independent destination well' if independent else 'branch-well trajectory'))
        if independent:
            code = '-'  # One simultaneous mixing/capture operation, not barcode-bit exposures.
        axes = self.figure.subplots(2, 1, gridspec_kw={'height_ratios': (1, 1.2)})
        from split_pool_synthesis import draw_addition_histogram
        x = np.arange(len(code))
        labels = ['Mix all K stocks\nCapture once'] if independent else [f'Round {r+1}\nbranch {bit}' for r, bit in enumerate(code)]
        draw_addition_histogram(axes[0], amounts, remaining, labels, recipe['settings']['branch_volume_nl'])
        if len(active):
            values = amounts[:, active].T
            im = axes[1].imshow(values, aspect='auto', interpolation='nearest', vmin=0)
            ticks = np.unique(np.linspace(0, len(active)-1, min(len(active), 20), dtype=int))
            axes[1].set(xticks=x, xticklabels=labels, yticks=ticks,
                        yticklabels=[f'K{active[i]+1}' for i in ticks], ylabel='Source vector well',
                        title='Exact K-vector transfers (unused vectors omitted; hover for nL)')
            if values.size <= 100:
                for i, r in np.ndindex(values.shape):
                    axes[1].text(r, i, f'{values[i,r]:g}', ha='center', va='center', fontsize=8,
                                 color='black' if values[i,r] > values.max()*.6 else 'white')
            def coordinates(xvalue, yvalue):
                r, i = int(np.floor(xvalue+.5)), int(np.floor(yvalue+.5))
                if 0 <= r < len(code) and 0 <= i < len(active):
                    destination = f'Design {row}' if independent else f'Round {r+1}, branch {code[r]}'
                    return f'K{active[i]+1} → {destination}: {values[i,r]:g} nL'
                return ''
            axes[1].format_coord = coordinates
            self.figure.colorbar(im, ax=axes[1], label='Volume added (nL)')
        else:
            axes[1].axis('off'); axes[1].text(.5, .5, 'No K-vector additions along this path.', ha='center')
        self.summary.set(('Each design has an independent destination well; K stocks are mixed before one capture step. ' if independent else
                         'Wells are shared by populations taking the same branch; these are logical branch wells, not plate coordinates. ')+
                         'Gray = final volume not supplied by K vectors (buffer / incoming fluid is not separately modeled). '
                         'Each round shows its own recipe, not cumulative retained liquid. All K transfers are multiples of 25 nL. '
                         f"Fit RMSE {result['mse']**.5:.4f}.")

    def draw_recipe(self, result):
        recipe = result.get('recipe')
        if recipe is None:
            ax = self.figure.subplots(); ax.axis('off')
            ax.text(.5, .5, 'Select a fitted ON-only or ON/OFF model with 25 nL recipes enabled.', ha='center', va='center')
            return
        import numpy as np
        source = recipe['source_transfers_nl']; branch = recipe['branch_transfers_nl']
        independent = result.get('synthesis_mode') == 'independent_wells'
        count, sites, _ = source.shape
        grid = self.figure.add_gridspec(2, 2)
        vector_ticks = np.unique(np.linspace(0, count-1, min(count, 16), dtype=int))
        site_ticks = np.unique(np.linspace(0, sites-1, min(sites, 10), dtype=int))
        maximum = max(float(source.max()), 25.)
        for component, name in enumerate(('ON', 'OFF')):
            ax = self.figure.add_subplot(grid[0, component])
            im = ax.imshow(source[..., component], aspect='auto', interpolation='nearest', vmin=0, vmax=maximum)
            ax.set(title=f'1. Prepare vector wells: {name} staples', xlabel='Analog position (0-based)', ylabel='Destination vector well')
            ax.set_xticks(site_ticks)
            ax.set_yticks(vector_ticks, [f'K{k+1}' for k in vector_ticks])
            self.figure.colorbar(im, ax=ax, label=f'{name} stock added (nL)')
        ax = self.figure.add_subplot(grid[1, :])
        im = ax.imshow(branch, aspect='auto', interpolation='nearest', vmin=0, vmax=max(float(branch.max()), 25.))
        ax.set(title='2. Combine prepared vectors in each '+('design well' if independent else 'branch well'), xlabel='Source vector well', ylabel='Destination well')
        ax.set_xticks(vector_ticks, [f'K{k+1}' for k in vector_ticks])
        row_ticks = np.unique(np.linspace(0, len(branch)-1, min(len(branch), 16), dtype=int)) if independent else np.arange(len(branch))
        ax.set_yticks(row_ticks, [f'Design {r}' if independent else f'Round {r//2+1} · branch {r%2}' for r in row_ticks])
        ax.tick_params(axis='y', labelsize=8)
        for boundary in ([] if independent else range(2, len(branch), 2)):
            ax.axhline(boundary-.5, color='white', linewidth=.5, alpha=.5)
        self.figure.colorbar(im, ax=ax, label='Prepared vector added (nL)')
        settings = recipe['settings']
        response = 'RGB ratio response (complete occupancy)' if result.get('response_model') == 'rgb_ratio' else f"Assumed C50 {settings['capture_c50_nm']:g} nM"
        comparison = ('Original RGB transfers retained' if result.get('initialization') == 'original_rgb_recipe' else f"Continuous relaxation RMSE {result['continuous_mse']**.5:.4f}")
        self.summary.set(('Each row is one design well; all K additions are mixed before capture. ' if independent else 'Branch 0/1 is the digital bit chosen in that round. ')+f"Each cell shows volume added; zero means no transfer. All nonzero transfers are multiples of 25 nL. ON {settings['on_stock_nm']:g} nM; OFF {settings['off_stock_nm']:g} nM. "
            f"Vector wells {settings['vector_volume_nl']:g} nL; destination wells {settings['branch_volume_nl']:g} nL. "
            f"{response}. {comparison}; realized recipe RMSE {result['mse']**.5:.4f}.")

    def new_example(self):
        try:
            self.example_seed.set(str(int(self.example_seed.get())+1))
        except ValueError:
            self.example_seed.set('42')
        self.plot.set('Image Monte Carlo'); self.draw()

    def draw_matrix_example(self, result, count, seed):
        import numpy as np
        from split_pool_encoding.evaluation import matrix_example
        try:
            expected, measured, errors = matrix_example(self.data, result, count, seed)
        except ValueError as exc:
            ax = self.figure.subplots(); ax.axis('off')
            ax.text(.5, .5, str(exc), ha='center', va='center'); return
        is_pca = result['model'] == 'pca'
        truth = getattr(self, 'synthetic_truth', None)
        exact_reference = truth is not None and 'k' in truth
        values_to_plot = [self.data, expected, measured]
        titles = ['Target dataset', 'PCA reference (clipped)' if is_pca else 'Blind fit reconstruction',
                  f'Finite count: N={count}, seed={seed}']
        reference_rmse = None
        if exact_reference:
            from split_pool_encoding.datasets import replay_exact_k
            reference = replay_exact_k(truth)
            reference_rmse = float(np.sqrt(np.mean((self.data-reference)**2)))
            values_to_plot.insert(1, reference)
            titles.insert(1, f"Known recipe · K={truth['k']}\nRMSE {reference_rmse:.2g}")
        axes = self.figure.subplots(1, len(values_to_plot), sharex=True, sharey=True)
        for ax, values, title in zip(axes, values_to_plot, titles):
            im = ax.imshow(values, aspect='auto', vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
            ax.set_title(title, fontsize=11); ax.set_xlabel('Analog position')
        axes[0].set_ylabel('Target row (original order)')
        self.figure.colorbar(im, ax=list(axes), label='ON occupancy')
        note = (('Realized 25 nL recipe; complete occupancy at each nonzero exposure.' if result.get('response_model') == 'rgb_ratio' else 'Realized 25 nL recipe; adjustable, uncalibrated C50.') if result.get('recipe') else
                'Hypothetical PCA reference, clipped to [0, 1].' if is_pca else
                'Ideal reference; no physical stock recipe.' if result['model'] == 'additive' else
                'Ideal continuous-mixture model.')
        if exact_reference:
            note += (f" Known generating recipe: K={truth['k']}, reference RMSE {reference_rmse:.2g}. "
                     "Exact expected occupancy is feasible with that ON/OFF recipe; the fitted result is a blind search.")
        self.summary.set(f"{result['model']} • Matrix Monte Carlo • RMSE: encoding {errors['encoding_rmse']:.4f}, "
            f"sampling {errors['sampling_rmse']:.4f}, total {errors['total_rmse']:.4f}\n"
            + (f"Raw PCA RMSE {errors['raw_encoding_rmse']:.4f}; clipped {errors['clipped_fraction']:.1%}. " if is_pca else '')
            + note + ' Each row is a target vector, not an RGB image location. Independent binomial sampling; perfect classification.')

    def draw_image(self, result):
        def message(text):
            ax = self.figure.subplots(); ax.axis('off')
            ax.text(.5, .5, text, ha='center', va='center', wrap=True)
        if self.image_designs and result.get('physical_mode') and result['model'] == 'additive':
            message('The signed additive reference has no ON/OFF stock recipe.\nSelect ON-only or ON/OFF for the 25 nL simulation, or disable recipe mode and refit for an ideal additive comparison.'); return
        try:
            count = int(self.sample_count.get()); seed = int(self.example_seed.get())
            if count < 1 or seed < 0:
                raise ValueError
        except ValueError:
            message('Use a positive integer origami count and a nonnegative example seed.'); return
        if not self.image_designs:
            self.draw_matrix_example(result, count, seed)
            return
        key = (self._revision, result['model'], count, seed)
        if key not in self.image_cache:
            message('Rendering finite-count image…')
            if not self._preview_busy:
                self._preview_busy = True
                designs = self.image_designs; fits = result['color_fits']; options = dict(self.render_options)
                def render():
                    try:
                        from split_pool_image import render_image_example
                        preview = render_image_example(designs, fits, count, seed, options)
                        self.events.put(('preview', (key, preview, None)))
                    except Exception as exc:
                        self.events.put(('preview', (key, None, str(exc))))
                Thread(target=render, daemon=True, name='split-pool-image').start()
                if self._poll_id is None:
                    self._poll_id = self.after(100, self.poll)
            return
        preview, error = self.image_cache[key]
        if error:
            message(error); return
        images, errors = preview
        is_pca = result['model'] == 'pca'
        titles = ('Current image targets', 'PCA reference (clipped)' if is_pca else 'Expected independent-well image' if result.get('synthesis_mode') == 'independent_wells' else 'Expected split–pool image',
                  f'Finite count: N={count}, seed={seed}')
        design = self._selected_design()
        if self.plot.get() == 'Dashboard' and design is not None:
            from split_pool_selection import draw_selection_comparison
            draw_selection_comparison(self.figure, images[0], design, result, self.address.get(), result_image=images[1])
            axes = []
        else:
            axes = self.figure.subplots(1, 3)
        for index, (ax, image, title) in enumerate(zip(axes, images, titles)):
            ax.imshow(image); ax.set_title(title, fontsize=11); ax.axis('off')
            if index == 0:
                from split_pool_selection import highlight_image
                design = self._selected_design()
                if design is not None:
                    highlight_image(ax, design, image)
                    ax.set_title(f"{title}\nSelected: {design['name']}", fontsize=10)
        if is_pca:
            ranks = sorted({fit['parameters'].shape[0] for fit in result['color_fits'].values()})
            self.summary.set(f"PCA reference • Rank {', '.join(map(str, ranks))} (independent of K) • All colors • "
                f"Raw PCA RMSE {errors['raw_encoding_rmse']:.4f} • Values clipped to [0, 1]: {errors['clipped_fraction']:.1%}\n"
                f"Clipped-reference RMSE {errors['encoding_rmse']:.4f} • Sampling RMSE {errors['sampling_rmse']:.4f} • Total RMSE {errors['total_rmse']:.4f}\n"
                'Hypothetical binomial readout of clipped PCA values, not a split–pool synthesis. Same Gaussian width and exposure; ideal alignment.')
            return
        recipe_note = (('25 nL at both stages; RGB ON/(ON+OFF) response assumes complete occupancy. ' if result.get('response_model') == 'rgb_ratio' else '25 nL at both stages; one well per vector; adjustable, uncalibrated C50 capture model. ')
                       if result.get('recipe') else 'Ideal continuous mixture; transfer constraints are disabled. ')
        self.summary.set(f"{result['model']} • K={result.get('basis_label', 'Unrestricted')} • All colors: unique assigned addresses • Site RMSE: encoding {errors['encoding_rmse']:.4f}, "
            f"sampling {errors['sampling_rmse']:.4f}, total {errors['total_rmse']:.4f}\n"
            + ' | '.join(f"{color}: {len(fit['assignment'])}/{fit.get('address_capacity', len(fit['assignment']))} addresses used" for color, fit in result['color_fits'].items()) + '\n' +
            recipe_note + 'Same Gaussian width and exposure. Independent binomial occupancy; ideal alignment and perfect site classification.')

    def close(self):
        self.destroy()

    def destroy(self):
        self.cancel.set()
        if self._poll_id is not None:
            self.after_cancel(self._poll_id); self._poll_id = None
        super().destroy()
