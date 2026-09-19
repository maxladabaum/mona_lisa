"""RGB image import and preview of independently synthesized color/shift designs."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from app_state import all_384_wells, create_run_output_directory, plate_wells, record_transfers, save_json, next_unused_wells
from image_encoding import (
    CHANNELS, FIT_MODES, ROWS, DEFAULT_SCHEME, ENCODING_SCHEMES, physical_raster_size,
    bin_image, compile_designs, geometry, prepare_runs, reconstruct, site_name, rotate_point, physical_reference_raster,
)
from gaussian_reconstruction import DEFAULT_FWHM_NM, DEFAULT_EXPOSURE, render_gaussian_overlay
from gaussian_optimization import fitted_target, optimize_designs
from alignment_reconstruction import AlignmentModel, AlignmentSettings, VALIDATION_SEED
from picklist_core import PICKLIST_COLUMNS, combine_picklists, parse_vector, valid_well_list, write_csv


class RGBImageView(ttk.Frame):
    def __init__(self, parent, app, app_dir, storage_path, plate_path):
        super().__init__(parent, padding=10)
        self.app, self.app_dir = app, app_dir
        self.storage_path, self.plate_path = storage_path, plate_path
        self.color_views = {view.title: view for view in app.set_views if view.title in CHANNELS}
        self.pixels = None
        self.comparison_window = None
        self.designs, self.runs = [], []
        self.saved = self.stored = self.committed = False
        self.image_path = tk.StringVar()
        self.fit_mode = tk.StringVar(value=FIT_MODES[0])
        self.rotation = tk.StringVar(value="0")
        self.gaussian_fwhm = tk.StringVar(value=str(app.config.get("image_gaussian_fwhm_nm", DEFAULT_FWHM_NM)))
        self.gaussian_exposure = tk.StringVar(value=str(app.config.get("image_gaussian_exposure", DEFAULT_EXPOSURE)))
        self.linear_light = tk.BooleanVar(value=app.config.get("image_linear_light", True))
        self.normalize_designs = tk.BooleanVar(value=app.config.get("image_normalize_designs", True))
        self.compensate_area = tk.BooleanVar(value=app.config.get("image_compensate_area", True))
        self.reconstruction = None
        self.reconstruction_summary = tk.StringVar()
        self.optimize_enabled = tk.BooleanVar(value=False)
        self.optimization_metrics = None
        self.optimization_summary = tk.StringVar()
        self.alignment_enabled = tk.BooleanVar(value=False)
        self.alignment_values = {key: tk.StringVar(value=str(value)) for key, value in vars(AlignmentSettings()).items()}
        self.alignment_models = {}
        self.example_seed = 42
        self.alignment_example = None
        self.ideal_reconstruction = None
        self.alignment_summary = tk.StringVar()
        saved_scheme = app.config.get("image_encoding_scheme", DEFAULT_SCHEME)
        if saved_scheme not in ENCODING_SCHEMES:
            saved_scheme = DEFAULT_SCHEME
        self.scheme = tk.StringVar(value=ENCODING_SCHEMES[saved_scheme].label)
        self.image_geometry = geometry(saved_scheme)
        self.geometry_summary = tk.StringVar()
        self.explanation = tk.StringVar()
        self.start_well = tk.StringVar(value="A01")
        self.design_name = tk.StringVar()
        self.status = tk.StringVar(value="Upload an image to preview its color/shift designs.")
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Upload RGB image…", command=self._upload).pack(side="left")
        ttk.Label(toolbar, textvariable=self.image_path, width=55).pack(side="left", padx=10)
        fits = ttk.Combobox(toolbar, textvariable=self.fit_mode, values=FIT_MODES, state="readonly", width=24)
        fits.pack(side="right")
        fits.bind("<<ComboboxSelected>>", lambda _event: self._load())
        scheme_bar = ttk.Frame(self)
        scheme_bar.pack(fill="x", pady=(6, 0))
        ttk.Label(scheme_bar, text="Encoding scheme").pack(side="left")
        self.scheme_picker = ttk.Combobox(scheme_bar, textvariable=self.scheme,
                                          values=[scheme.label for scheme in ENCODING_SCHEMES.values()],
                                          state="readonly", width=65)
        self.scheme_picker.pack(side="left", padx=8)
        ttk.Button(scheme_bar, text="Compare all schemes…", command=self._compare_schemes).pack(side="left", padx=4)
        rotation_bar = ttk.Frame(self)
        rotation_bar.pack(fill="x", pady=(6, 0))
        ttk.Label(rotation_bar, text="Origami rotation (degrees clockwise)").pack(side="left")
        rotation_picker = ttk.Combobox(rotation_bar, textvariable=self.rotation,
                                      values=("0", "90", "180", "270"), state="readonly", width=5)
        rotation_picker.pack(side="left", padx=8)
        rotation_picker.bind("<<ComboboxSelected>>", lambda _event: self._load())
        ttk.Button(rotation_bar, text="Origami left 90°", command=lambda: self._rotate(-90)).pack(side="left", padx=4)
        ttk.Button(rotation_bar, text="Origami right 90°", command=lambda: self._rotate(90)).pack(side="left", padx=4)
        ttk.Label(rotation_bar, text="The image stays upright; the origami grid rotates.").pack(side="left", padx=10)
        ttk.Label(self, textvariable=self.geometry_summary).pack(anchor="w", pady=6)
        settings = ttk.Frame(self)
        settings.pack(fill="x")
        ttk.Label(settings, text="Base volume (nL)").pack(side="left")
        ttk.Spinbox(settings, from_=25, to=100000, increment=25, textvariable=app.transfer_volume, width=7).pack(side="left", padx=6)
        for color, view in self.color_views.items():
            ttk.Label(settings, text=color + " excess").pack(side="left", padx=(8, 0))
            ttk.Entry(settings, textvariable=view.excess_multiplier, width=5).pack(side="left", padx=6)
        ttk.Button(settings, text="Run & Mixing Settings", command=self._open_settings).pack(side="right")
        reconstruction_bar = ttk.Frame(self)
        reconstruction_bar.pack(fill="x", pady=(6, 0))
        ttk.Label(reconstruction_bar, text="Gaussian FWHM (nm)").pack(side="left")
        ttk.Spinbox(reconstruction_bar, from_=0.2, to=100, increment=0.5,
                    textvariable=self.gaussian_fwhm, width=6).pack(side="left", padx=8)
        ttk.Label(reconstruction_bar, text="Exposure (shared across schemes)").pack(side="left")
        ttk.Spinbox(reconstruction_bar, from_=0.05, to=20, increment=0.1,
                    textvariable=self.gaussian_exposure, width=6).pack(side="left", padx=8)
        ttk.Label(reconstruction_bar, textvariable=self.reconstruction_summary).pack(side="left", padx=8)
        light_bar = ttk.Frame(self)
        light_bar.pack(fill="x", pady=(6,0))
        ttk.Checkbutton(light_bar, text="Linear-light image processing (sRGB input/output)",
                        variable=self.linear_light, command=self._light_changed).pack(side="left")
        ttk.Checkbutton(light_bar, text="Equal total signal across encoding schemes",
                        variable=self.normalize_designs, command=self._light_changed).pack(side="left", padx=12)
        ttk.Checkbutton(self, text="Match brightness for larger canvases (4× exposure compensation for quadrants)",
                        variable=self.compensate_area, command=self._light_changed).pack(anchor="w")
        alignment_bar = ttk.LabelFrame(self, text="Residual alignment error (after fiducial alignment, before decoded XY shift)", padding=5)
        alignment_bar.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(alignment_bar, text="Simulate alignment uncertainty", variable=self.alignment_enabled,
                        command=self._volumes_changed).grid(row=0, column=0, columnspan=2, sticky="w")
        fields = (("translation_x_nm", "X error SD (nm)"), ("translation_y_nm", "Y error SD (nm)"),
                  ("rotation_deg", "Angle SD (°)"), ("origami_per_design", "Origami / design"),
                  ("center_x_nm", "Fiducial center X (nm)"), ("center_y_nm", "Fiducial center Y (nm)"),
                  ("variability_penalty", "Extra variability penalty"))
        for index, (key, label) in enumerate(fields):
            row, column = 1 + index // 4, 2 * (index % 4)
            ttk.Label(alignment_bar, text=label).grid(row=row, column=column, sticky="w", padx=4)
            ttk.Entry(alignment_bar, textvariable=self.alignment_values[key], width=8).grid(row=row, column=column+1, padx=4)
        ttk.Label(alignment_bar, text="Illustrative errors; center is relative to C6 H01 before shifts. Default center: full staple grid.").grid(row=3, column=0, columnspan=8, sticky="w")
        optimization_bar = ttk.Frame(self)
        optimization_bar.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(optimization_bar, text="Optimize replacement percentages for Gaussian overlap",
                        variable=self.optimize_enabled, command=self._volumes_changed).pack(side="left")
        ttk.Label(optimization_bar, textvariable=self.optimization_summary).pack(side="left", padx=8)
        previews = ttk.Frame(self)
        previews.pack(fill="x", pady=8)
        self.pixel_canvases = []
        for label in ("Binned RGB", "Predicted RGB — Gaussian overlay"):
            frame = ttk.LabelFrame(previews, text=label)
            frame.pack(side="left", padx=(0, 12))
            canvas = tk.Canvas(frame, width=300, height=214, background="#15191d", highlightthickness=0)
            canvas.pack()
            self.pixel_canvases.append(canvas)
            self.prediction_frame = frame
        self.baseline_frame = ttk.LabelFrame(previews, text="Before optimization — Gaussian overlay")
        self.baseline_canvas = tk.Canvas(self.baseline_frame, width=300, height=214, background="#15191d", highlightthickness=0)
        self.baseline_canvas.pack()
        explanation = ttk.Label(previews, justify="left", wraplength=310, textvariable=self.explanation)
        explanation.pack(side="left", padx=6, anchor="n")
        self.alignment_previews = ttk.LabelFrame(self, text="Alignment comparison — active replacement assignments", padding=5)
        self.alignment_canvases = []
        for label in ("Ideal alignment", "Expected alignment result (sampled mean)", "Finite-count example"):
            frame = ttk.LabelFrame(self.alignment_previews, text=label)
            frame.pack(side="left", padx=5)
            canvas = tk.Canvas(frame, width=300, height=214, background="#15191d", highlightthickness=0)
            canvas.pack()
            self.alignment_canvases.append(canvas)
        self.alignment_details = ttk.Frame(self)
        ttk.Label(self.alignment_details, textvariable=self.alignment_summary, wraplength=900).pack(side="left")
        ttk.Button(self.alignment_details, text="New alignment example", command=self._new_alignment_example).pack(side="right")
        chooser = ttk.Frame(self)
        self.alignment_pack_before = chooser
        chooser.pack(fill="x")
        ttk.Label(chooser, text="Inspect design").pack(side="left")
        self.design_picker = ttk.Combobox(chooser, textvariable=self.design_name, state="readonly", width=22)
        self.design_picker.pack(side="left", padx=8)
        self.design_picker.bind("<<ComboboxSelected>>", lambda _event: self._draw_design())
        self.design_summary = tk.StringVar()
        ttk.Label(chooser, textvariable=self.design_summary).pack(side="left")
        detail = ttk.Frame(self)
        detail.pack(fill="both", expand=True, pady=6)
        self.design_canvas = tk.Canvas(detail, width=590, height=235, background="#15191d", highlightthickness=0)
        self.design_canvas.pack(side="left", fill="y")
        columns = ("Site", "Image pixel x,y", "RGB", "Replacement %", "Repl. nL", "Original nL")
        self.site_tree = ttk.Treeview(detail, columns=columns, show="headings", height=8)
        for column in columns:
            self.site_tree.heading(column, text=column)
            self.site_tree.column(column, width=85, stretch=True)
        scroll = ttk.Scrollbar(detail, orient="vertical", command=self.site_tree.yview)
        self.site_tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.site_tree.pack(side="left", fill="both", expand=True)
        actions = ttk.Frame(self)
        actions.pack(fill="x", pady=6)
        ttk.Label(actions, text="Start at well").pack(side="left")
        ttk.Entry(actions, textvariable=self.start_well, width=6).pack(side="left", padx=6)
        self.prepare_button = ttk.Button(actions, command=self._prepare)
        self.prepare_button.pack(side="left", padx=6)
        self.export_button = ttk.Button(actions, text="Save image batch…", command=self._export, state="disabled")
        self.export_button.pack(side="left", padx=6)
        self.store_button = ttk.Button(actions, command=self._store, state="disabled")
        self.store_button.pack(side="left", padx=6)
        ttk.Label(self, textvariable=self.status, wraplength=1150).pack(anchor="w", pady=4)
        self._update_scheme_controls()
        self.scheme.trace_add("write", self._scheme_changed)
        self.gaussian_fwhm.trace_add("write", self._render_settings_changed)
        self.gaussian_exposure.trace_add("write", self._render_settings_changed)
        for variable in self.alignment_values.values():
            variable.trace_add("write", self._alignment_changed)
        for variable in (app.transfer_volume, *(view.excess_multiplier for view in self.color_views.values())):
            variable.trace_add("write", self._volumes_changed)
        for variable in (self.start_well, app.base_path, app.destination_plate, app.source_plate_type, app.max_volume,
                         app.transfers_per_source, app.hinge_type, app.staple, app.scaffold, app.magnesium,
                         app.sodium, app.te_10x, app.desired, *(view.path_var for view in self.color_views.values())):
            variable.trace_add("write", self._invalidate)

    def _render_settings_changed(self, *_args):
        self._volumes_changed()
        if self.reconstruction is not None:
            self.app.config["image_gaussian_fwhm_nm"] = float(self.gaussian_fwhm.get())
            self.app.config["image_gaussian_exposure"] = float(self.gaussian_exposure.get())
            self.app._schedule_config_save()

    def _light_changed(self):
        self.alignment_models.clear()
        self.app.config["image_compensate_area"] = self.compensate_area.get()
        self.app.config["image_linear_light"] = self.linear_light.get()
        self.app.config["image_normalize_designs"] = self.normalize_designs.get()
        self.app._schedule_config_save()
        self._load()

    def _light_options(self):
        return {"linear_light": self.linear_light.get(), "normalize_designs": self.normalize_designs.get(), "compensate_area": self.compensate_area.get()}

    def _alignment_settings(self):
        values = {key: float(var.get()) for key, var in self.alignment_values.items()}
        settings = AlignmentSettings(**values)
        settings.validate()
        values["origami_per_design"] = int(values["origami_per_design"])
        return AlignmentSettings(**values)

    def _alignment_changed(self, *_args):
        self.alignment_models.clear()
        if self.alignment_enabled.get():
            self._volumes_changed()

    def _alignment_model(self, designs):
        settings = self._alignment_settings()
        key = (designs[0]["encoding_scheme"], designs[0]["origami_rotation_degrees_clockwise"],
               settings, float(self.gaussian_fwhm.get()), float(self.gaussian_exposure.get()),
               self.linear_light.get(), self.normalize_designs.get(), self.compensate_area.get())
        if key not in self.alignment_models:
            if len(self.alignment_models) >= 2:
                self.alignment_models.clear()
            self.alignment_models[key] = AlignmentModel(designs, settings, key[3], key[4], **self._light_options())
        return self.alignment_models[key]

    def _new_alignment_example(self):
        if not self.alignment_enabled.get() or not self.designs or self.reconstruction is None:
            return
        self.example_seed += 1
        self.alignment_example = self._alignment_model(self.designs).example(self.designs, self.example_seed)
        self._draw_overlay(self.alignment_canvases[2], self.alignment_example)
        # The exported realization and its seed must match a newly prepared snapshot.
        self._invalidate()

    def _render_overlay(self, designs):
        if self.alignment_enabled.get():
            return self._alignment_model(designs).expected(designs)
        return render_gaussian_overlay(designs, float(self.gaussian_fwhm.get()), float(self.gaussian_exposure.get()), **self._light_options())

    def _compile_scheme(self, pixels, scheme_id):
        """Use identical concentration settings for the main view and comparison."""
        return compile_designs(
            pixels, int(self.app.transfer_volume.get()),
            {color: float(view.excess_multiplier.get()) for color, view in self.color_views.items()},
            scheme_id, int(self.rotation.get()), linear_light=self.linear_light.get())

    def _optimize(self, designs):
        overlay = self._render_overlay(designs)
        target = fitted_target(self.image_path.get(), self.fit_mode.get(), overlay.rgb.size, linear_light=self.linear_light.get())
        if self.alignment_enabled.get():
            model = self._alignment_model(designs)
            result, metrics = model.optimize(designs, target)
            validation = AlignmentModel(designs, model.settings, model.fwhm, model.exposure, seed=VALIDATION_SEED, **self._light_options())
            metrics["validation_seed"] = VALIDATION_SEED
            metrics["validation_baseline"] = validation.evaluate(designs, target)
            metrics["validation_optimized"] = validation.evaluate(result, target)
            metrics["validation_used_for_selection"] = False
            return result, metrics
        return optimize_designs(designs, target, float(self.gaussian_fwhm.get()), float(self.gaussian_exposure.get()), **self._light_options())

    def _draw_overlay(self, canvas, result, info=None, draw_width=288):
        from PIL import Image, ImageTk
        info = self.image_geometry if info is None else info
        scale = draw_width / info["field_width_nm"]
        width = round(scale * info["display_field_width_nm"])
        height = round(scale * info["display_field_height_nm"])
        canvas.delete("all")
        canvas.configure(height=height + 14)
        canvas._overlay_photo = ImageTk.PhotoImage(result.rgb.resize((width, height), Image.Resampling.LANCZOS), master=canvas)
        canvas.create_image(6 + (draw_width - width) / 2, 7, anchor="nw", image=canvas._overlay_photo)

    def _compare_schemes(self):
        if self.pixels is None:
            messagebox.showinfo("Image required", "Upload an image to compare the encoding schemes.", parent=self)
            return
        if self.comparison_window is None or not self.comparison_window.winfo_exists():
            self.comparison_window = EncodingComparison(self)
        else:
            self.comparison_window.lift()
        self.comparison_window.refresh()

    def _refresh_comparison(self):
        if self.comparison_window is not None and self.comparison_window.winfo_exists():
            self.comparison_window.refresh()

    def _scheme_id(self):
        return next(key for key, scheme in ENCODING_SCHEMES.items() if scheme.label == self.scheme.get())

    def _update_scheme_controls(self):
        self.image_geometry = geometry(self._scheme_id(), int(self.rotation.get()))
        info = self.image_geometry
        self.geometry_summary.set(
            "Upright image: {display_columns} × {display_rows} pixels · {display_pixel_pitch_x_nm:.4g} × {display_pixel_pitch_y_nm:.4g} nm/pixel · "
            "{display_field_width_nm:g} × {display_field_height_nm:g} nm field · Origami {origami_rotation_degrees_clockwise}° clockwise".format(**info))
        if info.get("nonuniform_sampling"):
            self.geometry_summary.set(self.geometry_summary.get() + " · nonuniform spacing (listed by quadrant below)")
        self.prepare_button.configure(text="Prepare {} runs".format(info["design_count"]))
        self.store_button.configure(text="Send {} runs to storage".format(info["design_count"]))
        scheme = ENCODING_SCHEMES[self._scheme_id()]
        meaning = ("All {} bits jointly identify color and XY shift.".format(scheme.bit_count) if scheme.joint
                   else "2 color bits + {} X bits + {} Y bits{}.".format(
                       (scheme.x_states-1).bit_length(), (scheme.y_states-1).bit_length(),
                       " + 1 even-parity bit" if scheme.parity else ""))
        if scheme.tiled and scheme.joint:
            allocations = "; ".join("{}: {}×{} shifts ({:.3g}×{:.3g} nm)".format(name, q["x_states"],q["y_states"],q["pixel_pitch_x_nm"],q["pixel_pitch_y_nm"])
                for name,q in zip(info["quadrant_order"],info["quadrant_sampling"]))
            meaning = "Joint six-bit color/quadrant/shift codebook. " + allocations + ". Equal physical quadrant areas; sampling density varies."
        elif scheme.tiled:
            meaning = "2 color bits + 2 quadrant bits + 1 local X + 1 local Y. Four 12×16 quadrants; the physical origami is unchanged. The brightness-matching option compensates for the 4× canvas area; it does not increase photon counts."
        meaning += " {} sites per bit.".format(len(scheme.bit_sites[0]))
        if scheme.parity:
            meaning += " Parity detects single-bit errors; it does not correct them."
        self.explanation.set(
            ("{} designs: 3 colors distributed across 4 quadrants.\n\n".format(scheme.design_count) if scheme.tiled else "{design_count} designs: 3 colors × {x_states} X-shifts × {y_states} Y-shifts.\n\n".format(**info))
            + meaning + " Barcode bits stay at 100% or 0%.\n\n"
            "Overlay: shifted Gaussian signals weighted by each site’s replacement percentage. Assumes equal design abundance and channel brightness; barcode sites are excluded.")

    def _scheme_changed(self, *_args):
        self._update_scheme_controls()
        self.app.config["image_encoding_scheme"] = self._scheme_id()
        self.app._schedule_config_save()
        self._invalidate()
        if self.image_path.get():
            self._load()
        else:
            self.status.set("Upload an image to preview {} color/shift designs.".format(self.image_geometry["design_count"]))

    def _invalidate(self, *_args):
        self.runs = []
        self.saved = self.stored = self.committed = False
        self.export_button.configure(state="disabled")
        self.store_button.configure(state="disabled")
        if self.pixels is not None:
            self.status.set("Preview ready. Prepare {} runs to allocate separate unused wells using the current settings.".format(self.image_geometry["design_count"]))

    def _open_settings(self):
        self.app.main_tabs.tab(self.app.settings_page, state="normal")
        self.app.main_tabs.select(self.app.settings_page)

    def _upload(self):
        path = filedialog.askopenfilename(parent=self, title="Choose RGB image", filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp"), ("All files", "*")])
        if path:
            self.image_path.set(path)
            self.rotation.set("0")
            self._load()

    def _rotate(self, degrees):
        self.rotation.set(str((int(self.rotation.get()) + degrees) % 360))
        self._load()

    def _load(self):
        if not self.image_path.get():
            return
        self._update_scheme_controls()
        self.pixels = None
        self.designs = []
        self.reconstruction = None
        self.alignment_previews.pack_forget()
        self.alignment_details.pack_forget()
        self.alignment_example = self.ideal_reconstruction = None
        self._invalidate()
        try:
            self.pixels = bin_image(self.image_path.get(), self.fit_mode.get(), int(self.rotation.get()), self._scheme_id(), linear_light=self.linear_light.get())
            self._volumes_changed()
        except Exception as exc:
            self.status.set(str(exc))
            for canvas in (*self.pixel_canvases, self.design_canvas):
                canvas.delete("all")
            self.site_tree.delete(*self.site_tree.get_children())
            self._refresh_comparison()
            messagebox.showerror("Cannot import image", str(exc), parent=self)

    def _volumes_changed(self, *_args):
        self._invalidate()
        self.optimization_metrics = None
        self.optimization_summary.set("")
        self.baseline_frame.pack_forget()
        self.alignment_previews.pack_forget()
        self.alignment_details.pack_forget()
        self.alignment_example = self.ideal_reconstruction = None
        self.reconstruction = None
        if self.pixels is None:
            return
        self._draw_pixels(self.pixel_canvases[0], self.pixels)
        try:
            self.designs = self._compile_scheme(self.pixels, self._scheme_id())
            if self.optimize_enabled.get():
                self.optimization_summary.set("Optimizing…")
                self.update_idletasks()
                baseline = self._render_overlay(self.designs)
                self.designs, self.optimization_metrics = self._optimize(self.designs)
                self.baseline_frame.pack(side="left", padx=(0, 12), before=self.prediction_frame)
                self._draw_overlay(self.baseline_canvas, baseline)
                metrics = self.optimization_metrics
                self.optimization_summary.set("RGB RMSE {:.4f} → {:.4f} · squared error reduced {:.1%}".format(
                    metrics["baseline_rmse"], metrics["optimized_rmse"], metrics["mse_reduction_fraction"]))
                if "validation_optimized" in metrics:
                    self.optimization_summary.set(self.optimization_summary.get() + " · held-out RMSE {:.4f} → {:.4f}".format(
                        metrics["validation_baseline"]["expected_mse"]**.5, metrics["validation_optimized"]["expected_mse"]**.5))
            self.design_picker.configure(values=[design["name"] for design in self.designs])
            if self.design_name.get() not in {design["name"] for design in self.designs}:
                self.design_name.set(self.designs[0]["name"])
            self.reconstruction = self._render_overlay(self.designs)
            self._draw_overlay(self.pixel_canvases[1], self.reconstruction)
            self.reconstruction_summary.set("{:.1%} of display pixels clipped".format(self.reconstruction.metadata["clipped_pixel_fraction"]))
            if self.alignment_enabled.get():
                self.prediction_frame.configure(text="Predicted RGB — expected alignment result")
                model = self._alignment_model(self.designs)
                self.ideal_reconstruction = render_gaussian_overlay(self.designs, model.fwhm, model.exposure, **self._light_options())
                self.alignment_example = model.example(self.designs, self.example_seed)
                self.alignment_previews.pack(fill="x", pady=5, before=self.alignment_pack_before)
                self.alignment_details.pack(fill="x", before=self.alignment_pack_before)
                for canvas, result in zip(self.alignment_canvases, (self.ideal_reconstruction, self.reconstruction, self.alignment_example)):
                    self._draw_overlay(canvas, result)
                self.alignment_summary.set("{} independent origami/design · zero-mean Gaussian rigid errors · 32 sampled transforms for expectation. "
                    "Fractional signals; example simulates alignment only, not occupancy or photon noise. Error axes rotate with the origami.".format(model.settings.origami_per_design))
            else:
                self.prediction_frame.configure(text="Predicted RGB — Gaussian overlay")
            area_gain = self.reconstruction.metadata.get("canvas_area_gain", 1)
            self.reconstruction_summary.set(self.reconstruction_summary.get() + " · canvas exposure compensation {:g}×".format(area_gain))
            self._draw_design()
        except ValueError as exc:
            self.designs = []
            self.reconstruction = None
            self.baseline_frame.pack_forget()
            self.alignment_previews.pack_forget()
            self.alignment_details.pack_forget()
            self.status.set("Check volume and reconstruction settings: {}".format(exc))
            self.reconstruction_summary.set(str(exc))
            self.pixel_canvases[1].delete("all")
            self.design_canvas.delete("all")
            self.site_tree.delete(*self.site_tree.get_children())
        self._refresh_comparison()

    def _pixel_bounds(self, x, y, info=None, draw_width=288):
        info = self.image_geometry if info is None else info
        x1,x2 = info["pixel_edges_x_nm"][x:x+2]
        y1,y2 = info["pixel_edges_y_nm"][y:y+2]
        angle = info["origami_rotation_degrees_clockwise"]
        a = rotate_point(x1,y1,info["field_width_nm"],info["field_height_nm"],angle)
        b = rotate_point(x2,y2,info["field_width_nm"],info["field_height_nm"],angle)
        scale = draw_width / info["field_width_nm"]
        left = 6 + (draw_width - scale * info["display_field_width_nm"]) / 2
        return (left+min(a[0],b[0])*scale,7+min(a[1],b[1])*scale,
                left+max(a[0],b[0])*scale,7+max(a[1],b[1])*scale)


    def _draw_pixels(self, canvas, pixels, info=None, draw_width=288):
        info = self.image_geometry if info is None else info
        canvas.delete("all")
        canvas.configure(height=round(draw_width * info["display_field_height_nm"] / info["field_width_nm"] + 14))
        for y, row in enumerate(pixels):
            for x, rgb in enumerate(row):
                canvas.create_rectangle(*self._pixel_bounds(x, y, info, draw_width),
                                        fill="#{:02x}{:02x}{:02x}".format(*rgb), outline="")

    def _draw_design(self):
        if not self.designs:
            return
        design = next(item for item in self.designs if item["name"] == self.design_name.get())
        binned_canvas = self.pixel_canvases[0]
        binned_canvas.delete("design_pixels")
        for site in design["image_sites"]:
            bounds = self._pixel_bounds(site["pixel_x"], site["pixel_y"])
            # A black halo keeps the white outline visible against any RGB value.
            binned_canvas.create_rectangle(*bounds, fill="", outline="black", width=3, tags="design_pixels")
            binned_canvas.create_rectangle(*bounds, fill="", outline="white", width=1, tags="design_pixels")
        self.design_summary.set("Bits {} · composite placement +{:g} nm X, +{:g} nm Y".format(
            "".join(map(str, design["bits"])), design["x_shift_nm"], design["y_shift_nm"]))
        if "quadrant" in design:
            self.design_summary.set(self.design_summary.get() + " · origami-frame {} · local shift ({}, {})".format(
                ("top-left", "top-right", "bottom-left", "bottom-right")[design["quadrant"]],
                design["local_x_shift"], design["local_y_shift"]))
        canvas = self.design_canvas
        canvas.delete("all")
        color = [0, 0, 0]
        color[design["channel"]] = 255
        rotation = int(self.rotation.get())
        portrait = rotation in (90, 270)
        canvas.configure(width=260 if portrait else 590, height=570 if portrait else 235)
        def position(x, y):
            px, py = rotate_point(x, y, 118.9, 35.7, rotation)
            return 55 + px * 4, 42 + py * 4
        for c in range(12):
            x_nm = c * 9.9 + (10 if c >= 6 else 0)
            label_x, label_y = position(x_nm, -6.5)
            canvas.create_text(label_x, label_y, text="{:02d}".format(12-c), fill="white")
            for r, row in enumerate(ROWS):
                x, y = position(x_nm, r * 5.1)
                if c == 0:
                    canvas.create_text(*position(-8, r * 5.1), text=row, fill="white")
                fraction = design["site_fractions"].get(site_name(row, "{:02d}".format(12-c)), 0)
                shade = "#{:02x}{:02x}{:02x}".format(*(round(bg*(1-fraction)+fg*fraction) for bg, fg in zip((21,25,29),color)))
                canvas.create_oval(x-8, y-8, x+8, y+8, fill=shade, outline="#748087")
        self.site_tree.delete(*self.site_tree.get_children())
        for site in design["image_sites"]:
            self.site_tree.insert("", "end", values=(
                "{} C{}".format(site["row"], site["column"]), "{},{}".format(site["display_pixel_x"], site["display_pixel_y"]),
                ",".join(map(str, site["rgb"])), "{:.4g}".format(site["replacement_fraction"]*100),
                site["replacement_nl"], site["original_nl"]))

    def _used_wells(self, plate, ignore_ids=()):
        used = set(plate_wells(deepcopy(self.app.plate_state), plate))
        for item in self.app._storage_items():
            if item.get("destination_plate_name") == plate and item.get("id") not in ignore_ids:
                used.update(item.get("destination_wells", []))
        return used

    def _prepare(self):
        self._invalidate()
        if not self.designs:
            messagebox.showinfo("Image required", "Upload an image and enter valid transfer volumes first.", parent=self)
            return
        try:
            app = self.app
            wells = all_384_wells()
            start = valid_well_list(self.start_well.get())
            if len(start) != 1:
                raise ValueError("Enter one starting well.")
            plate = app.destination_plate.get().strip() or "Destination[1]"
            used = self._used_wells(plate)
            available = [well for well in wells[wells.index(start[0]):] if well not in used]
            settings = {"destination_plate_name": plate, "source_plate_type": app.source_plate_type.get().strip(),
                        "max_destination_volume_ul": float(app.max_volume.get()),
                        "transfers_per_source": int(app.transfers_per_source.get()), "hinge_type": app._selected_hinge_type()}
            mixing = {key: parse_vector(getattr(app, attr).get(), length, label) for key, attr, length, label in (
                ("staple", "staple", 4, "Staple"), ("scaffold", "scaffold", 4, "Scaffold"),
                ("magnesium", "magnesium", 4, "Mg"), ("sodium", "sodium", 4, "Na"),
                ("te_10x", "te_10x", 4, "10X TE"), ("desired", "desired", 6, "Desired"))}
            runs = prepare_runs(self.designs, app.base_path.get(),
                                {color: (view.path_var.get(), view.plate_name) for color, view in self.color_views.items()},
                                available, settings, mixing)
            self.batch_id = uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            for run in runs:
                run.update(id=uuid4().hex, created_utc=now, source_run_id=uuid4().hex,
                           image_batch_id=self.batch_id, source_picklist_path="", source_recipe_path="")
            self.runs = runs
            self.snapshot = {"format": "origami-rgb-image-v2", "batch_id": self.batch_id,
                             "color_processing": self._light_options(),
                             "source_image": self.image_path.get(), "fit_mode": self.fit_mode.get(),
                             "origami_rotation_degrees_clockwise": int(self.rotation.get()),
                             "source_image_rotation_degrees": 0,
                             "png_coordinate_frame": "upright image; JSON RGB arrays remain origami-local",
                             "encoding_scheme": self._scheme_id(),
                             "geometry": deepcopy(self.image_geometry), "binned_rgb": deepcopy(self.pixels),
                             "quantized_rgb": reconstruct(self.designs),
                             "quantized_rgb_encoding": "linear intensity × 255" if self.linear_light.get() else "legacy direct RGB",
                             "quantized_rgb_role": "per-site intensity reference; predicted appearance is gaussian_reconstruction",
                             "gaussian_reconstruction": deepcopy(self.reconstruction.metadata),
                             "optimization": deepcopy(self.optimization_metrics),
                             "alignment_simulation": deepcopy(self.reconstruction.metadata) if self.alignment_enabled.get() else None,
                             "alignment_example_seed": self.example_seed if self.alignment_enabled.get() else None,
                             "settings": settings, "mixing": mixing,
                             "designs": deepcopy(self.designs)}
            count = sum(len(run["destination_wells"]) for run in runs)
            self.status.set("Prepared {} independent runs using {} unused wells on {} ({}–{}). Save the batch or send it to storage.".format(
                len(runs), count, plate, runs[0]["destination_wells"][0], runs[-1]["destination_wells"][-1]))
            self.export_button.configure(state="normal")
            self.store_button.configure(state="normal")
        except Exception as exc:
            self.status.set("Preparation failed: {}".format(exc))
            messagebox.showerror("Cannot prepare image runs", str(exc), parent=self)

    def _check_destinations(self):
        if self.committed:
            return
        for run in self.runs:
            overlap = set(run["destination_wells"]) & self._used_wells(run["destination_plate_name"], {item["id"] for item in self.runs})
            if overlap:
                raise ValueError("Destination wells were used since this preview. Prepare the batch again.")

    def _commit(self):
        if self.committed:
            return
        state = deepcopy(self.app.plate_state)
        record_transfers(state, combine_picklists([run["picklist"] for run in self.runs]))
        save_json(self.plate_path, state)
        self.app.plate_state = state
        self.committed = True
        plate = self.runs[0]["destination_plate_name"]
        self.app.destination_wells.set(",".join(next_unused_wells(state, plate, 3)))
        self.app._refresh_plate_view()

    def _export(self):
        if not self.runs or self.saved:
            return
        folder = filedialog.askdirectory(parent=self, title="Choose parent folder for image batch",
                                         initialdir=self.app.output_root.get())
        if not folder:
            return
        try:
            from PIL import Image
            import numpy as np
            self._check_destinations()
            output = create_run_output_directory(Path(folder))
            write_csv(output / "picklist_combined.csv", combine_picklists([run["picklist"] for run in self.runs]), PICKLIST_COLUMNS)
            for run in self.runs:
                run_dir = output / run["run_name"]
                write_csv(run_dir / "picklist.csv", run["picklist"], PICKLIST_COLUMNS)
                write_csv(run_dir / "mixing_recipe.csv", run["mixing_recipe"], ("Reagent", "Volume_uL"))
            manifest = deepcopy(self.snapshot)
            manifest["runs"] = [{key: run[key] for key in ("run_name", "destination_plate_name", "destination_wells", "transfer_count")} for run in self.runs]
            save_json(output / "image_manifest.json", manifest)
            pixel_rows = []
            for design in self.snapshot["designs"]:
                for site in design["image_sites"]:
                    pixel_rows.append({"design": design["name"], "color": design["color"],
                                       "encoding_scheme": design["encoding_scheme"], "barcode_code": design["barcode_code"],
                                       "barcode_bits": "".join(map(str, design["bits"])),
                                       "x_shift": design["x_shift"], "y_shift": design["y_shift"],
                                       **{key: design[key] for key in ("quadrant", "quadrant_x", "quadrant_y", "local_x_shift", "local_y_shift", "local_x_states", "local_y_states", "quadrant_normalization_weight", "quadrant_offset_x_nm", "quadrant_offset_y_nm") if key in design},
                                       **{key: value for key, value in site.items() if key != "rgb"},
                                       "R": site["rgb"][0], "G": site["rgb"][1], "B": site["rgb"][2]})
            write_csv(output / "pixel_assignments.csv", pixel_rows, list(pixel_rows[0]))
            for key in ("binned_rgb", "quantized_rgb"):
                raster = Image.new("RGB", (self.snapshot["geometry"]["columns"], self.snapshot["geometry"]["rows"]))
                raster.putdata([tuple(pixel) for row in self.snapshot[key] for pixel in row])
                if key == "quantized_rgb" and self.snapshot.get("color_processing", {}).get("linear_light"):
                    from color_light import display_rgb
                    raster = display_rgb(np.asarray(raster, dtype=float)/255, True)
                rotation = self.snapshot["origami_rotation_degrees_clockwise"]
                physical = physical_reference_raster(raster,self.snapshot["encoding_scheme"],rotation)
                raster = raster.rotate(-rotation, expand=True)
                raster.save(output / (key + ".png"))
                physical.save(output / (key + "_physical.png"))
            overlay_settings = self.snapshot["gaussian_reconstruction"]
            light_options = {key: overlay_settings.get(key, False) for key in ("linear_light", "normalize_designs", "compensate_area")}
            if self.snapshot.get("alignment_simulation"):
                alignment = self.snapshot["alignment_simulation"]
                model = AlignmentModel(self.snapshot["designs"], AlignmentSettings(**alignment["alignment"]),
                                       overlay_settings["fwhm_nm"], overlay_settings["exposure"], seed=alignment["model_seed"], **light_options)
                overlay = model.expected(self.snapshot["designs"])
                example = model.example(self.snapshot["designs"], self.snapshot["alignment_example_seed"])
                example.rgb.save(output / "alignment_example.png")
                np.savez_compressed(output / "alignment_example_density.npz", density=example.density,
                                    pixel_x_nm=example.metadata["raster_pixel_x_nm"], pixel_y_nm=example.metadata["raster_pixel_y_nm"])
                render_gaussian_overlay(self.snapshot["designs"], model.fwhm, model.exposure, **light_options).rgb.save(output / "ideal_alignment.png")
            else:
                overlay = render_gaussian_overlay(self.snapshot["designs"], overlay_settings["fwhm_nm"], overlay_settings["exposure"], **light_options)
            overlay.rgb.save(output / "gaussian_reconstruction.png")
            np.savez_compressed(output / "gaussian_reconstruction_density.npz",
                                density=overlay.density, pixel_x_nm=overlay.metadata["raster_pixel_x_nm"],
                                pixel_y_nm=overlay.metadata["raster_pixel_y_nm"])
            self._commit()
            self.saved = True
            self.export_button.configure(state="disabled")
            self.status.set("Saved all {} picklists, recipes, pixel assignments, previews, and manifest to {}".format(len(self.runs), output))
        except Exception as exc:
            messagebox.showerror("Cannot save image batch", str(exc), parent=self)

    def _store(self):
        if not self.runs or self.stored:
            return
        try:
            self._check_destinations()
            state = deepcopy(self.app.storage_state)
            # Keep the complete image mapping with each stored design so it survives restarts.
            existing_ids = {item.get("id") for item in state.get("items", [])}
            for run in self.runs:
                if run["id"] in existing_ids:
                    continue
                item = deepcopy(run)
                item["gaussian_reconstruction"] = deepcopy(self.snapshot["gaussian_reconstruction"])
                item["alignment_simulation"] = deepcopy(self.snapshot.get("alignment_simulation"))
                item["alignment_example_seed"] = self.snapshot.get("alignment_example_seed")
                item["color_processing"] = deepcopy(self.snapshot.get("color_processing"))
                item["image_optimization"] = deepcopy(self.snapshot.get("optimization"))
                item["image_geometry"] = deepcopy(self.snapshot["geometry"])
                item["image_encoding_scheme"] = self.snapshot["encoding_scheme"]
                item["image_source"] = self.snapshot["source_image"]
                item["origami_rotation_degrees_clockwise"] = self.snapshot["origami_rotation_degrees_clockwise"]
                item["source_image_rotation_degrees"] = 0
                item["image_fit_mode"] = self.snapshot["fit_mode"]
                state.setdefault("items", []).append(item)
            state["version"] = 2
            state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            save_json(self.storage_path, state)
            self.app.storage_state = state
            self._commit()
            self.stored = True
            self.store_button.configure(state="disabled")
            self.app._refresh_storage_tree()
            self.status.set("Stored all {} image designs with separate destination wells and mixing recipes.".format(len(self.runs)))
        except Exception as exc:
            messagebox.showerror("Cannot store image batch", str(exc), parent=self)


class EncodingComparison(tk.Toplevel):
    """Compare all schemes at the same physical scale without preparing any runs."""

    def __init__(self, owner):
        super().__init__(owner.winfo_toplevel())
        self.owner = owner
        self.title("Compare RGB encoding schemes")
        self.geometry("1200x700")
        self.minsize(620, 400)
        self.summary = tk.StringVar()
        ttk.Label(self, text="Compare encoding schemes", font=("TkDefaultFont", 16, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
        ttk.Label(self, text="Previews fit each complete field to the same display width; physical field sizes are listed. Quadrant versions appear below their base schemes; scroll down to compare. They use a 118.8 × 81.6 nm virtual field. Sampling pitch is not resolution. Barcode errors are not simulated; fiducial sites must be reserved separately.",
                  wraplength=1100).pack(anchor="w", padx=12, pady=(0, 8))
        loading = ttk.Frame(self)
        loading.pack(fill="x", padx=12, pady=(0, 8))
        self.progress_text = tk.StringVar(value="Ready to compare")
        ttk.Label(loading, textvariable=self.progress_text).pack(anchor="w")
        self.progress = ttk.Progressbar(loading, mode="determinate", maximum=len(ENCODING_SCHEMES))
        self.progress.pack(fill="x", pady=(4, 0))
        host = ttk.Frame(self)
        host.pack(fill="both", expand=True, padx=8)
        viewport = tk.Canvas(host, highlightthickness=0)
        horizontal = ttk.Scrollbar(host, orient="horizontal", command=viewport.xview)
        vertical = ttk.Scrollbar(host, orient="vertical", command=viewport.yview)
        viewport.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        viewport.grid(row=0, column=0, sticky="nsew")
        horizontal.grid(row=1, column=0, sticky="ew")
        vertical.grid(row=0, column=1, sticky="ns")
        host.rowconfigure(0, weight=1)
        host.columnconfigure(0, weight=1)
        cards = ttk.Frame(viewport)
        viewport.create_window(0, 0, window=cards, anchor="nw")
        cards.bind("<Configure>", lambda _event: viewport.configure(scrollregion=viewport.bbox("all")))
        self.metrics = {}
        self.previews = {}
        self.buttons = {}
        self.images = {}
        self.optimization_metrics = {}
        self._binned_cache = {}
        self._source_key = None
        family_columns = {key: index for index, key in enumerate(
            key for key, scheme in ENCODING_SCHEMES.items() if not scheme.tiled)}
        for index, (scheme_id, scheme) in enumerate(ENCODING_SCHEMES.items()):
            info = geometry(scheme_id)
            card = ttk.LabelFrame(cards, text="{}-bit {}×{}{}".format(scheme.bit_count, scheme.x_states, scheme.y_states, " · 4 quadrants" if scheme.tiled else (" + parity" if scheme.parity else "")), padding=6)
            card.grid(row=1 if scheme.tiled else 0,
                      column=family_columns[scheme.parent_scheme if scheme.tiled else scheme_id],
                      sticky="n", padx=4, pady=4)
            self.metrics[scheme_id] = tk.StringVar()
            ttk.Label(card, justify="center", textvariable=self.metrics[scheme_id]).pack(pady=(0, 6))
            canvases = []
            for label in ("Binned RGB", "Predicted Gaussian overlay"):
                ttk.Label(card, text=label).pack(anchor="w", pady=(4, 2))
                canvas = tk.Canvas(card, width=264, height=187, background="#15191d", highlightthickness=0)
                canvas.pack()
                canvases.append(canvas)
            button = ttk.Button(card, text="Use this scheme", command=lambda key=scheme_id: self.use_scheme(key))
            button.pack(pady=(10, 4))
            self.previews[scheme_id] = canvases
            self.buttons[scheme_id] = button
        ttk.Label(self, textvariable=self.summary, wraplength=1100).pack(anchor="w", padx=12, pady=10)

    def use_scheme(self, scheme_id):
        if self.owner._scheme_id() != scheme_id:
            self.owner.scheme.set(ENCODING_SCHEMES[scheme_id].label)

    def _show_progress(self, completed, message):
        self.progress.configure(value=completed)
        self.progress_text.set(message)
        # Paint the window and bar before entering each expensive computation.
        # Avoid update(), which could re-enter refresh via queued user actions.
        self.update_idletasks()

    def refresh(self):
        owner = self.owner
        total = len(ENCODING_SCHEMES)
        self.configure(cursor="watch")
        for button in self.buttons.values():
            button.configure(state="disabled")
        self._show_progress(0, "Preparing comparison…")
        try:
            if owner.pixels is None:
                raise ValueError("Upload a valid image to compare the schemes.")
            path = Path(owner.image_path.get())
            source_key = (str(path), path.stat().st_mtime_ns, path.stat().st_size,
                          owner.fit_mode.get(), int(owner.rotation.get()), owner.linear_light.get())
            if source_key != self._source_key:
                self._binned_cache = {}
                self._source_key = source_key
            images = {}
            optimization_metrics = {}
            for index, (scheme_id, scheme) in enumerate(ENCODING_SCHEMES.items()):
                label = "{}-bit {}×{}".format(scheme.bit_count, scheme.x_states, scheme.y_states)
                self._show_progress(index, "{} of {} complete · Loading {}…".format(index, total, label))
                if scheme_id not in self._binned_cache:
                    self._binned_cache[scheme_id] = bin_image(path, owner.fit_mode.get(), int(owner.rotation.get()), scheme_id, linear_light=owner.linear_light.get())
                binned = self._binned_cache[scheme_id]
                designs = owner._compile_scheme(binned, scheme_id)
                if owner.optimize_enabled.get():
                    self._show_progress(index, "{} of {} complete · Optimizing {}…".format(index, total, label))
                    designs, optimization_metrics[scheme_id] = owner._optimize(designs)
                self._show_progress(index, "{} of {} complete · Reconstructing {}…".format(index, total, label))
                achievable = owner._render_overlay(designs)
                images[scheme_id] = (binned, achievable)
                self._show_progress(index + 1, "{} of {} computed".format(index + 1, total))
            self._show_progress(total, "Drawing comparison previews…")
            self.images = images
            self.optimization_metrics = optimization_metrics
            for scheme_id, (binned, achievable) in images.items():
                info = geometry(scheme_id, int(owner.rotation.get()))
                self.metrics[scheme_id].set(("{display_columns} × {display_rows} pixels · {design_count} designs\n"
                    "{display_pixel_pitch_x_nm:.4g} × {display_pixel_pitch_y_nm:.4g} nm/pixel").format(**info))
                if scheme_id in optimization_metrics:
                    metrics = optimization_metrics[scheme_id]
                    self.metrics[scheme_id].set(self.metrics[scheme_id].get() + "\nOptimized RMSE {:.4f} → {:.4f}\nSquared error reduced {:.1%}".format(
                        metrics["baseline_rmse"], metrics["optimized_rmse"], metrics["mse_reduction_fraction"]))
                if scheme_id in optimization_metrics and "validation_optimized" in optimization_metrics[scheme_id]:
                    metrics = optimization_metrics[scheme_id]
                    self.metrics[scheme_id].set(self.metrics[scheme_id].get() + "\nHeld-out RMSE {:.4f} → {:.4f}".format(
                        metrics["validation_baseline"]["expected_mse"]**.5, metrics["validation_optimized"]["expected_mse"]**.5))
                scheme = ENCODING_SCHEMES[scheme_id]
                detail = "\n{} bits · {} sites/bit · {}".format(scheme.bit_count, len(scheme.bit_sites[0]),
                    "even parity" if scheme.parity else "no parity")
                detail += "\n{display_field_width_nm:g} × {display_field_height_nm:g} nm field".format(**info)
                if scheme.tiled:
                    detail += "\nQuadrant shifts: " + " / ".join("{}×{}".format(q["x_states"],q["y_states"]) for q in info["quadrant_sampling"])
                    if info["nonuniform_sampling"]:
                        detail += "\nUnequal sampling; equal physical areas"
                detail += "\nCanvas exposure compensation {:g}×".format(achievable.metadata.get("canvas_area_gain", 1))
                detail += "\nAt least {} destination wells".format(scheme.design_count)
                if scheme.design_count > 384:
                    detail += " (exceeds one plate)"
                if owner.alignment_enabled.get():
                    detail += "\n{} origami total at {} / design".format(
                        scheme.design_count*owner._alignment_settings().origami_per_design,
                        owner._alignment_settings().origami_per_design)
                self.metrics[scheme_id].set(self.metrics[scheme_id].get()+detail)
                owner._draw_pixels(self.previews[scheme_id][0], binned, info, draw_width=252)
                owner._draw_overlay(self.previews[scheme_id][1], achievable, info, draw_width=252)
                active = scheme_id == owner._scheme_id()
                self.buttons[scheme_id].configure(text="Selected" if active else "Use this scheme", state="disabled" if active else "normal")
            settings_summary = ("Replacement optimization ON — applied independently to every scheme. "
                                if owner.optimize_enabled.get() else "Replacement optimization OFF for every scheme. ")
            settings_summary += "Base transfer {} nL; excess multipliers {}. ".format(
                owner.app.transfer_volume.get(), ", ".join(
                    "{} {}×".format(color, view.excess_multiplier.get()) for color, view in owner.color_views.items()))
            if owner.alignment_enabled.get():
                alignment = owner._alignment_settings()
                settings_summary += ("Alignment: X SD {translation_x_nm:g} nm, Y SD {translation_y_nm:g} nm, "
                    "rotation SD {rotation_deg:g}°, {origami_per_design} origami/design, "
                    "center ({center_x_nm:g}, {center_y_nm:g}) nm, variability penalty {variability_penalty:g}. "
                    "Previews show expected alignment results. ").format(**vars(alignment))
            else:
                settings_summary += "Alignment uncertainty OFF. "
            light_summary = ("Linear-light processing. " if owner.linear_light.get() else "Legacy direct RGB processing. ")
            light_summary += ("Fixed total signal across schemes (48/design-count weighting); unequal quadrant sampling is balanced by per-design weights; no automatic mixing changes. "
                              if owner.normalize_designs.get() else "Equal per-design weight: more designs produce more total signal. ")
            light_summary += "Canvas-area exposure compensation enabled. " if owner.compensate_area.get() else "No canvas-area exposure compensation. "
            self.summary.set(settings_summary + light_summary + "{} · origami {}° clockwise · FWHM {} nm · shared exposure {}×. Equal channel brightness. Clipped pixels: {}.".format(
                owner.fit_mode.get(), owner.rotation.get(), owner.gaussian_fwhm.get(), owner.gaussian_exposure.get(),
                "; ".join("{}×{}: {:.1%}".format(ENCODING_SCHEMES[key].x_states, ENCODING_SCHEMES[key].y_states,
                         result.metadata["clipped_pixel_fraction"]) for key, (_, result) in images.items())))
            self._show_progress(total, "Comparison complete · {} of {} schemes".format(total, total))
        except Exception as exc:
            self._show_progress(0, "Comparison could not be completed")
            self.images = {}
            self.optimization_metrics = {}
            for metric in self.metrics.values():
                metric.set("")
            for canvases in self.previews.values():
                for canvas in canvases:
                    canvas.delete("all")
            for button in self.buttons.values():
                button.configure(state="disabled")
            self.summary.set("Cannot compare: {}".format(exc))
        finally:
            self.configure(cursor="")
