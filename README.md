# Mona Lisa: DNA-PAINT analysis and Echo picklist generation

This repository contains two Python desktop applications for designing DNA-origami experiments and analyzing their DNA-PAINT localization data:

- **PAINT Analysis** loads localization tables, corrects drift, measures regions of interest and binding events, identifies and decodes origami, and reconstructs images from measured signal or molecular site occupancy.
- **Echo Picklist Generator** turns staple-replacement selections or an input image into source-to-destination transfer CSVs and synthesis recipes. It also designs barcode templates, predicts image appearance, manages plate and barcode usage, and provides split–pool encoding research tools.

Both applications use Tkinter and run locally. They can be used independently. For image experiments, the generator's calibrated templates, bit definitions, and batch manifest connect experimental design to PAINT classification and reconstruction. PAINT Analysis starts from **localization tables**, not raw microscope movies; the generator exports instructions and does not operate an Echo instrument.

## Contents

- [Getting started](#getting-started)
- [Shared concepts and the design-to-analysis workflow](#shared-concepts-and-the-design-to-analysis-workflow)
- [PAINT Analysis](#paint-analysis)
  - [Load, correct, and inspect localizations](#load-correct-and-inspect-localizations)
  - [Histograms, filters, temporal metrics, and linking](#histograms-filters-temporal-metrics-and-linking)
  - [Detect, align, and classify origami](#detect-align-and-classify-origami)
  - [Inspect classifications and site statistics](#inspect-classifications-and-site-statistics)
  - [Large datasets, checkpoints, and saved analyses](#large-datasets-checkpoints-and-saved-analyses)
  - [Reconstruct measured images](#reconstruct-measured-images)
  - [Binary occupancy reconstruction](#binary-occupancy-reconstruction)
  - [Performance and diagnostics](#performance-and-diagnostics)
- [Echo Picklist Generator](#echo-picklist-generator)
  - [Manual replacement and synthesis workflow](#manual-replacement-and-synthesis-workflow)
  - [Transfer calculations and mixing recipes](#transfer-calculations-and-mixing-recipes)
  - [Plate usage, storage, and barcode reservations](#plate-usage-storage-and-barcode-reservations)
  - [Build templates and digital-bit schemas](#build-templates-and-digital-bit-schemas)
  - [Encode an RGB or grayscale image](#encode-an-rgb-or-grayscale-image)
  - [Simulation, optimization, and scheme comparisons](#simulation-optimization-and-scheme-comparisons)
  - [Split–pool and digital encoding research](#splitpool-and-digital-encoding-research)
- [Files, exports, and persistent state](#files-exports-and-persistent-state)
- [Troubleshooting](#troubleshooting)
- [Repository layout and development checks](#repository-layout-and-development-checks)
- [Detailed references](#detailed-references)

## Getting started

### Requirements

Use a Python installation with **Tkinter**, virtual-environment support, and pip. PAINT's macOS launcher requires Python **3.10+**; the picklist launchers check for **3.9+**. Python **3.12** is a practical common starting point for the two environments. Actual installability also depends on the scientific packages available for your Python version and platform.

PAINT uses NumPy, SciPy, pandas, h5py, Matplotlib, PyYAML, tifffile, scikit-learn, tqdm, and `picassosr`. Picklist image and research features use Pillow, NumPy, Matplotlib, and SciPy. The applications' `requirements.txt` files define their dependencies. Internet access is needed when installing missing packages; a CUDA GPU is optional for PAINT.

### Launch with the supplied scripts

| Platform | PAINT Analysis | Picklist Generator |
| --- | --- | --- |
| macOS | Double-click [`paint_analysis/run_paint_gui.command`](paint_analysis/run_paint_gui.command). | Double-click [`picklist_generator/launch.command`](picklist_generator/launch.command). |
| Windows | Double-click [`paint_analysis/run_paint.bat`](paint_analysis/run_paint.bat); `run_paint_gui.bat` is an alias. | Double-click [`picklist_generator/launch.bat`](picklist_generator/launch.bat). |
| Linux | Use the manual installation below. | Run `bash picklist_generator/launch.sh` from the repository root. |

If a macOS launcher needs executable permission, run this once from the repository root:

```sh
chmod +x paint_analysis/run_paint_gui.command picklist_generator/launch.command
```

PAINT's Windows launcher reuses a suitable Python 3.10–3.14 installation with the required GUI/environment components, or installs Python 3.12 for the current user. Its private environment is outside the repository:

- macOS: `~/Library/Application Support/PaintAnalysis/venv`
- Windows: `%LOCALAPPDATA%\PaintAnalysis\venv`

Set `PAINT_ANALYSIS_HOME` before launching to relocate PAINT's environment and application state. Picklist launchers create `picklist_generator/.venv`. They install requirements when their image-library import check fails; after updating an existing installation, explicitly install the requirements again if a newly needed dependency is missing.

### Manual installation

These commands start in the repository root. Keep separate environments so dependency changes in one application do not affect the other.

**macOS/Linux — PAINT:**

```sh
cd paint_analysis
python3 -m venv "$HOME/.venvs/mona-lisa-paint"
source "$HOME/.venvs/mona-lisa-paint/bin/activate"
python -m pip install -r requirements.txt
python paint_analysis_gui.py
```

**macOS/Linux — picklists, in a new terminal:**

```sh
cd picklist_generator
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python picklist_app.py
```

**Windows PowerShell — PAINT:**

```powershell
cd paint_analysis
py -3.12 -m venv "$env:LOCALAPPDATA\MonaLisaPaintManual\venv"
& "$env:LOCALAPPDATA\MonaLisaPaintManual\venv\Scripts\python.exe" -m pip install -r requirements.txt
& "$env:LOCALAPPDATA\MonaLisaPaintManual\venv\Scripts\python.exe" paint_analysis_gui.py
```

**Windows PowerShell — picklists, in a new terminal:**

```powershell
cd picklist_generator
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe picklist_app.py
```

On Linux, install your distribution's Tkinter package, commonly `python3-tk`, if `python3 -m tkinter` fails. The desktop applications need a graphical session.

## Shared concepts and the design-to-analysis workflow

| Term | Meaning in these applications |
| --- | --- |
| Localization | A fitted molecular-event position, usually with a frame number and optional photon, precision, width, and background fields. Repeated localizations are not automatically independent molecules. |
| ROI | A rectangular region of interest selected on a localization map. |
| Origami / candidate | A molecular structure, or a detected region being tested as one. A candidate can be accepted, unclassified, excluded, or suppressed as a duplicate. |
| Fiducial / alignment group | An expected bright pattern used to establish position and orientation. It does not encode a variable barcode bit. |
| Dark alignment group | Explicitly authored regions expected to lack signal, used to check a fitted pose. |
| Logical bit / digital pixel | One ON/OFF decision supported by one or more physical sites. Its group ID connects measurements to classification templates. |
| Analog image site | A position whose programmed replacement fraction represents image intensity. It is distinct from a binary barcode site. |
| Replacement fraction | The requested fraction of a site's target concentration supplied by a replacement staple; the remaining fraction is original competitor. |
| Batch manifest | `image_manifest.json`, which records how designs, barcodes, channels, physical sites, and image locations correspond. |

A complete image experiment follows this sequence:

1. **Design:** load an image in Picklist Generator; choose the canvas, barcode schema, layout, and synthesis settings.
2. **Review and export:** prepare the designs, inspect transfers and recipes, and save the full image batch. Retain the manifest and alignment-template exports with the experiment.
3. **Acquire:** carry out synthesis and imaging separately, then produce localization tables using your localization workflow.
4. **Measure:** load those tables in PAINT, verify calibration, correct drift, and validate detection/alignment/classification on a representative ROI.
5. **Scale:** analyze additional distributed tiles or the entire field and save the analysis/checkpoints.
6. **Reconstruct:** load the saved batch manifest into PAINT's composite editor, verify mappings and the image-side region, and build localization-overlay or binary-occupancy images.

For the classification handoff, retain **three distinct inputs**: the shared detection/fiducial template, the reusable digital-group JSON, and the per-class barcode PNGs. Export the reusable JSON with **Save bit schema JSON** in Origami Templates. For reconstruction, retain the full **image batch manifest**; an Echo CSV or the alignment-template folder's `manifest.json` is not a substitute.

Physical calibration matters throughout. Localization input can be in camera pixels or nanometers; drift traces and template geometry must use the correct conversion. Template grid coordinates are one-based `[row, column]`. Picklist templates preserve physical geometry in PNG metadata and JSON sidecars. Load that calibration instead of estimating nanometers from the apparent image size.

## PAINT Analysis

The main workspace has **Raw**, **Corrected**, **Linked**, **Filtered**, **Histogram**, **Temporal**, and **Classification** tabs. The Classification workspace organizes source selection, identification, and overlay inspection in a collapsible sidebar. Older documentation sometimes calls it the Origami or Origami Overlay tab.

### Load, correct, and inspect localizations

1. Open a Picasso `.h5`/`.hdf5` localization file or a `.csv` localization table.
2. Check the displayed metadata, especially pixel size and acquisition dimensions. Keep any companion YAML beside the input file.
3. Inspect the Raw map, select the drift method, and apply correction.
4. Inspect the Corrected map and drag a rectangle to select an ROI, or use **Select Whole Image**.
5. Render, analyze, or export the selected data as needed.

CSV import accepts Picasso-style `frame`, `x`, and `y` and recognized headers such as `frameIndex`, `x (nm)`, and `y (nm)`. Common photon, background, PSF-width, localization-precision, and channel fields are imported when present. Nanometer coordinates are converted to Picasso pixel units using YAML `Pixelsize`; without that metadata, the fallback is **130 nm/pixel**. Verify that fallback before interpreting physical distances. Imports show progress and load large tables in chunks.

**Drift choices:**

| Method | Use |
| --- | --- |
| `none` | Treat the loaded coordinates as the corrected table. |
| `rcc` | Picasso redundant cross-correlation. Segmentation sets frames per drift segment; optional Fourier lattice suppression reduces periodic lattice interference. Set lattice pitch to `0` to disable that suppression. |
| `aim` | Picasso AIM, with intersection and ROI settings in nm. The implementation bounds trial-shift memory. |
| `file` | Load a per-frame drift CSV and subtract its displacement. Optional z drift is applied when available in both inputs. |

Use **Save Drift CSV…** to export the applied trace. To estimate drift from a crop, save a **Raw ROI CSV** with its YAML, correct that crop, export its drift, then load the original raw acquisition and apply the drift file. The frame numbering must match and the trace must cover every required frame; missing drift values are rejected.

Map rendering uses Picasso. **Render pixel**, **Render blur**, and **Min blur** control rendering without rerunning drift correction. **Dynamic zoom rendering** progressively rerenders the visible viewport, down to 1 nm/render pixel. An explicit render uses the entered resolution; later zoom/pan actions resume automatic sizing. Toolbar Home restores the acquisition or saved crop bounds.

**Auto density (histogram)** estimates contrast from populated pixels; manual minimum/maximum density and the density multiplier control brightness. These are display settings, not localization filters.

**ROI CSV export:** **Save Raw ROI CSV** and **Save Corrected ROI CSV** select points using the corresponding coordinates, including boundaries. They retain absolute coordinates and original frame numbers. CSV-origin exports preserve original column order, text fields, and units; corrected exports replace coordinate values. The original CSV must remain available for this preservation. HDF5-origin exports contain loaded numeric fields in Picasso pixel units. Histogram filtering and linking do not alter these exports. Companion YAML records calibration and crop bounds, and the source file is protected from overwrite.

### Histograms, filters, temporal metrics, and linking

After selecting an ROI, use **Histogram** to inspect photons, localization precision, frame number, PSF sigma x/y, local background, and nearest-neighbor distances where the input supplies the required data. Apply histogram filters and inspect the resulting **Filtered** map. Check the active filters before using filtered data as a classification source.

Configure **Exposure**, **Link radius**, and **Max gap**, then use **Run Linking Analysis** to combine nearby observations into binding events. The summary shows event duration in frames and milliseconds, localizations per event, and photons per event. **Color By Links** colors multi-localization events by ID and shows isolated observations in gray. Linking can use an ROI or the whole image; whole-field tiled work with linked events requires whole-image linking first.

The **Temporal** tab plots localization or linked-event metrics over frame windows. Add a frame number and text under **Vertical frame annotations** to mark acquisition changes. Annotations can be updated, removed, or cleared without recalculating the metric.

### Detect, align, and classify origami

Start with a representative ROI so you can inspect the effect of settings before committing to a large run.

1. **Load source data.** In Classification, choose corrected localizations or linked events, select the ROI/filter options, and click **Load Source Data**. Linking is optional. Corrected localizations are the default source.
2. **Step 1 — detect candidates.** Load the calibrated shared fiducial image using **Load Detection Template**. Review the physical geometry, pick-bin size, density threshold, recovery distance, and point-count cutoff; run Step 1 and inspect coarse density/components. **Clear Template · Use Density Detection** enables connected-bin detection with its Signal gap control. Synthetic grids and custom bright-on-dark raster templates are also supported.
3. **Step 2 — fit and inspect.** Fit rigid rotations/translations and inspect footprints, fiducial support, overlap, and quality diagnostics. This locks the pose used by later measurements. The **Alignment stages** view explains retained/discarded refinements. Optional corner and correlation gates are separate from display toggles; image fits still require independent fiducial support at both ends.
4. **Step 3 — measure digital pixels.** Load the standalone logical-bit JSON and measure its site groups on the locked poses. Review support, prominence, ON/OFF thresholds, evidence-site subsets, and group brightness factors.
5. **Step 4 — classify.** Load the per-class barcode templates and classify the measured patterns. Inspect type counts, unclassified candidates, and QC before proceeding. The detection image, group schema, and classification templates must describe compatible physical grids.
6. **Step 5 — optional tiles.** Use the validated settings to inspect random windows, analyze distributed tiles, or analyze the whole image.

Changing source/detection settings requires rerunning from Step 1. Changing pose settings requires Step 2 onward; changing digital measurement thresholds requires Steps 3 and 4. Loading a different class set requires rerunning classification. Progress bars show which stages have completed. **Identify Origami** remains a shortcut once the needed inputs are loaded.

**Digital classification is exact pattern matching.** Every expected ON and OFF group must agree by digital-pixel ID. A unique eligible match receives the candidate; unknown patterns, multiple matching templates, or invalid evidence remain unclassified. There is no nearest-barcode fallback, even with one loaded template. Alignment-excluded candidates are removed before lookup and do not enter class/unclassified count denominators. Legacy image-only templates use their separate image-based rules.

**Alignment and orientation controls:**

- Site support, prominence, spacing, correlation, point limits, and optional corner support help separate valid structures from poor fits. Diagnostic overlays expose counts, measured centroids, prominence sampling, and failure reasons.
- **Max overlap (%)** checks fitted active footprints; its default of 0% rejects both fits when they overlap. It cannot identify every superposition that was detected as only one candidate.
- Explicit dark groups test the density in authored empty regions against supported bright sites. A failed dark check can trigger a constrained 180° retry; the retry must still satisfy bright-fit and dark checks.
- **Check 0° / 180° orientation** compares the two directional fits and excludes ambiguous candidates when enabled. A symmetric fiducial cannot establish direction with this check; lowering the score-gap threshold does not recover missing orientation information. Review the diagnostics before interpreting image-side placement.
- Global horizontal/vertical template mirrors carry the corresponding group geometry. Calibrated Picklist PNGs also have a defined column-convention conversion; do not add an ad hoc mirror without checking the resulting physical sites. Reload templates/schema and rerun affected stages after changing calibration or authored dark groups.
- Per-group `brightness_factor` normalizes support for groups with different expected brightness. It changes detection evidence, not physical transfer volumes or the rendered template's brightness.

### Inspect classifications and site statistics

Select a classification, **All templates**, or **Unclassified**, then select the desired view. Available inspection tools include:

| View or feature | What it provides |
| --- | --- |
| Candidate/coarse-density overview | Connected regions, fitted footprints, accepted/rejected status, optional theoretical bright/dark sites, and QC labels. |
| Type counts | Assigned counts by template and unclassified count, with a configurable x-axis order. |
| Open vs closed | Paired proportions for matching `_open`/`_closed` names and recognized ON/OFF aliases; unmatched names and unclassified particles are excluded. |
| Aligned density | Mean aligned localization density for one class or a panel grid for all classes. |
| Individual galleries and details | Paged particles, saved candidate IDs, rejection reasons, alignment stages, site assignments, and digital-bit derivation. Gallery resolution and panel size are configurable. |
| Site summaries | Integrated density, mean counts, site occupancy, and digital-pixel spatial heatmaps. |
| Digital-group threshold audit | Saved support, score, and prominence failures and group ON fractions. |
| Unmatched-pattern audit | Common unknown patterns, distances and ties to known templates, and all-ON dropout comparisons. These do not establish a particle's true class. |
| Unclassified investigation | Filter by rejection group, inspect evidence distributions, and run support/prominence threshold sweeps. |

Use **Build Fast Overlay (no G5M)** to combine aligned particles and obtain approximate site-assignment/occupancy statistics. **Refine Current Overlay with G5M** optionally applies Picasso's Gaussian mixture analysis to those accepted particles. It is not required for decoding or whole-image processing. Overlay pixel size, padding, blur, colormap, legends, white digital masks, and fiducial color control inspection; they do not replace the saved classification decisions.

Symmetric synthetic grids can use equal-weight 0°/180° population contributions to avoid arbitrary orientation bias. Barcode classification overlays preserve saved poses and saved digital calls. An asymmetric fiducial is needed to attach a physical directional meaning to a symmetric grid.

Threshold sensitivity sweeps reuse saved measurements and fixed eligibility, report predicted recoveries/losses/switches, and **do not apply new assignments**. Their baseline must reproduce the saved calls. Some audit expectation plots use the named experimental reference mixture `align_fid:code1:code2:code3:code4:full = 1:1:1:2:1:1`; this is a reference, not an inferred abundance or a classification prior.

Use **Export Site CSVs** for per-particle and per-site statistics, gallery PDF export for paged inspection, and **Export Classification Audits** for a ZIP of candidate/group evidence, thresholds, pattern distances, and completed sweep tables. Candidate IDs connect the tables. **Plot Only**, **Full Screen** (Escape to exit), and **Pop Out** provide larger inspection views.

### Large datasets, checkpoints, and saved analyses

**Validate before tiling.** Your validated ROI defines tile dimensions and the grid alignment. Candidate extraction uses neighboring halo points while retaining candidates centered in each core, reducing boundary truncation and double counting. Saved ROI-file bounds constrain the field. Whole-image processing can clip edge tiles; distributed/random inspection uses fully fitting windows.

- **Inspect Random ROI** spot-checks other windows without replacing the validated ROI/settings.
- **Analyze N Tiles** spreads a deterministic sample across the field (described as distributed-tile analysis in the detailed manual). Results describe that sample, not an extrapolated whole-field count.
- **Analyze Whole Image as Tiles** processes the field sequentially with the validated settings.
- Choose a checkpoint destination before a tiled run. The app creates a `paint-tiles-…` directory containing `run.paintanalysis` and completed tile archives. Keep the directory together.
- **Pause after Current Tile & Preview** finishes/checkpoints the current tile, then exposes partial results. Wait for the Paused status before inspection/reconstruction; **Continue Paused Analysis** continues in the same application session.
- **View Completed Tiles…** opens the readable completed subset without calculating missing tiles. Plots mark partial analyses.
- **Resume Tiled Analysis…** restores the saved settings and computes missing work after a restart. A partially processed tile may run again; completed checkpoints are reused.

**Save Analysis…** writes a `.paintanalysis` archive containing source tables, aligned particles, templates, schemas, measurements, classifications, settings, and saved views/overlays. **Load Analysis…** restores those results without repeating detection or classification. Original localization/template files are not required to browse the archive. Closing with results prompts to save, discard, or cancel.

Opening a tiled run can load the full result, view completed tiles, resume, or load only the first N completed windows. The latter remains explicitly partial. Individual tile files are not standalone analysis archives.

**Classify Unclassified Only** can add missing classes after loading an analysis: load compatible new class templates and apply the action. Existing assignments remain intact; excluded and duplicate-suppressed particles are left alone. Saved measurements and consistent schema/candidate ordering are required. Save under a new name to retain the earlier result.

Large loads extract arrays to `.analysis-cache` beside the archive and use memory maps. Tiled aggregation uses `.combined-cache` inside the recovery folder. These caches need additional disk space; keep active caches in place while the analysis is open. Tiling reduces working memory but does not eliminate source-data, metadata, or reconstruction memory requirements. Checkpoint write failures stop the run while retaining completed work.

Automatic drift-session caching is separate from saved analyses: it can restore the latest valid development session and rejects changed sources/drift files. Set `PAINT_ANALYSIS_DISABLE_DEVELOPMENT_CACHE=1` before launch to disable automatic restoration.

### Reconstruct measured images

Open **6 · Reconstruct Composite Image…** after classification, including a completed subset of a tiled run.

1. Choose **Load saved RGB picklist batch…** and load the batch's `image_manifest.json`. It supplies barcode-to-color/position mappings, active image sites, and display orientation. Manual mappings and saved encoding/mapping JSON are also supported for overlay reconstruction.
2. Review the per-template table. Original design names and exported barcode-template names are matched automatically; assign renamed templates explicitly. Unmatched controls are excluded until mapped and included.
3. Enable **Preview image region** and check the image-side extraction rectangle against measured localizations. The local frame uses X right/Y up; decoded image shifts use right/down. Check saved template mirrors, section orientation, and final display rotation separately.
4. Select **Localization Overlay (Existing)**, set output pixel size and rendering controls, then **Build composite**.
5. Inspect the RGB image, channels, contribution counts, and Coverage. Export PNG, floating-point NPZ densities, and the JSON report.

This mode extracts measured signal from the source coordinates using accepted saved poses, including image-side signal outside the barcode crop. Unclassified/rejected particles do not contribute. Source localizations shared by multiple accepted image regions are excluded to prevent duplicate use.

By default, each species is averaged over its accepted origami, including copies with no extracted image signal; summed signal is optional. Per-channel brightness controls, per-section and whole-image Gaussian smoothing, section/display mirrors and rotations, and template inclusion support reconstruction inspection. Optional uniform fill for an excluded color section is a display/reconstruction choice, not a recovered measurement.

**Overlap blend width/height** can extend extraction with Gaussian-weighted tails. Marker-exclusion sigma attenuates barcode/fiducial signal only in that extension. **Normalize overlapping sections** optionally normalizes same-channel blended contributions. Coverage still describes the normal extraction rectangles, and must not be interpreted as measured molecular occupancy.

For faster revisits:

- **Load reconstruction from analysis** reads reconstruction inputs from a `.paintanalysis` archive or tiled run without restoring the full analysis workspace.
- **Save reconstruction session…** writes a compact `.paintrecon` with accepted poses, nearby source points, the embedded batch manifest, mappings, and settings.
- On a fresh launch, use **Open reconstruction session…** directly and then **Build composite**; loading the full analysis first is unnecessary.
- A reopened `.paintrecon` has fixed extraction/blend geometry. Rendering, placement, brightness, and inclusion settings remain editable. To change the extraction region, return to the original analysis and save a new reconstruction session. Re-saving an open session requires a different filename.

### Binary occupancy reconstruction

**Binary Occupancy (New)** measures an ON/OFF observation for each analog site of each accepted origami, then aggregates molecular votes. **Compare Both** shows it beside the localization overlay. Both modes reuse the same saved decoding and poses.

1. Load a full RGB batch manifest with physical analog-site identifiers and pixel assignments. A codebook alone is insufficient. The current implementation supports the full 12-column/8-row design, including a reduced active analog canvas.
2. Set **ROI radius (nm)**, **ON count threshold**, and optional **Uncertain count margin**. Starting values are 2 nm, 3 localizations, and zero margin; these need experimental calibration.
3. Set **Occupancy Gaussian FWHM (nm)** or leave it blank to use the manifest's width, with a 7.5 nm fallback for older manifests.
4. Build the composite and inspect **Site occupancy**, coverage, channel views, template heatmaps, individual-site examples, count distributions, barcode maps, and target errors.
5. Export the measurement bundle, and optionally **Export sample-count study…**.

Each localization is assigned to at most one nearest analog site within the radius. A count at or above the threshold is ON; a valid lower count is OFF. Uncertain observations are omitted from the denominator:

```text
occupancy = ON observations / valid (ON + OFF) observations
```

For example, 140 ON among 200 valid origami gives 0.70, regardless of how many excess localizations the ON molecules produced. Zero valid observations produce missing data, not zero occupancy. Uncertainty includes the optional threshold margin, out-of-region site ROIs, shared source signal, available orientation-ambiguity flags, and explicit manual exclusions. Rejected/unclassified particles never enter the denominator.

**Load occupancy settings JSON…** accepts per-position thresholds and manual uncertain-site assignments using exact manifest site names and saved candidate IDs. Settings persist in mappings/reconstruction sessions and reports.

The **Site occupancy** view is a fixed 0–1 fraction map. The rendered composite instead sums Gaussian spots at physical site coordinates with integrated weights proportional to occupancy and the manifest's display gains. FWHM changes appearance, not the measured fractions. Missing numerical values remain NaN; RGB previews mark missing channels magenta.

Exports include PNGs, NPZ measurement/render arrays, individual observations, site summaries, exclusions, per-position/per-barcode errors, and a JSON provenance report. Target errors compare against programmed `replacement_fraction`, not the photograph's raw RGB values. The optional study samples whole molecules without replacement within each barcode, by default N = 10, 20, 50, 100, 200, 300, with 20 repeats and seed 0; classes too small for an N remain missing.

Localization density and occupancy have different meanings. Their comparison image is a display-scale diagnostic, not calibrated occupancy accuracy. Programmed ON/OFF controls must be independently validated before interpreting threshold-based fractions as experimentally accurate occupancy or assembly yield.

### Performance and diagnostics

The normal classification backend is optimized CPU. Set `PAINT_CLASSIFICATION_COMPUTE` before launch:

| Value | Behavior |
| --- | --- |
| `cpu` | Default optimized CPU operations. |
| `auto` | CUDA where available, with CPU fallback on initialization/operation failure. |
| `gpu` | Require CUDA for GPU-eligible operations; report failure instead of silently falling back. |
| `reference` | Original numerical paths for comparison. |

CUDA acceleration requires a compatible NVIDIA GPU; it does not use Apple GPUs. Eligible operations include large detection maps, digital-group/distance batches, and overlay filtering; pose selection remains on CPU. Optional dependencies are in `paint_analysis/requirements-gpu.txt`. Windows GPU/reference launchers use an isolated `.gpu-venv` and `.gpu-state`; see the [acceleration guide](paint_analysis/README.md#optional-classification-acceleration).

`PAINT_CLASSIFICATION_LOG` can point to a JSONL timing/worker-diagnostics file. The application also writes local crash diagnostics under its state directory's `diagnostics/`: `latest.json`, `events.jsonl`, sampled `stacks.txt`, fatal tracebacks, and previous-run copies. These record paths, actions, and diagnostic metadata; they are not uploaded. Preserve them with the checkpoint folder when investigating an interrupted run.

## Echo Picklist Generator

The desktop tabs are **1. Replacements**, **2. Run & Mixing Settings**, **3. Results**, **4. Destination Plate**, **5. Storage**, **6. RGB Image**, and **7. Origami Templates**. RGB Image contains **Image generator** and **Split–pool encoding analysis** subtabs.

### Manual replacement and synthesis workflow

1. In **Replacements**, choose the base plate and hinge type. Select staples in the appropriate panels: PAINT P1, PAINT R1, LowConc PAINT R1, miRNA, or the bundled extension/aptamer/MB panels. Gray unavailable sites cannot be selected.
2. Set **Manual synthesis per run**: actual base stock/source, final volume, book-base and scaffold targets, minimum transfer, and transfer increment.
3. Set each panel's desired staple concentration and **Replacement % for new selections**, then click sites. **New selections (nM; blank = default)** supplies optional per-site concentration overrides.
4. Inspect the selected-site targets/highlights and use **Check selections**. Ordinary duplicate replacements of the same logical base well are clashes. **Save selections PDF** records the selected panels and settings.
5. Click **Make Picklist** to validate compatibility and unlock **Run & Mixing Settings**. Changing selections/source sheets locks it again until revalidated.
6. Review destination plate/wells, source labels/types, capacity, stock compositions, and final buffer targets. Synthesis summaries link back to the editable controls in Replacements.
7. Click **Generate Preview** and inspect **Results**. This calculates transfers and recipes without saving or consuming destination wells.
8. Choose **Save picklist + recipe**, **Send to storage**, or both. Saving/storing records usage once for that prepared result and advances the pending destination wells.

Each selected site retains its percentage. Changing the percentage picker affects new selections only; deselect/reselect a site to change it. A site-specific concentration override also remains fixed when the plate default changes. The read-only calculated base transfer follows the synthesis settings; infeasible existing selections are reported rather than silently rounded.

**Base plates and hinges:** the app supports the 200 µM base plate and both original-layout and interleaved 20 µM base inventories. Logical replacement keys remain distinct from actual source-well addresses. The interleaved inventory uses the ordered rigid connectors; flexible hinges are unavailable for that inventory. Other supported inventories offer flexible or rigid hinges as defined by their sheets. See the [interleaved plate/loading guide](picklist_generator/replacement_sheets/book_base_20uM_interleaved.README.md).

**Low-concentration and biotin panels:** LowConc PAINT R1 uses the R1 replacement layout with its own stock/source definition. Its D-Biotin panel uses the eight additions at `SourcePlate8[8]`, P17–P24; selecting the same biotin target in another panel is a clash. Source CSVs and displayed stocks determine transfers, so confirm them against the loaded laboratory plates.

**miRNA:** independent miR-21, miR-141, miR-155, miR-210, and miR-223 panels, Smart common locks, and previous row-B panels preserve explicit selections when you switch views. No target locks are automatically selected. A Smart common lock overlapping an R1 selection upgrades to the combined lock+R1 staple, marks the R1 site **via lock**, and produces one combined transfer using the lock's settings. Deselecting either component restores the corresponding standalone behavior. Other clashes remain errors. The [miRNA reference](picklist_generator/README.md#independent-mirna-panels) lists physical mappings and source variants.

### Transfer calculations and mixing recipes

The basic source transfer is:

```text
transfer_nL = 1000 × final_volume_µL × target_concentration_nM / stock_concentration_nM
```

For a replacement fraction `f`, replacement and original competitor use separate stocks:

```text
replacement_nL = 1000 × final_volume_µL × site_target_nM × f / replacement_stock_nM
competitor_nL  = 1000 × final_volume_µL × site_target_nM × (1 − f) / base_stock_nM
```

Every nonzero transfer must meet the configured **minimum** and be a multiple of the **increment**. Both default to 25 nL and are independent; calculating a 50 nL base transfer does not change the increment to 50 nL. Barcode sites are always binary. Analog/manual sites use only fractions whose replacement and competitor transfers are both feasible.

For example, a 5 nM base target in 100 µL from 20 µM stock requires 25 nL per base staple. A 20 nM replacement-site target with 20 µM stock on both sides permits 0%, 25%, 50%, 75%, and 100% replacement in 25 nL steps. Raising the final volume can allow finer fractions without changing concentration targets.

Manual and RGB workflows use the same mixing solver but maintain **independent synthesis settings**. It accounts for the collected staple mix, scaffold, and TE/Mg/Na carried by stock solutions, solves nonnegative buffer additions, then fills with water to the requested final volume. Non-1× TE targets and redundant stock compositions are supported. Invalid/nonfinite inputs, insufficient pooled volume, excessive carried salts, impossible buffer targets, or total volumes exceeding the requested synthesis volume fail with an explanation.

The inputs currently describe one shared staple-buffer composition, not a different buffer for every staple plate. A design spread over multiple destination wells is pooled for **one recipe per design**; the recipe does not apply separately to each collection well.

### Plate usage, storage, and barcode reservations

**Destination Plate** displays a 16×24 map with used, capacity-limited, and pending wells. Click a well to inspect recorded transfers and volume, edit its label, or explicitly change its Used state. Labels do not change Echo well addresses.

Automatic allocation finds unused wells while respecting capacity and reservations. **Destination starting well / Start at well** sets where allocation begins. RGB also supports manual destination assignment. More than one destination well may be needed for a design; a 384-design batch is not guaranteed to fit one 384-well plate.

**Start fresh plate** clears recorded usage and stored-run destination reservations for the named plate after confirmation, retaining stored outputs. It invalidates prepared RGB runs targeting that plate. This is a software ledger operation, not evidence that a physical plate was emptied or replaced.

**Storage** keeps named runs, transfer rows, individual mixing recipes, and selection/image metadata. Select runs and use **Generate combined picklist + recipe** to concatenate transfer rows while preserving separate recipes. Reuse of the same destination plate/well across combined runs is rejected. **Generate selections PDF** produces a combined inspection report. Combined exports go to the configured output root and can replace files with matching names; individual saved runs use new timestamped directories.

**Barcode collision protection** is tracked separately from destination usage:

- RGB protection is active independently of manual design choices. Manual **Use barcodes for replacements** is optional and off by default.
- With protection enabled, load the schema or use the RGB schema, set the bit count, and use the same **Experiment / pool** name for everything decoded together.
- Bits must be wholly ON or OFF and required fiducials ON. **Select an unused code + fiducials** applies an available pattern while preserving unrelated image-site selections.
- Preparation checks availability; saving/storing rechecks and reserves codes. Recipe-only export does not reserve codes or wells.
- **Import used codes…** accepts full image manifests and exported registry JSON. **Registry…** shows run/destination/origin history; export it for a portable record.
- **Release selected reservation…** permits deliberate reuse after confirmation and retains an audit event. Deleting a stored run or starting a fresh plate does not release its barcode.
- Uniqueness compares physical bit-site patterns, not just integer code labels or schema names. Incompatible physical layouts in the same pool are blocked.

Old manual runs without barcode metadata cannot be reconstructed automatically into the registry. Import their complete usage record before relying on apparent availability.

### Build templates and digital-bit schemas

Open **Origami Templates**, or use **Make template from this panel** to start from a manual selection. The editor provides geometry controls and a **Custom digital-bit groups** workspace.

1. Set the physical grid, spacing, margin, and raster calibration.
2. Select physical sites and create named **Digital bit**, **Alignment-only**, or **Dark (alignment)** groups.
3. For digital groups, configure evidence sites and a positive brightness factor where needed; choose the group's preview ON/OFF state. Alignment groups are always present. Dark groups use a mask radius and cannot share sites with bright groups.
4. Save reusable definitions with **Save bit schema JSON**.
5. Export each class with **Save template PNG**. The calibrated bright-on-dark PNG embeds metadata; a same-name JSON sidecar records geometry, groups, and active bits.

The general editor allows arbitrary group sizes and arrangements, and starts without predefined groups. Overlapping bright groups use union/OR semantics. Evidence-site subsets let PAINT omit ambiguous positions while still rendering the full physical group.

For RGB, use **Edit in Origami Templates…**, **Apply to RGB Image**, or **Use Origami Templates**. RGB has stricter physical-layout rules: use the 8×12 grid, the chosen number of digital groups, valid barcode-region sites, and nonconflicting permanent fiducials. The RGB encoding determines each design's ON/OFF pattern; the editor's current bit checkboxes are only its preview. Numbered Bit labels preserve ordering.

Preserve PNG metadata/sidecars and use the exported calibration in PAINT. A figure screenshot with labels or scale bars is not an equivalent alignment template.

### Encode an RGB or grayscale image

1. Open **RGB Image → Image generator** and upload PNG, JPEG, TIFF, BMP, or WebP. EXIF orientation is honored and transparency is composited onto black.
2. Choose **RGB** or **Black & white**. Grayscale uses luminance and one physical R1 R source channel; it retains gray levels rather than thresholding pixels to binary.
3. Enter **Barcode bits** (2–48), apply it, and load/edit the matching schema. Choose the encoding scheme, tile count, and rectangular grid. Current production choices are tile layouts; older saved shift layouts remain readable.
4. Choose active analog rows/sites and permanent fiducials. Reserved fiducial sites cannot simultaneously be active analog pixels. Reduced canvases, including the five-column image canvas, are supported.
5. Choose **Fit with black padding**, **Center crop**, or **Stretch**. Use **Best fit** or the origami 90° rotation controls. The source image stays upright while the origami geometry rotates beneath it.
6. Set independent RGB synthesis inputs: base stock/layout, final volume, base/scaffold targets, per-color site targets, **Biotin target**, and transfer minimum/increment. Expand **Sources & mixing…** to inspect source maps/stocks, hinges, buffers, output root, and plate type. Set the experiment/pool and destination capacity.
7. Inspect the binned/quantized image and predicted Gaussian image. Configure simulation and optimization as desired. Press Enter to apply alignment-field edits.
8. Use **Inspect design** to check barcode bits, analog-site values, achievable fractions, and volumes. Set the starting well/assignment and click **Prepare N runs**.
9. Review **Prepared picklist** and **Prepared mixing recipe** for individual designs, including the collected staple volume per destination well. Click a transfer/site row to highlight its mapped physical location.
10. Use **Save image batch…** for the complete export, **Send N runs to storage**, or **Save mixing recipe batch…** for recipes only.

Changing relevant image/schema/synthesis settings invalidates prepared runs so exports cannot use stale assignments. RGB settings are saved independently from manual settings, while both workflows share destination usage and stored runs.

**Capacity:** each RGB tile needs three designs; each grayscale tile needs one. Joint RGB encoding can use up to `floor(2^bits / 3)` positions within the 128-position/384-design production limits; fixed-color layouts reserve color bits and may hold fewer. Grayscale is limited by code space and 384 designs. **Redundancy (unique barcodes per location)** creates multiple grayscale designs with the same image sites but distinct codes. Redundancy consumes design capacity; previews average copies to preserve brightness.

**Image physics:** linear-light processing is enabled by default, so sRGB 128 corresponds to roughly 21.6% intensity rather than 50%. Intensities are quantized to fractions feasible with the selected stocks and droplet constraints. Barcode sites remain 0%/100%. The full six-column analog field has 9.9 nm horizontal and 5.1 nm vertical site pitch; active-canvas/layout changes alter its dimensions, so use the exported manifest for the specific experiment.

**Mandatory RGB biotin:** every newly prepared image design includes all eight D-Biotin staples, by default from 20 µM stock at `SourcePlate8[8]`, P17–P24. Their independent target is separate from the base and RGB targets, and their original competitors are omitted. Confirm those wells are loaded using the [D-Biotin map](picklist_generator/replacement_sheets/book_base_20uM_interleaved.README.md#d-biotin-loading-map). Previously saved picklists do not update automatically.

Automatic PAINT alignment-template export is enabled by default when preparing runs. **Open alignment templates** opens `rgb_alignment_templates/<batch ID>/`, containing per-design barcode PNGs/sidecars, shared fiducial templates when defined, schema, and a mapping manifest. Analog image pixels are excluded from barcode templates. Export the reusable bit schema separately for PAINT Step 3.

A full image batch includes the combined Echo picklist, per-design picklists/recipes, `pixel_assignments.csv`, `image_manifest.json`, binned/quantized previews, Gaussian reconstruction PNG/NPZ, and applicable alignment-simulation outputs. Keep the entire folder as the design record. The image manifest stores geometry, codebook, orientation, site fractions, rendering settings, and destination assignments.

### Simulation, optimization, and scheme comparisons

| Feature | How to use it and what it means |
| --- | --- |
| Gaussian overlay | Set FWHM and exposure. The renderer sums unit-integral circular Gaussian site signals in physical coordinates; wider spots spread signal without increasing its integral. Barcode sites are excluded. |
| Brightness normalization | Linear-light processing, equal total signal across schemes, and larger-canvas exposure compensation make selected comparisons consistent. Raw exported density retains values beyond display clipping. |
| Residual alignment simulation | Enter X/Y translation SD, angle SD, alignment center, and observed origami count. Compare ideal alignment, expected image, and a finite-count example; **New alignment example** changes the realization. All sites on one origami share one rigid error. |
| R1/body competition errors | **Simulate stoichiometry errors** models ratio bias, source variation shared across designs, and independent transfer variation. **New brightness example** changes the sample. This is preview-only and does not change picklists or optimization assignments. |
| Replacement optimization | **Optimize replacement percentages for Gaussian overlap** fits the predicted image to the fitted source image, then quantizes/refines on feasible transfer levels. Barcode bits stay fixed. Exports use the active optimized assignments. |
| Regularization | Strength penalizes departure from direct assignments. With alignment simulation, the objective also includes finite-count alignment variance and an optional extra variability penalty. |
| Original-scheme comparison | **Compare original schemes…** shows comparable previews and **Use this scheme** applies a choice. Refresh stale comparisons after settings change. |
| Tiling/FWHM comparison | **Compare tiling & FWHM…** explores supported position budgets, tile/shift allocations, orientations, and blur widths, including legacy comparison families. |
| Exhaustive PDF | **Export exhaustive PDF…** creates the broader encoding/layout comparison report. |
| Similarity graph | **Export similarity graph…** ranks source-image similarity under the selected assumptions and saves comparison outputs. |
| Robustness search | **Run similarity optimization** ranks schemes and stress-tests the best 20 over X/Y/angle/FWHM ranges. It selects the lowest worst-case source-image RMSE, with mean RMSE as a tie-breaker, and exports PDF/PNG/CSV/JSON results. |

The robustness search keeps each finalist's optimized assignments fixed during its uncertainty sweep; it does not reoptimize every condition or replace the active prepared design. Long comparison jobs report progress and support cancellation where offered.

These are forward models and local numerical optimizations. Equal design abundance/channel brightness, specified alignment distributions, finite-count sampling assumptions, and the chosen Gaussian width determine the prediction. Source RMSE and held-out simulation metrics do not establish experimental fidelity; systematic drift, deformation, binding kinetics, and unmodeled acquisition effects require separate validation.

### Split–pool and digital encoding research

Use **RGB Image → Split–pool encoding analysis** to analyze the current encoded image, imported CSV/NumPy targets, or synthetic datasets. Configure the model, library size K, optimization budget, seeds, and physical transfer settings as applicable; run the fit and open **Visualize analysis**.

The toolkit includes:

- **PCA references**, bounded additive models, irreversible **ON-only**, and competitive **ON/OFF** incorporation models, with assignment optimization and reconstruction/readout-error comparisons.
- Learned shared **K-vector libraries**, PCA-guided initialization, K sweeps, extra rounds, and discrete 25 nL physical recipe refinement.
- **Independent-well synthesis** comparisons alongside shared-population split–pool models. RGB split–pool synthesis uses one global address assignment and shared library across colors; color selectors choose detail views.
- Vector canvases, target/reconstruction heatmaps, learning curves, sampling-error plots, **Canvas construction**, **Branch walkthrough**, **Well trajectory**, **Transfer recipe**, and **Full synthesis** views.
- **Image Monte Carlo** for finite-count image examples. PCA sampling is explicitly a hypothetical clipped reference, not a realizable synthesis model.
- Exactly representable synthetic benchmarks, repeat/restart reports, parameter/assignment exports, and validation against known generated cases.
- Separate CLI tools for reversible GF(2) digital mappings, flexible bulk digital-writing schedules, and reversible-encoding/bulk-writing comparisons.

Run the CLI from `picklist_generator` in its Python environment:

```sh
python -m pip install -r split_pool_encoding/requirements.txt
python -m split_pool_encoding --quick --bits 3 --output split_pool_encoding/results/my_quick_run
python -m split_pool_encoding --input my_targets.npy --output split_pool_encoding/results/my_targets
python -m split_pool_encoding --config split_pool_encoding/config.json --output split_pool_encoding/results/my_experiment
python -m split_pool_encoding.digital_affine --demo --output split_pool_encoding/results/my_affine_demo
python -m split_pool_encoding.digital_bulk --demo --search --output split_pool_encoding/results/my_bulk_demo
```

Choose new output directories for separate experiments. The main imported targets are headerless CSV or NumPy matrices. `--quick` is a smoke-test budget, not evidence of global optimality. Detailed model/configuration/export instructions, including the reversible bulk tool, are in the [research manual](picklist_generator/split_pool_encoding/README.md).

Research transfer estimates and logical branch diagrams are not instrument-ready picklists. The models do not fully specify washing, recovery, carryover, pooling/resplitting volumes, or calibrated incorporation kinetics. A successful numerical fit or digital round trip does not by itself demonstrate a practical synthesis protocol.

## Files, exports, and persistent state

| File/location | Purpose and handling |
| --- | --- |
| Localization `.csv`, `.h5`, `.hdf5` + YAML | Experimental inputs. Retain original data and calibration metadata. |
| Template PNG + same-name JSON | Calibrated alignment/class patterns. Preserve metadata and physical-site definitions. |
| Logical-bit schema JSON | Shared group definitions for PAINT Step 3 and template editing. Example: [`logical_bit_schema.example.json`](picklist_generator/logical_bit_schema.example.json). |
| RGB barcode schema JSON | Encoding/fiducial mapping. Examples: [`barcode_schema.example.json`](picklist_generator/barcode_schema.example.json), [`barcode_schema.fiducials.example.json`](picklist_generator/barcode_schema.fiducials.example.json). |
| `image_manifest.json` | Full image design/reconstruction mapping. Retain it with the saved image batch. |
| `.paintanalysis` | Portable saved analysis with embedded source and results. Tiled `run.paintanalysis` also needs its tile folder for completed tile results. |
| `.paintrecon` | Compact reconstruction-only session with fixed extraction region and embedded manifest. |
| PAINT application state directory | Private environment, recent paths, development cache, and diagnostics; relocated by `PAINT_ANALYSIS_HOME`. |
| `.analysis-cache`, `.combined-cache` | Disk-backed analysis/aggregation arrays. Do not remove while in use. |
| `picklist_generator/config.json` | Local manual/RGB settings and recent/pending output information. |
| `destination_plate_state.json` | Destination-well usage and labels. |
| `storage_state.json` | Stored named runs and metadata. |
| `barcode_registry.json` | Experiment-scoped barcode reservations and release history. |
| `picklist_generator/generated_output/` | Default output root; configurable. Individual saves generally create timestamped folders, while combined storage exports use the root directly. |

Picklist CSV columns are `Source Plate Name`, `Source Plate Type`, `Source Well`, `Sample Comments`, `Destination Plate Name`, `Destination Well`, and `Transfer Volume` (**nL**). Mixing recipes report reagent volumes in **µL**. Source inventories map logical replacement keys to actual source wells; do not substitute a logical key for a physical address.

Back up the local picklist JSON state along with exported experiments if you need to preserve reservations and plate history. These state files and environments are ignored by the corresponding application `.gitignore` rules; example JSON files provide clean starting defaults. Review `git status` before committing generated data: not every output/session format is ignored everywhere in the repository.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Launcher fails or window never opens | Run from a terminal to see the error. Check Python version, `python -m tkinter`, and install that app's requirements into the interpreter actually used by its launcher. |
| Picklist reports missing SciPy after an update | Reinstall `requirements.txt` in `picklist_generator/.venv`; the launcher checks only a subset of dependencies before deciding to install. |
| Unexpected physical scale | Check YAML `Pixelsize`, source coordinate units, and template calibration. The localization fallback is 130 nm/pixel. |
| Drift-file rejection | Confirm the same acquisition/frame numbering, required frame coverage, and supported coordinate units. |
| Too few or merged candidates | Inspect Step 1 density/components, density cutoff, recovery/Signal gap distance, minimum point count, and fiducial geometry before changing later classification thresholds. |
| Good-looking fits are excluded | Inspect saved failure reasons, both-end fiducial support, overlap, correlation, optional corner gates, and dark/bright checks. Display overlays do not disable gates. |
| Many unclassified particles | Confirm group IDs/geometry and complete class templates, then inspect exact unmatched patterns and saved evidence. Use threshold sweeps for diagnosis before rerunning measurements. |
| Reconstruction looks mirrored or misplaced | Load the correct image batch manifest, verify matched codes, preview the image rectangle, and distinguish identification mirrors from section/final display transforms. |
| Magenta or missing occupancy regions | Inspect valid-count coverage, missing designs, uncertain sites, and exclusions. Missing measurements are not OFF observations. |
| Large analysis stalls or exits | Check progress stage, diagnostics, RAM and disk space, including the checkpoint/cache drive. Use a smaller ROI/distributed sample or reconstruction-only loading; resume saved tiles instead of restarting them. |
| Manual settings remain disabled | Make a nonempty compatible selection and click **Make Picklist** again after selection/source changes. |
| No feasible percentages or rejected transfers | Check actual stocks, final volume, site target, minimum/increment, and both competitor/replacement transfers. Change synthesis settings rather than rounding instrument transfers by hand. |
| Mixing recipe cannot be solved | Check buffer carried by every stock, DNA/scaffold targets, pooled volume, and requested final volume. Carried salt already above target cannot be removed by adding water at a fixed final volume. |
| Not enough destination wells | Inspect usage/reservations and per-design collected volume; a design may span wells. Use the correct new plate identity or deliberately reset a physically fresh plate. |
| Barcode already used or schema incompatible | Review the shared experiment registry. New destination wells do not create new barcode identities. Import earlier usage or release a reservation only when reuse is intended. |
| Preview/export became unavailable after edits | Prepare the batch again; relevant changes intentionally invalidate earlier prepared transfers and metadata. |

## Repository layout and development checks

```text
mona_lisa/
├── README.md
├── paint_analysis/
│   ├── paint_analysis_gui.py       # Desktop entry point
│   ├── origami_analysis.py         # Detection, alignment, site evidence, classification
│   ├── origami_composite.py        # Localization-density reconstruction
│   ├── binary_occupancy.py         # Molecular site votes and quantitative summaries
│   ├── occupancy_rendering.py      # Gaussian rendering of measured occupancy
│   ├── analysis_session.py         # Saved analyses
│   ├── composite_session.py        # Compact reconstruction sessions
│   ├── tile_checkpoint.py          # Recovery/checkpoint support
│   ├── tiled_aggregation.py        # Disk-backed combined results
│   ├── requirements.txt
│   └── tests/
└── picklist_generator/
    ├── picklist_app.py             # Desktop entry point and manual workflows
    ├── picklist_core.py            # Transfers and mixing calculations
    ├── rgb_image_view.py           # Image workflow and design inspection
    ├── image_encoding.py           # Image layouts and design assignments
    ├── template_generator.py       # Calibrated templates and group schemas
    ├── barcode_registry.py         # Pool-scoped code reservations
    ├── gaussian_*.py               # Image prediction and optimization
    ├── split_pool_encoding/        # Research package and CLI
    ├── replacement_sheets/         # Source inventories and replacement maps
    ├── requirements.txt
    └── tests/
```

Run tests separately in each application's environment and directory; modules use local imports, so a single root-level test invocation is not the documented workflow.

```sh
# In paint_analysis, with its environment active:
python -m pip install pytest
python -m pytest tests -q
python benchmark_classification.py --modes reference cpu auto --output .gpu-results/cpu-benchmark.json
```

```sh
# In picklist_generator, with its environment active:
python -m pip install pytest
python -m pytest tests -q
python -m pytest split_pool_encoding/tests -q
```

PAINT's synthetic benchmark compares numerical paths and stage timings without experimental data. GPU tests normally skip when CUDA is unavailable; `PAINT_REQUIRE_GPU_TESTS=1` makes GPU regression checks strict. The optional Windows GUI regression uses `PAINT_TEST_WINDOWS_GUI=1`. These checks supplement, rather than replace, experimental calibration and inspection.

## Detailed references

- [PAINT Analysis manual](paint_analysis/README.md): detailed controls, algorithms, saved sessions, tiled recovery, QC, reconstruction, and occupancy exports.
- [Picklist Generator manual](picklist_generator/README.md): source maps, recipe behavior, encoding families, simulations, reports, and registry operations.
- [Split–pool encoding manual](picklist_generator/split_pool_encoding/README.md): model definitions, configuration, CLI tools, datasets, physical recipe models, and limits.
- [Split–pool technical report](picklist_generator/split_pool_encoding/technical_report.md): benchmark interpretation and mathematical context.
- [Interleaved base plate and biotin loading map](picklist_generator/replacement_sheets/book_base_20uM_interleaved.README.md): physical inventory and logical replacement mapping.

The detailed manuals include historical feature notes; current UI labels, loaded calibration, and saved experiment manifests determine the settings for a particular run.
