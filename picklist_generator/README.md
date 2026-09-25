# Echo Picklist Generator

**Currently enabled replacement tabs: PAINT P1, PAINT R1, R1 R, R1 B, and R1 G.** Other replacement definitions and sheets are retained for later use.
The R1 R, R1 B, and R1 G tabs use the same extensions and source plate as PAINT R1,
with independent selections and red, blue, and green percentage-scaled highlights.

A standalone desktop version of the GUI and picklist workflow from
`picklist_generator_replacement-Copy1.ipynb`. Yaritza Extensions, Aptamers,
MB, PAINT P1, PAINT R1, and the base source sheet are loaded
automatically with portable, relative paths. The
original Set B CSV remains archived in `replacement_sheets/` but is not loaded
or shown by the application.

## Start the app

**Soyeon MB** in the Replacements tab includes 60 D-MB extensions (rows H03,
H05, H07, H09, and H11) and a separate panel for the six DL extensions supplied
in `DNA Book Sub-group (Myoungseok) - Soyeon MB.csv`. The bundled normalized
sheet is `replacement_sheets/soyeon_MB_replace.csv`; original names, sequences,
and source wells are preserved. Its source plate is labeled `SoyeonMB[1]` in
the picklist. Targets B21, B22, and B20 are translated to logical replacement
keys B11, B14, and L14, respectively, to account for relocated base staples.
These selections participate in the existing clash checks, previews, and storage.

Each replacement tab has an **Excess Multiplier**, initially **1**, also editable
in **Run & Mixing Settings**. Values are independent per tab and stay synchronized
between the two locations. For example, PAINT R1 at **5** with a **25 nL** transfer
volume produces **125 nL** transfers for its selected replacements. Base staples
at unselected sites keep the normal transfer volume. Positive decimal multipliers
are supported when the resulting volume is a multiple of 25 nL.
Destination capacity, available volume, and the mixing recipe account for the
scaled transfers; the desired staple concentration refers to the nominal 1× staples.

Each tab has an editable **Base transfer volume (nL, all tabs)** field, shared
with **Run & Mixing Settings**. Set the base volume independently of each tab’s
excess multiplier; 25 nL is the transfer increment, not a fixed base volume.
Each tab also has a **Replacement percentage** dropdown, defaulting to **100%**.
The selected percentage splits the total volume (transfer volume × excess
multiplier) between the replacement and original staple. For example, 100 nL
at 25% transfers 25 nL replacement and 75 nL original. Only splits with both
volumes in 25 nL increments are offered; zero-volume transfers are omitted.
Changing volume or excess refreshes the choices. If a previous percentage is
no longer valid, choose a new one before generating. Cyan highlight opacity
matches the percentage.

**miRNA** loads `replacement_sheets/miRNA_replace.csv` from the supplied Source [3]
anchor sheet, using `Well` as the source well and `Replace` as the base replacement
key. The source plate is `SourcePlate3[3]`. Its two 8×12 panels infer sites from
the name suffix: `p1_H3-3` is H03, column 03, and `p2_H3-1` is H03, column 01.
Both panels contain anchors at H03, H05, H07, H09, H11, and H13, with columns
displayed 1 through 12. Names, sequences, wells, and replacement targets are
preserved from the CSV. Toehold Lock and miR-21-DNA lack replacement targets
and are excluded. Both H03 anchors target M13, so selecting both is a clash.

In **Replacements**, choose **Hinge type**: **Flexible hinges (original
wells)** (the default) or **Rigid center connectors (F19–H24)**. Each option includes 18 hinges and
excludes the other set. The choice is remembered and recorded with stored runs.
The base CSV uses `Hinge Type` and `Original Well Position` to associate the
alternatives with the same replacement sites, so replacements work with either
hinge type. Staple counts and mixing volumes use only the selected set.

The source plate map also shows six no-center-connector alternatives at
J19–J24, targeting J16, I16, C02, D02, L16, and K16 respectively. These source
wells and sequences are absent from the bundled `book_base.csv`, so this
option is not yet available in the picker.

Python 3.9 or newer is required. A standard Python installer from
[python.org](https://www.python.org/downloads/) includes Tkinter, which the app
uses for its desktop interface.

- macOS: double-click `launch.command`. If macOS blocks the first launch,
  Control-click it, choose **Open**, then confirm.
- Windows: double-click `launch.bat`.
- Linux: run `./launch.sh`. If Tkinter is not installed, install your
  distribution's `python3-tk` package first.
- Terminal: run `python3 picklist_app.py` from this folder.

The launchers create a local `.venv` on first use. There are no third-party
Python packages to download. Generated files default to `generated_output/`.
Every successful generation creates a new date-and-time folder, for example
`generated_output/2026-08-10_14-32-07/`, containing that run's picklist and
mixing recipe. A numeric suffix is added if two runs occur in the same second.

On first launch, the app creates two local JSON files automatically from the
bundled `.example.json` defaults:

- `config.json` stores the most recent output paths, the wells used by the last
  successful run, the pending wells for the next run, output root, destination
  plate name, transfer settings, and last successful run.
- `destination_plate_state.json` records transfer count, volume, and last-use
  time for every destination well used by a successful generation.
- `storage_state.json` stores each named run's picklist, mixing-recipe rows, and
  replacement-panel selection metadata in the persistent Storage queue.

At startup, the most recent output paths are restored. The destination-well
field defaults to the first unused wells on the current destination plate,
using the same number of wells as the most recent entry.

Immediately after successful generation, the destination plate state records
the transfer count and volume delivered to each used well. The destination-well
field then advances to the next unused wells, and both the completed and pending
well lists are saved in `config.json`.

Use **Destination starting well** in Run & Mixing Settings to override where
automatic destination allocation begins. For example, choosing C01 skips open
wells before C01 and fills from unused wells at or after C01. A starting well
already recorded as used is rejected. The override does not change normal
unused-well behavior on the next launch. Use **Start fresh plate** on the
Destination Plate tab to clear all recorded usage for the currently named
plate after confirmation. Existing output folders are retained.

## Workflow

**Run & Mixing Settings** starts greyed out. Select replacements and click the
large green **Make Picklist** button at the bottom of Replacements to check for
clashes and open the settings. Empty or conflicting selections keep settings
locked. Changing selections or a replacement CSV locks settings again until
you click **Make Picklist** and pass the check.

1. Open **Replacements** and select wells in Yaritza Extensions, Aptamers, MB,
   PAINT P1, or PAINT R1. Grey wells do not have a corresponding
   row in the selected replacement CSV.
2. To generate an identification template from a replacement design, open its
   panel and click **Make template from this panel**. This transfers the exact
   selected-position pattern to **Origami Templates**. The editor starts with
   no predefined bit groups. Select any physical sites, enter a name, and create
   either a digital-bit group or an always-on alignment group. Repeat this for
   any encoding scheme; a digital bit may contain any number or arrangement of
   physical sites, so one missing extension need not become a complete bit
   error. Adjacent replacement
   columns use a uniform 10.909 nm pitch and adjacent displayed rows use a
   uniform 5.0 nm pitch. This makes the 12×8 outer-site span 120×35 nm.
   Up-extension panels use the mirrored physical
   orientation (left-to-right columns 12 through 1), while down-extension
   panels use columns 1 through 12. Both the interactive replacement sheets
   and their exported PDF reports use this physical pitch. The default 8×12
   grid and 20 nm image margin therefore produce a 160×80 nm
   raster footprint. Choose which digital groups are ON, then click **Save
   template PNG**. The PNG colors every rendered group differently for visual
   inspection; Paint Analysis removes those annotation colors before alignment.
   Upload the resulting bright-on-dark PNG in the paint-analysis Origami identification panel. The
   PNG embeds its x/y nanometres-per-pixel calibration, and a same-name JSON
   sidecar also records the physical dimensions, alignment strokes, logical-bit
   definitions and states, and expanded physical site list; it is documentation
   and does not need to be uploaded because the same metadata is embedded in the PNG.
3. Click **Check selections** at the bottom of the Replacements tab. Any
   selections targeting the same base well turn red with an × and are listed
   in the compatibility box. Use **Save selections PDF** to export one PDF
   report containing selected replacements from every used panel, including
   panels that are not currently visible. The PDF is saved automatically in a
   new timestamped folder beneath the configured Generated output root.
4. Open **Run & Mixing Settings**. Confirm destination wells, volumes, plate
   names, and mixing inputs.
5. Click **Generate Preview**. This calculates the picklist and mixing recipe
   without writing files, recording destination usage, or advancing wells.
6. Open **Destination Plate** to see the 16×24 plate map. Previously used wells,
   wells at capacity, and the wells currently entered in Settings have distinct
   colors. Click any well for its recorded transfer and volume details. Edit its
   **Label** and **Used** checkbox, then click **Save well**. Uncheck **Used** to
   clear that well's recorded transfers and make it available again (for example,
   after deleting an unused picklist). Labels are saved per plate; a blank label
   shows the well address. Long labels are shortened on the map and shown in full
   in the editor. Labels do not change well addresses in generated picklists.
7. Review the previews under **Results**. Choose **Save picklist + recipe** to
   write the individual CSV files, or choose **Send to storage** and enter a run
   name without writing those individual files. Either action records the
   destination usage and advances the wells once. You can subsequently use the
   other action on the same preview without recording the transfers twice.
8. In **Storage**, select individual runs (Command/Ctrl-click for multiple), or
   click **Select all**, then click **Generate combined picklist + recipe**. The
   transfer rows are concatenated into one large picklist, while each run keeps
   its own mixing-recipe CSV with the run name in its filename. Volumes are not
   summed across runs. These files are written directly to **Generated output
   root** in Settings, alongside selections PDFs. Re-exporting replaces files
   with matching names in that folder.
   Runs that reuse the same destination plate and well cannot be combined.
9. Use **Generate selections PDF** in Storage to create one large PDF for the
   selected runs. Every named run starts on a new page and includes all of its
   selected replacement panels. **Select all** includes every stored run. The
   PDF is saved automatically in **Generated output root**, in the same folder
   as combined picklists and mixing recipes. **Save selections PDF** in
   Replacements also uses this folder.
10. Individual CSV paths are shown in Settings; after a combined generation,
   the Storage tab reports its output folder and a confirmation shows the
    full path.

### Custom logical-bit encodings

The Origami Templates tab includes a visual group editor plus **Load bit schema
JSON** and **Save bit schema JSON**. There are no default groups. A schema may
use any physical grid dimensions, number of bits, bit names, and site
groupings. Physical coordinates are one-based `[row, column]` pairs.

```json
{
  "format": "paint-analysis-logical-bits-v1",
  "physical_rows": 4,
  "physical_columns": 6,
  "alignment_groups": [
    {"id": "anchor", "physical_sites": [[1, 1], [1, 2], [2, 1]]}
  ],
  "logical_bits": [
    {
      "id": "bit_a",
      "label": "Bit A",
      "physical_sites": [[2, 3], [2, 4], [2, 5]],
      "evidence_sites": [[2, 3], [2, 4], [2, 5]]
    }
  ]
}
```

`physical_sites` controls what is rendered when a bit is on.
`evidence_sites` controls which of those sites Paint Analysis aggregates when
measuring the bit and may omit ambiguous intersections. If omitted, it defaults
to `physical_sites`. Alignment groups are always rendered and never classified.
Physical sites may belong to more than one group. Their PNG colors blend, and
their physical state follows union/OR semantics: the site is bright when any
active group contains it. Paint Analysis does not use a shared bright site as
negative evidence against an overlapping group that is OFF.
Selecting a group in the table loads its sites into the physical-grid editor.
Use **Toggle bit ON/OFF** to choose whether a digital group is present in the
current template; alignment groups are always present. Saved template PNGs add
`active_logical_bits` and stable per-group display colors to the embedded copy
of the schema.

Each replacement row identifies a base `Replace Well`. The generator removes
that base well and substitutes the replacement sheet's source well. Selecting
two replacements for the same base well is rejected. Transfers are assigned to
destination wells in order, up to the configured volume capacity.

The mixing fields retain the notebook's comma-separated vector format. The
water balance fixes a notebook arithmetic omission by subtracting sodium along
with the other reagent volumes, so the recipe sums to the requested final
volume.

## Repository contents

- `picklist_app.py` — Tkinter desktop interface
- `picklist_core.py` — CSV parsing, replacement, transfer, and mixing logic
- `selection_report.py` — dependency-free PDF selection-report renderer
- `template_generator.py` — dependency-free grayscale PNG barcode-template renderer
- `logical_bit_schema.example.json` — editable arbitrary logical-bit schema example
- `app_state.py` — persistent configuration and destination usage helpers
- `config.example.json` — clean first-run configuration defaults
- `destination_plate_state.example.json` — clean, unused plate-state example
- `storage_state.example.json` — clean persistent-storage example
- `config.json` — local user defaults and recent-run metadata, created at runtime
- `destination_plate_state.json` — local 384-well state, created at runtime
- `storage_state.json` — local stored-run queue, created at runtime
- `replacement_sheets/` — all bundled source/replacement CSV files
- `launch.command` / `launch.bat` / `launch.sh` — platform launchers
- `tests/` — logic regression tests

The `.venv/`, generated CSVs, `config.json`, `destination_plate_state.json`,
and `storage_state.json` are intentionally ignored by Git. This keeps
each user's paths and plate history private. Commit the example JSON files so a
fresh clone always starts with an unused destination plate.

### Template image orientation

The Origami Templates editor, preview, and new PNG exports display columns
C12 → C1, matching PAINT R1 replacement ordering. Column/site IDs and logical
schemas retain their canonical numbering; changing the display does not rename
sites or bits. PNG embedded metadata and the JSON sidecar both record
`image_mirrored_x: true`. The renderer can also export canonical images with
`image_mirrored_x: false`.

Paint Analysis uses this flag to avoid reflecting an already mirrored image.
Legacy calibrated images without the flag retain the previous import behavior
(horizontal reflection). Logical sites and column offsets remain canonical in
both export orientations and are transformed independently by analysis.

### Color and shift encoding

R1 R, R1 B, and R1 G have X-shift and Y-shift dropdowns. States 0, 1, 2,
and 3 map to off/off, off/on, on/off, and on/on respectively. X uses bits
3–4; Y uses bits 5–6. These state numbers do not yet specify physical shifts
or change PAINT analysis.

Bits 1–3 occupy H01/H03/H05 at C11, C09, and C07. Bits 4–6 occupy
H11/H13/H15 at C11, C09, and C07. Bits 1–2 are fixed by the color tab:
red is on/on, blue is on/off, green is off/off. Encoded sites cannot be
toggled individually; image sites remain independently selectable.

Enable **Include this color in run** to include a color design. Selecting an
image site or choosing a shift also enables it. Unused color tabs start disabled.
**Clear set** disables that color and resets both shifts to state 0. The fixed
color-code highlights remain visible even when the color is excluded from the
run or cleared; red shows bits 1 and 2, blue shows bit 1, and green shows neither. Shift states
are included in the selection metadata saved with runs. Existing compatibility
checks still apply to overlapping replacement sites across enabled tabs.

### RGB images → selectable six-bit encoding

Open **7. RGB Image** and choose an **Encoding scheme**:

| Scheme | Image size | Pixel spacing (nm) | Designs | Default destination wells |
|---|---|---|---|---|
| 4×4 (original) | 24×32 | 2.475 × 1.275 | 48 | 144 |
| 6×3 (nearly square pixels) | 36×24 | 1.65 × 1.70 | 54 | 162 |
| 7×3 (maximum capacity) | 42×24 | 1.4143 × 1.70 | 63 | 189 |
| 5×4 | 30×32 | 1.98 × 1.275 | 60 | 180 |

The original scheme preserves the existing two color bits, two X bits, and two
Y bits. The other three schemes use all six bits jointly:
`code = channel_index * x_states * y_states + y_shift * x_states + x_shift`,
with R=0, G=1, B=2 and bit 1 most significant. Each uses codes 0 through
`design_count - 1`; the remaining six-bit codes are reserved. For example,
7×3 uses codes 0–62 and reserves 63. The physical locations of all six bit
groups stay unchanged. In joint schemes, individual bits do not have fixed
color or axis meanings.

Switching schemes re-bins the original image, updates pixel spacing, previews,
pixel outlines, and batch size, and invalidates any prepared batch. The selected
scheme is remembered across application launches. The manual color tabs retain
their original encoding; this selector applies to the RGB image workflow.
Exports and stored designs include the scheme ID and complete decoding table.

Choose **Upload RGB image** and a fitting mode. Click **Compare original schemes**
to see all four options side by side, with each binned image above its predicted
Gaussian RGB reconstruction. Every preview uses the same physical field and display scale;
the cards list image resolution, design count, and pixel spacing. The comparison
updates with image, rotation, fit, base volume, and excess multiplier changes.
Click **Use this scheme** on a card to apply it. Opening the comparison does not
prepare runs or change an existing prepared batch.
PNG, JPEG, TIFF, BMP, and WebP are supported. Import honors EXIF orientation
and composites transparency onto black. Use **Origami left 90°**, **Origami right
90°**, or the **Origami rotation** dropdown (0°/90°/180°/270° clockwise) to rotate
the origami under an upright image. The source image is never rotated beyond
its EXIF orientation. Quarter turns swap the field to 40.8 × 59.4 nm and swap
the displayed pixel dimensions and pitches. The upright image is fitted and
binned into this rotated field, then mapped back to the origami's staple sites.

The barcode diagram, row/column labels, affected-pixel outlines, and comparison
previews follow the rotated origami. Changing orientation refreshes previews
and requires preparing the batch again. Exports and storage explicitly record
`origami_rotation_degrees_clockwise` and a zero additional source-image rotation.
**Fit with black padding** preserves
all image content; **Center crop** fills the field without distortion; **Stretch**
uses the whole image but changes its physical aspect ratio.

The image field uses C6–C1 from left to right and H01–H15 from top to bottom.
Horizontal site pitch is **9.9 nm**, and vertical site pitch is **5.1 nm**.
All schemes cover **59.4 × 40.8 nm** when measured at pixel-cell boundaries.
Pixel pitch equals site pitch divided by the selected axis’s shift count. The
original 4×4 scheme has a center span of **56.925 × 39.525 nm**; other schemes
record their own center spans in the manifest. Previews preserve this
physical aspect ratio. Fitting occurs in physical space before area averaging.
The C7–C6 boundary gap is outside the image field and is retained in origami
coordinates; it is not added to the image pixel pitch.

For site row index `r`, column index `c` (C6 is index 0), and shift states `sx`,
`sy`, the image pixel is `(x_states*c + sx, y_states*r + sy)`. Positive shifts
point right and down. Each pixel appears exactly once in each of the three
color channels. Every design still contributes 48 image sites (6×8).

By default, sRGB values are decoded to linear-light intensity and rounded to the
nearest achievable replacement fraction. A display value of 128 corresponds to
about 21.6% intensity, not 50%. The number of levels depends on **base volume × that
color's excess multiplier**, divided by 25 nL. For example, 100 nL at 1× offers
0%, 25%, 50%, 75%, and 100%. Every image site uses that total volume, split
between original and replacement staples; zero-volume transfers are omitted.
This uses the standard sRGB transfer function, not a calibrated fluorescence
response model. Input RGB images are assumed to be sRGB. The manual tabs' uniform percentage is not used for imported
images.

Barcode sites use the same total volume but always encode 0% or 100%,
independently of image intensity. Both on and off barcode sites participate in
the split, so total volume is independent of the barcode state. All six bits
use the previously defined three-extension groups. Other base staples remain
at the base transfer volume. No additional alignment markers are introduced.

The **Predicted RGB — Gaussian overlay** is formed by summing image-site signals
from every encoded design after applying its XY shift and RGB pseudocolor. Each
site contributes a circular Gaussian in physical nanometer coordinates, with
integrated signal proportional to its quantized replacement fraction. Barcode
sites are excluded from this image-region reconstruction. It is not a mosaic
of colored pixels or a separately normalized image for each design.

**Gaussian FWHM (nm)** is adjustable and defaults to **7.5 nm**. FWHM is the full
width at half maximum; the corresponding sigma is FWHM / 2.35482. Gaussians have
unit integral before multiplying by replacement fraction, so widening the blur
spreads signal without increasing its total amount. Overlap adds signal and
mixes colors. The field is sampled at 0.15 nm or finer for very narrow Gaussians,
and tails outside the displayed image field are cropped.

**Exposure** uses the same linear gain across all colors and comparison schemes.
No image or channel is independently normalized to its brightest pixel. The
fixed white reference is the mean density of a fully populated original 4×4
sampling lattice at 100% replacement. **Equal total signal across encoding
schemes**, enabled by default, weights each design by 48 / design count. This
prevents denser encodings from becoming brighter merely because they contain
more designs. Disable it to retain equal per-design weight; the displayed clipping percentage
makes this visible. The raw exported density retains values beyond the PNG's
white level.

This is an expected-signal model with equal representation of every design,
equal channel brightness, and perfect alignment. The FWHM can approximate your
measured effective blur; the preview does not simulate stochastic binding,
background, localization counts, or alignment errors separately.

Inspect the binned image and Gaussian reconstruction side by side, then choose
any design to see its barcode and per-site RGB values,
replacement percentages, and original/replacement transfer volumes. The image
workflow creates independent runs rather than combining the three color tabs
into a single origami mixture. Manual designs remain available separately.

Use **Run & Mixing Settings** to configure the base plate, hinge type,
destination plate, capacity, repeat transfers, and recipe. **Prepare N runs**
allocates unused wells at or after the indicated starting well, skipping wells
already recorded or reserved by stored runs. A design can span multiple wells;
no well is shared between designs. At the default 50 nL base, 5× excess, and
12.5 µL capacity, the original 4×4 batch uses **144 wells** (see the table for other schemes). Preparation validates all
picklists and recipes before making any changes to storage or plate usage.
Changing image, fit, volumes, source sheets, or run settings invalidates a
prepared batch and requires preparation again.

**Save image batch** creates a new dated folder containing:

- A combined Echo picklist, plus one picklist and mixing recipe per design.
- `pixel_assignments.csv` with every pixel/channel/site/shift and its volumes.
- `image_manifest.json` with geometry, bit definitions, RGB arrays, settings,
  the selected scheme and full codebook, per-site percentages, and destination assignments.
- `gaussian_reconstruction.png`, the predicted composite in upright-image coordinates.
- `gaussian_reconstruction_density.npz`, the unclipped floating-point RGB density
  and raster pixel sizes in nm. The manifest records Gaussian width, exposure,
  shared intensity reference, coordinate origin, and model assumptions.
- Binned and quantized PNGs (per-site intensity references, not predicted appearance)
  at the chosen scheme’s image dimensions, plus `*_physical.png` previews displaying
  rectangular pixels at the correct aspect ratio. Use the manifest for nm units;
  the raw PNGs display square pixels in ordinary image viewers. All PNG previews
  keep the source image upright. Version 2 manifests keep JSON RGB arrays in
  origami-local coordinates and include separate upright-display dimensions.
  Pixel assignments include both local `pixel_x/y`, `x_nm/y_nm` and upright
  `display_pixel_x/y`, `display_x_nm/y_nm`; shifts and barcode definitions remain
  in origami coordinates. Display nm coordinates are relative to the upright
  image's upper-left pixel center.

**Send N runs to storage** stores each design separately with its geometry and
pixel metadata. Save and storage both record destination usage, once per batch.
PAINT analysis is not modified; exported shifts and nm coordinates provide the
mapping for its later reconstruction workflow.

Image import and Gaussian reconstruction require **Pillow and NumPy**
(`python -m pip install -r requirements.txt` in the application environment).
The supplied launchers install missing dependencies.

Enable **Optimize replacement percentages for Gaussian overlap** on the RGB Image
page to fit the predicted Gaussian composite to the upright source image. The
optimizer uses physical site spacing, encoded shifts, origami rotation, the
selected FWHM (default 7.5 nm), and the shared exposure. It first solves a bounded
continuous intensity problem, rounds to the permitted 25 nL increments, then
refines the discrete assignments. Original and replacement volumes still sum to
base volume × the color's excess multiplier; barcode bits remain unchanged.

The page shows the before and after Gaussian overlays and RGB root-mean-square
error. The objective uses unclipped RGB values at fixed exposure, so saturation
cannot hide excess signal. The best result is retained, including the original
assignment if no improvement is found. This is a local optimization and cannot
recover detail beyond the available sampling and Gaussian width. Its prediction
retains the equal-abundance, equal-channel-brightness assumptions of the preview.

With optimization enabled, **Compare original schemes** compares the optimized
reconstructions and reports each scheme's before/after error. Optimization runs
independently for every scheme, including the nine-bit and four-quadrant options.
The popup follows the RGB tab's image, fit, rotation, base transfer volume,
per-color excess multipliers, FWHM, exposure, linear-light processing, signal
normalization, canvas brightness compensation, and alignment settings. Its summary
shows whether optimization is enabled for all schemes, the concentration settings,
and the alignment parameters. Alignment previews show the expected result; the
random example seed only controls the separate example realization. Changes to
these image settings recalculate the assignments. Disable the option
to restore direct pixel assignment. Prepared runs, exported picklists, and stored
runs use the active assignments; the manifest records optimization settings and
error metrics. The target is the fitted source on the reconstruction raster,
using linear light by default; the reconstructed signal is encoded back to sRGB
for display and PNG export. Both Gaussian and alignment optimizers use the same
color and abundance settings as their previews.

### Residual alignment uncertainty

On the RGB Image page, enable **Simulate alignment uncertainty** to model imperfect
corner-fiducial alignment. Set the X and Y translation standard deviations (nm),
angular standard deviation (degrees), and observed origami count per design.
The initial 1 nm / 1 nm / 1° errors and 20 origami/design are illustrative inputs,
not measured instrument performance. Zero errors are valid. Independent,
zero-mean Gaussian errors are assumed; systematic bias, correlated corner-fitting
errors, and deformation are not modeled in this version.

One residual rigid transform moves all image sites on each origami together.
The decoded XY shift is added **after** that transform, followed by the display
rotation and pseudocoloring. X/Y error axes are the aligned origami axes. The
fiducial alignment center is editable in nm relative to the unshifted C6 H01
image site. Its default is the center of the full staple grid, including the
middle column gap: X = −9.95 nm, Y = 17.85 nm. Enter the actual alignment center
if your corner fiducials define a different one.

The alignment comparison shows ideal alignment, a sampled ensemble mean, and a
finite-count example. **New alignment example** generates another independent
realization without changing the assignments. Each design's signal is divided
by its origami count so changing count changes uncertainty rather than exposure.
These examples isolate alignment error: site intensities remain fractional;
random staple occupancy, photon noise, and differing design abundances are not
simulated. The alignment renderer uses a physical raster at most 0.75 nm per
sample (finer for narrower Gaussians), separately from the ideal preview's
0.15 nm raster. Very narrow widths that would exceed the alignment model's
memory limit are rejected with an explanatory message.

With replacement optimization also enabled, the objective is:

`error of the expected image + (1 + extra variability penalty) × alignment variance`

The variance includes the correlation between sites on the same origami and
scales inversely with the number of independent origami per design. A penalty of
zero already minimizes expected finite-count squared error; a positive penalty
adds a preference for consistency and may trade off mean image fidelity. Thirty-two
reproducible, antithetically paired Gaussian transforms estimate the moments.
Continuous bounded optimization and discrete 25 nL refinement preserve barcode
bits and total site volumes. The original assignment is retained if its training
objective is better. This is an approximate local optimization, not a guarantee
of improved performance for every realization or actual instrument.

An independent set of 32 transforms evaluates the result. The before/after
**held-out RMSE** is reported without being used to select the solution; it can
occasionally get worse. Scheme comparison uses the same uncertainty settings
and reports these validation results for each scheme. Settings, model seeds,
validation metrics, and the example seed are included in batch metadata and
stored runs. Exports include `ideal_alignment.png`, `alignment_example.png`,
its raw density NPZ, and the expected alignment reconstruction in the existing
`gaussian_reconstruction.png` / density NPZ files. Changing settings or requesting
a new example invalidates a previously prepared batch so its saved metadata and
preview remain consistent.

### Comparing nine-bit image schemes

The RGB Image scheme selector and **Compare original schemes** include the original four
single-field six-bit options, their four quadrant variants, and three nine-bit options. Scroll the comparison horizontally:

| Nine-bit scheme | Bits | Image sampling grid | Designs |
| --- | --- | --- | --- |
| Fixed 16×8 | 2 color + 4 X + 3 Y | 96×64 | 384 |
| Joint 18×9 | Joint color and XY code | 108×72 | 486 |
| Parity 8×8 | 2 color + 3 X + 3 Y + even parity | 48×64 | 192 |

Nine-bit layouts use two staples per bit: bits 1–3 are H01/H03 at C12/C10/C08,
bits 4–6 are H07/H09 at C11/C09/C07, and bits 7–9 are H13/H15 at C12/C10/C08.
Bit 9 in the parity scheme is chosen so the total number of on bits is even.
This detects every single-bit flip but does not correct it. The comparison does
not simulate barcode decoding errors or automatically allocate corner fiducials.
Fiducials must be reserved independently of switchable barcode sites.

Cards report bit count, staples per bit, parity, physical sampling pitch, and
minimum destination-well count. Alignment-enabled comparisons also report total
origami count at the selected count per design. Shared exposure and fixed total
design weight are the default brightness convention. More designs still imply
more observed origami at the chosen count per design; brightness weighting does
not change that observation budget or physically change mixture volumes. Compare clipping and optimized
errors accordingly. Finer sampling does not override Gaussian width or alignment
uncertainty. Nine-bit options work with Gaussian and alignment optimization;
large alignment problems avoid constructing a full quadratic-size Hessian.

All schemes can be previewed regardless of plate capacity. Batch preparation still
requires enough unused wells on the selected destination plate and rejects an
oversized batch before recording transfers. The 486-design scheme cannot fit on
one 384-well plate even at one well per design; other schemes can also exceed
capacity depending on transfer volumes. Multi-plate batch allocation is not added
by this comparison feature.


### Linear-light processing and comparable brightness

The RGB Image page enables **Linear-light image processing (sRGB input/output)**
and **Equal total signal across encoding schemes** by default. They are separate
controls and persist across launches. The first decodes sRGB before resampling,
alpha compositing against black, intensity assignment, and optimization. Signals
are added and errors are evaluated in linear light, then encoded back to sRGB
for previews and PNGs. Binned previews remain display RGB; quantized JSON arrays
remain per-site intensity × 255 and carry an explicit encoding label.

The second control uses a fixed scheme-dependent population weight of
`48 / number_of_designs` (equivalently `16 / shifts_per_color`). For 486 designs,
that weight is approximately 0.09877. This is an explicit equal-total-signal
comparison convention, not image-dependent auto-exposure. It does not change
barcode bits, transfer constraints, observed origami counts, or automatically
rescale physical mixing volumes. Disable it if equal amounts of every design
should contribute without normalization. Gaussian and alignment optimization,
validation, comparison, and export all use the selected convention.

Unclipped raw density NPZ files retain the sum of per-design signals, including
per-quadrant balancing weights when enabled for unequal-quadrant variants.
To obtain the modeled linear RGB signal, multiply density by the manifest's
`linear_rgb_gain`, which includes exposure, the selected population weight,
and any canvas-area exposure compensation.
Both settings and that gain are saved with each reconstruction. Legacy snapshots
without these settings continue to render using the previous direct-RGB,
equal-per-design convention.

### Four-quadrant composite with 48 types

Choose **4 quadrants — 24×32 pixels, 48 designs, 118.8×81.6 nm** in the RGB Image
scheme selector or **Compare original schemes**. This uses the existing six-bit,
three-staple-per-bit barcode layout: bits 1–2 specify color, bit 3 quadrant Y,
bit 4 quadrant X, bit 5 local X shift, and bit 6 local Y shift. There are
3 colors × 4 quadrants × 2 × 2 local shifts = 48 types; the unused color code
leaves 16 reserved codes. Each origami still has the same physical 6×8 image area.

Each quadrant samples 12×16 pixels with 4.95×2.55 nm spacing. Quadrants are placed
at offsets (0,0), (59.4,0), (0,40.8), and (59.4,40.8) nm, creating a 24×32 image
on a 118.8×81.6 nm virtual canvas. Placement is applied after residual rigid
alignment and local XY shift. Alignment rotates each physical origami around
its fiducial center; quadrant offsets do not introduce extra rotational lever
arms. Display rotation then rotates the complete composite, keeping the source
image upright. Design inspection outlines only the 48 samples contributed by
the selected design in its assigned quadrant.

Gaussian and alignment optimizers fit the complete composite, including signal
spillover across internal quadrant boundaries. Only the outer canvas boundary
crops Gaussian tails. Codebooks, manifests, design names, and assignment CSVs
record quadrant placement and local shifts; preparing/exporting the batch still
uses only 48 independent designs.

The comparison fits each full canvas to the same display width and lists actual
physical field dimensions. The same 7.5 nm Gaussian is half as wide relative to
the tiled canvas. At equal total signal and exposure, mean signal density is
approximately one quarter of the original field: tiling does not create photons.
**Match brightness for larger canvases** is enabled by default and applies 4×
exposure compensation to quadrants only. Leave shared exposure at 1× for a
brightness-matched comparison. Disable this option to inspect the uncompensated
signal density. The compensation is used consistently by reconstruction,
optimization, alignment validation, and export; it does not add photons or
change physical mixing volumes. The effective gain is recorded in the manifest
as `canvas_area_gain` and included in `linear_rgb_gain`. Raw density arrays stay
unmodified. Settings are saved, and older snapshots without this option retain
their uncompensated rendering.

### Quadrant variants under every six-bit family

**Compare original schemes** places each four-quadrant card directly below its
single-field six-bit parent. Scroll down to see the second row. These are
alternative codebooks, not four independent copies of the parent batch:

| Parent budget | Types | Local XY shifts: top-left / top-right / bottom-left / bottom-right |
| --- | ---: | --- |
| Original 4×4 | 48 | 2×2 / 2×2 / 2×2 / 2×2 |
| Joint 6×3 | 54 | 3×2 / 3×2 / 3×1 / 3×1 |
| Joint 7×3 | 63 | 4×2 / 3×2 / 4×1 / 3×1 |
| Joint 5×4 | 60 | 3×2 / 2×2 / 3×2 / 2×2 |

Every quadrant covers the same 59.4×40.8 nm physical region on the virtual
canvas, despite differing sample counts. Names refer to quadrants before display
rotation. The combined sample arrays retain their parent's dimensions, but
samples are not uniformly spaced in the joint quadrant variants. Import bins
each equal-area quadrant separately. Previews and physical PNGs draw the actual
cell sizes, and optimizers use actual site positions. Gaussian signals can
cross quadrant boundaries. There are no extra origami types: all codebooks
remain within 64 codes and preserve the parent's total type count.

The codebook records quadrant assignment, local shifts/counts, physical sample
centers and edges, and placement offsets. Offsets include a half-cell correction
when neighboring quadrants have different sampling pitches, so all quadrants
occupy equal physical areas without shifting the source-image boundaries.
Residual alignment transforms act on the physical origami before those offsets.
Raw sample-grid PNGs/JSON are index references; use physical PNGs or reconstructed
previews to see correct spatial proportions.

With **Equal total signal across encoding schemes** enabled, a per-design
`quadrant_normalization_weight` balances signal between quadrants with unequal
numbers of shifts. Its value is `(total shifts per color / 4) / shifts in this
quadrant`; weights sum to the original shift budget. This is reconstruction
weighting, not a change in synthesis count or mixing volumes. Alignment variance
uses the same weights. The separately controlled 4× canvas-area compensation
still applies to all quadrant schemes. Disabling signal normalization removes
quadrant balancing as well as the global design-count normalization.

For these unequal-quadrant variants, exported density arrays include the selected
per-quadrant weights; multiply by `linear_rgb_gain` for displayed linear signal.
The weights are recorded in the design and geometry metadata and assignment CSV.


### Compare XY shifts, tiles, and Gaussian widths

In **RGB Image**, choose **Compare tiling & FWHM…**. Rows are position-bit
allocations and columns are Gaussian FWHM values, initially **5, 7.5, 10 nm**.
Enter one to six comma-separated widths and click **Render comparison**.

The original fixed-color family reserves two bits for color and distributes the remaining four
bits among tile X, tile Y, local X shift, and local Y shift. There are **35**
allocations, each using **48 RGB origami types** and **768 RGB sample positions**.
The default filter shows the 16 allocations with a 24×32 sample grid;
**All allocations (427 layouts)** includes other grid aspect ratios. Filters for
each tile count and for each joint-code family isolate the tradeoffs. At one extreme all four bits
specify local shifts; at the other all four specify tile coordinates.
Each physical origami retains its original 6×8 image-site layout.

The fitted target shows tile boundaries. Previews preserve physical aspect
ratio in equally sized display boxes and include a 20 nm scale bar. Each row
lists field dimensions, sample spacing, sample counts, and the four bit budgets.
Each reconstruction reports Gaussian width relative to sample spacing,
clipping, area exposure compensation, and optimization error when enabled.
Larger mosaics spread the same signal over more area: brightness matching
applies exposure compensation equal to tile count (up to 21×), without adding
photons or modifying physical mixing volumes.

The matrix inherits the RGB tab's source image, fit, origami rotation, base
transfer volume, color excess multipliers, exposure, linear-light processing,
signal normalization, brightness compensation, optimization, and alignment
settings. Only Gaussian width is overridden by each column. Optimization fits
each layout and width independently; alignment-enabled previews show the
expected aligned result. Large alignment mosaics use a coarser raster while
keeping at least four raster samples per FWHM; numerical limits are reported
in individual cells. Progress is shown while rendering. To avoid repeatedly
starting a large calculation while editing, changed settings clear the matrix;
click **Render comparison** to update it.

All 427 sweep layouts also appear in the RGB Image **Encoding scheme** dropdown.
Choose one there, or click **Use this layout** beside a comparison row, to use it
for design inspection, run preparation, picklist export, and storage. Selecting a
layout uses the main tab's FWHM. Rendering the comparison alone does not change
the active assignments or Gaussian width. **Compare original schemes** retains
the original six-/nine-bit comparison; **Compare tiling & FWHM** shows the new
allocation matrix.

### Exhaustive PDF report

Click **Export exhaustive PDF…** in the RGB Image tab and choose a PDF filename.
All six panels carry the same white, black-bordered pixel outlines as the binned
RGB preview. Each page identifies the representative first design whose 48 nominal
pixels are outlined; the overlay follows the physical spacing, rotation, and crop.
These guides are display annotations and do not affect predictions or scores.
The report includes every encoding-scheme dropdown entry once, automatically
choosing **0° or 90° origami rotation** to best match the upright uploaded image's
aspect ratio. This minimizes aspect-ratio padding (or center-crop loss); ties use
0°. Image EXIF orientation is respected. Each layout gets a landscape page with **Binned RGB**,
**Before optimization — Gaussian overlay**, **Predicted RGB**, **Ideal alignment**,
**Expected alignment**, and **Finite count**. With the current 438 entries, this
is **438 view pages plus a settings cover page**. Equivalent layouts under different
names are retained so every dropdown entry is represented.

The report snapshots the current image and RGB settings, including the main
FWHM, fitting mode, base volume, color excess multipliers, exposure, color/brightness
options, optimization toggle, alignment settings, and finite-count seed. Each
chosen orientation is independently binned and optimized when optimization is enabled.
The source image remains upright. The multi-FWHM comparison's column widths do
not override the report's FWHM. With optimization off, before/after assignments
are identical and the report labels this explicitly.

Expected-alignment and finite-count panels always use the configured alignment
errors, even when alignment is disabled for the main prediction. Predicted RGB
uses the alignment toggle exactly as the main preview does. Ideal alignment uses
the final assignments without residual errors. Finite count simulates alignment
uncertainty only, not occupancy or photon noise. Pages include physical field
size, a 20 nm scale bar, clipping, exposure compensation, and optimization error.

Export runs in the background with a progress window and a Cancel button.
The progress bar shows percentage and completed view count, and advances through
each view's calculation and PDF-writing stages; the current stage is shown below it.
Cancellation takes effect between calculations. The report streams pages to a
temporary file and replaces the chosen destination only when complete; failure
or cancellation preserves any existing PDF. No runs are prepared, and the active
scheme, rotation, assignments, and main FWHM are unchanged.

### Regularizing Gaussian optimization

**Regularization strength** in the RGB Image tab controls how strongly optimized
replacement fractions stay near the direct, quantized image assignments. The
initial strength is **0.01**; **0** restores unregularized optimization. Increase
it to make corrections more conservative. This starting value is adjustable,
not a measured optimum for your experiment.

The objective adds `strength × mean((fraction − direct_fraction)²)` to the
image MSE (and the existing alignment-variance term when enabled). This is an
L2 penalty around the direct assignment, rather than a smoothing filter over
image edges. It discourages large compensating changes and extreme alternating
patterns while preserving the baseline image's edges. The penalty is normalized
by site count and the image loss by raster count. Barcode bits remain binary;
replacement and original volumes still use 25 nL increments and preserve their
sum. Reported RGB/held-out RMSE excludes the regularization penalty; optimization
metadata records the strength, assignment deviation, penalty, and total objective.

The optimization target remains the **uploaded image**, fitted/cropped/padded
according to the selected mode and resampled to the reconstruction raster in
linear light when enabled. The binned RGB sets the direct-assignment baseline
and regularization reference; it is not the optimization target.

The setting is saved and applies to ordinary and alignment-aware optimizers,
all comparison schemes/FWHM columns, prepared assignments, stored/exported
optimization metadata, and exhaustive PDF exports. Changing it recalculates the
active optimized image and invalidates prepared runs. The tiling/FWHM matrix
clears until **Render comparison** is clicked. Regularization constrains the
inverse problem; it does not change the alignment sample count or eliminate
uncertainty in the alignment-error model.


PDF panels preserve the complete fitted image at its natural physical aspect
ratio. They are not cropped to a fixed landscape rectangle. With **Fit with black
padding**, only the geometrically known fitting margins are trimmed; real black
image content is retained. Unused page area stays white, with no black frame.
All six panels share the same displayed physical region and magnification, and
scale bars sit outside the images. Headers list both the full field and the
shown region. Source-fitting settings are respected: **Center crop** still crops
at the encoding stage, so use **Fit with black padding** to retain the entire
uploaded image. Display trimming does not change optimization or reconstruction.


The main RGB Image tab uses the same best-fit orientation rule as the PDF:
uploading an image or selecting a different encoding layout chooses **0° or 90°**
to minimize aspect-ratio padding/cropping. A portrait image typically selects 90°
for the original layout. The source image stays upright. Manual rotation remains
available and is preserved when changing exposure, Gaussian width, optimization,
or color-processing settings. Click **Best fit** to restore the automatic choice.


While a PDF export is running, its button reads **Show PDF export progress…**;
click it to bring a hidden or minimized progress window back without starting
a second export. Startup and save-dialog errors are displayed in the app and
leave the button usable for retry. If files were updated while the application
was open, restart it so all modules use the same version.


### Ranked image-similarity graph

Choose **Export similarity graph…** to rank all 438 dropdown layouts in one plot,
using each layout's best-fitting 0°/90° orientation. The export saves a vector
PDF, PNG, ranked CSV, and JSON settings/scores together. It has a background
progress window and cancellation, and uses the current RGB/optimization settings.
Click the same button during a calculation to reopen its progress window.

The horizontal axis is **RGB RMSE: lower is closer**. Hollow markers show direct
assignments before optimization; filled markers show final assignments. Colors
separate layouts using at most 64 origami types from those needing more, and each
label gives its count and orientation. Duplicate dropdown layouts remain visible.
This is numerical image error, not an arbitrary percentage or a perceptual metric.

With alignment enabled, scoring uses the **held-out expected alignment image**
(seed 2718), not the training-model prediction or a single finite-count example.
The score excludes finite-count variance and the regularization penalty; those
terms still affect the optimizer. Without alignment it scores the ideal prediction.
Signals are unclipped, so overbrightness is penalized. The color space follows
the linear-light checkbox. Known fitting padding is excluded, and predictions
are resampled to a common source-content raster (256 pixels on its longest side).
With **Fit with black padding**, the entire uploaded image is compared; **Center
crop** instead evaluates its fitted/cropped region, which can vary by layout.
The chart notes these conventions so blank margins cannot artificially improve
the ranking.


Joint six-bit tiling sweeps
--------------------------
The 18-, 21-, and 20-position joint-code budgets include every ordered X:Y
factorization, then every tile count from 1 to the available states on each axis.
This gives 108, 84, and 120 layouts, respectively, while retaining 54, 63, and
60 origami types. For example, the 18-position family includes 1×18, 2×9,
3×6, 6×3, 9×2, and 18×1 state grids, with all tile/shift allocations within each.
Local shifts are distributed as evenly as possible among equal-sized tiles
(for example, 7 X states across 3 tile columns gives 3, 2, and 2 local X shifts).
The sweep covers these balanced rectangular layouts; it does not enumerate
arbitrary unequal partitions or permutations of which tile gets extra shifts.
These 392 position-budget sweep entries appear in the encoding dropdown, tiling/FWHM comparison,
exhaustive PDF, and similarity graph. Family filters identify the total position
and origami budgets, independently of the chosen X:Y factorization.
Existing layout identifiers and original/four-quadrant aliases are preserved.

The original 16-position budget also has a complete sweep: **80 layouts**
across 1×16, 2×8, 4×4, 8×2, and 16×1 state grids. Each uses 48 types.
Two original color bits are preserved; the remaining four bits jointly index
the tile and local shift, allowing uneven tile partitions. The original 35
bit-allocation entries retain their existing identifiers and barcode mappings.
Choose **16 positions / 48 types (80 layouts)** in the tiling comparison.

### Run similarity optimization

**Run similarity optimization** first optimizes and ranks every dropdown scheme
at the current starting values, choosing its best-fitting 0°/90° orientation.
This analysis enables replacement optimization and alignment simulation, without
changing the active RGB tab. Other light, volume, fit, alignment-center, count,
and regularization settings are copied from the tab.

The best 20 entries retain their optimized assignments and orientation throughout
a full Cartesian sweep of X translation SD, Y translation SD, rotation SD, and
Gaussian FWHM. Each axis uses 50%, 100%, 150%, and 200% of its starting value
(up to 256 conditions per finalist). An initial error of zero remains zero;
duplicate zero-valued conditions are evaluated once. Assignments are **not**
reoptimized under each condition. Equivalent dropdown entries remain candidates.

The winner has the lowest **worst-case source-image RGB RMSE** over the grid;
mean RMSE breaks ties. Scores use the same unclipped, source-content comparison
as the similarity graph and independent stress-test alignment samples (seed 31415).
They measure the expected image, not finite-count variability or the regularization
penalty. The conclusion is conditional on this sampled uncertainty range and model.

The progress window covers both stages and supports cancellation. A completed run
saves a three-page PDF (ranking, sensitivity curves, starting settings), a ranking
PNG, nominal/robustness/sweep CSV files, and JSON with settings, rankings, and the
winning assignments. Sensitivity curves show the worst score across the other
three axes, exposing coupled errors. Unsupported simulation settings stop the run
with an error rather than silently dropping difficult conditions. The current
selected design and prepared runs are not modified.
