# Split–pool encoding analysis

Open **RGB Image → Split–pool encoding analysis** subtab to analyze current image site fractions, load a headerless CSV / `.npy` matrix, or run synthetic benchmarks. Split–pool image mode fits all colors together as one synthesis population with one shared K library. The **Channel for detail plots** selector chooses the channel shown in numerical plots. **X/Y design location** selects an origami design by its X/Y shift (and tile when applicable), opening its analog canvas. Channel switching preserves that location when available. These are design shifts, not individual image-pixel coordinates. The selected target design stays the same when changing fitted models, even when their address assignments differ. CSV/NumPy benchmark rows must be a power of two, and columns identify analog positions. Image designs may have any positive count: all colors share a unique barcode space sized to the total design count (at least one round), plus any requested extra rounds. For example, 21 total designs use 21 of 32 addresses across five minimum rounds. Every design is retained; unused address populations are excluded from the loss and displayed image, rather than filling the dataset with invented targets. The preview states the used/available addresses for each color. N remains the number of measured origami per used barcode, not a total divided across unused populations. Image dimensions are inferred, so a non-256 × 40 image is analyzed at its actual dimensions. Metadata records site names, target design names, and original barcodes. The optimized address mapping is an analysis artifact; applying it to image synthesis requires a separate barcode/layout redesign.

Click **Visualize analysis** to explore results directly in the window. Only the selected model is fitted; choose Image Monte Carlo, model comparison, target/reconstruction heatmaps, an individual analog canvas, sampling error, or convergence. Select a model and address to inspect its results. The plot toolbar supports zoom, pan, and optional image saving. Analysis runs in memory on a background thread and can be cancelled; no output folder or automatic export is required. Use the CLI below for full batch exports. SciPy is required in addition to the RGB generator's NumPy and Matplotlib dependencies.

**Image Monte Carlo** opens by default for image targets, with the ON/OFF model when 25 nL recipes are enabled (additive in ideal mode). It shows current image targets, the expected split–pool image, and a finite-count realization. Set **Origami / barcode / color** (default 300), then **Update image** or **New finite-count example**. Counts and example seeds change the sampled image without refitting the basis vectors. All channels use their optimized inverse address mappings before reconstruction; current image designs and barcodes remain unchanged. Each color is an independent library with its own mixtures, not one shared 16-mixture RGB library.

For **Synthetic benchmarks** and imported CSV/NumPy datasets, the same Monte Carlo view shows three matched heatmaps: target matrix, expected reconstruction, and a finite-count realization. Rows are restored to original target order; columns are analog sites. No RGB geometry is invented for these datasets. **Update image**, sample count, and **New finite-count example** work without refitting. Quantized ON-only/ON/OFF fits use their realized recipe probabilities; PCA uses a labeled clipped reference, and additive is labeled an ideal reference. All three panels share the [0, 1] color scale.

Sampling uses `K ~ Binomial(N, p)` independently for each analog site and barcode. This is equivalent to sampling N independent irreversible incorporation trajectories conditional on that barcode. Previews use the RGB view’s Gaussian FWHM, exposure, linear-light display, and normalization settings captured when fitting starts. Alignment and site classification are ideal; this view does not add the main window’s alignment or brightness error models. The first panel is the current encoded target image rendered with those same settings, not the raw uploaded photograph. PCA is available as a **hypothetical reference preview**: its reconstructed values are clipped to [0, 1] before rendering and binomial sampling. The preview reports the fraction clipped, raw PCA RMSE, clipped-reference RMSE, and sampling/total errors separately. Raw fitted predictions and comparison metrics remain unchanged. This does not make PCA a split–pool synthesis model.

The progress bar updates within fits and across K values, models, and channels. Its percentage tracks the configured fitting budget, including restarts, assignment cycles, optimizer iterations, and discrete 25 nL refinement; it is not a time estimate. Early convergence can advance it by a larger step. Cancelling preserves partial progress. “100% • fits complete” refers to fitting and readout calculations; an asynchronous image render may still briefly be finishing.

From `picklist_generator`:

```sh
python -m pip install -r split_pool_encoding/requirements.txt
python -m split_pool_encoding --config split_pool_encoding/config.json --output results/experiment
python -m split_pool_encoding --input my_targets.npy --output results/real_data
python -m split_pool_encoding --quick --bits 2 3 4 5 6 7 8 --output results/scaling
python -m split_pool_encoding --quick --bits 3 --strategy per_site --output results/per_site
python -m pytest split_pool_encoding/tests -q
```

`--quick` uses one restart, three assignment cycles, 35 continuous iterations, one dataset realization, and ten readout repetitions. This is a validation budget, not evidence of global optimality. The default configuration uses three restarts, twelve cycles, 150 continuous iterations, two realizations, and 100 readout repetitions. The GUI uses one dataset realization per click; the CLI controls replicated experiments. Use the same seed and budget to compare `joint` and `per_site` optimization. The latter solves independent analog positions with a shared fixed assignment at each alternating step. It can converge more thoroughly per position but incurs more optimizer calls.

## Models and constraints

* PCA: centered rank-R SVD, without clipping. Its arbitrary per-record coefficients are not split–pool implementable. Batch experiments and the sampling-error curve omit PCA readout. The image preview separately supports hypothetical binomial sampling of clipped PCA values, explicitly labeled as a reference.
* Additive: `c + address @ basis`. A softmax over constant-one, each bit, each complemented bit, and constant-zero coefficients ensures all vertices lie in [0,1]. This covers the interior of every valid affine cube, and approaches boundary solutions as logits diverge. There are (R+1)J affine degrees of freedom, though the redundant parameterization uses (2R+2)J logits.
* ON-only: `1 − product(1−u)`. It is additive in nonnegative hazard space, `−log(1−p)`.
* ON/OFF: the full irreversible U/ON/OFF recurrence. Bounded occupancy `q` and ON share `t` give `u=q*t`, `v=q*(1−t)`. These are effective conditional incorporation probabilities, **not pipetting volumes**.

The solver alternates exact Hungarian row assignment with analytic-gradient L-BFGS-B fitting, retaining the best restart. Fixed-assignment controls use the same seeds. Every reported loss uses the saved address-to-target-row permutation. `converged` describes the stopping condition only. Exact recovery of arbitrary full-size scrambled generated datasets is not guaranteed by alternating optimization. Warm-start recovery and independent gradient/state tests distinguish implementation checks from global search performance.

## Benchmarks and exports

Synthetic families include independent uniform, continuous low-rank (2/4/8), valid binary cubes, scrambled ON/OFF-generated data, smooth curves, Gaussian peaks, correlated vectors, and clusters. The attachment's sixth category is real data; it is supported through CSV/NumPy import rather than invented measurements.

Each experiment saves targets, configuration, a technical report, and CSV metrics. Each model saves `model.npz` (raw parameters, predicted probabilities, address assignment, and physical u/v or affine baseline/basis), `assignment.csv`, `per_file.csv`, `restarts.csv`, `convergence.csv`, and individual repeated sampling results. Generator ground truth is separate from fitted parameters and never used by blind fits. PNG/PDF figures show synthesis, heatmaps, canvases, learning, sampling, and model/structure comparisons. Error bars across realizations and raw restart outcomes represent different uncertainties; a single realization has no estimated between-dataset uncertainty.

Sampling separates encoding, measured-versus-generated, and total RMSE at N=50,100,200,300,500,1000. Optional false-positive/negative probabilities change the observation probability. With classification errors enabled, the measured-versus-generated term includes detection error as well as binomial fluctuations. Other biochemical and correlated measurement errors are not modeled.

Transfer estimates explicitly assume one dispense per site-specific reagent per branch environment plus one barcode dispense per branch; they exclude splitting, pooling, washing, stock mixing, volume constraints, and calibration. The additive estimate additionally assumes a separately realizable independent basis-writing mechanism. These outputs must not be sent to a liquid handler as a picklist.

## Scientific limits

Eight-round ON/OFF predictions are a polynomial image of at most 1280 transition parameters in a 10240-dimensional data space. Permuting rows gives a finite union of such images, still measure zero. Thus almost every arbitrary collection is not exactly representable. This does not supply a useful lower bound on approximate reconstruction error; experiments quantify that for the tested distribution and search budget.

Nonidentifiability is explicit: changing final-round OFF probability cannot affect final ON occupancy. Saturation hides later parameters. Parameter counts therefore overstate independent output dimensions. With finite parameter precision L, 1280 log2(L) is only a loose programming-bit upper bound, not measured storage capacity. Binomial standard error is at most 0.029 at N=300. Useful capacity additionally depends on distinguishability tolerances, calibration precision, population size, and correlations.

`results/full_quick` contains a reproducible full-size smoke benchmark, and `results/scaling_quick` contains the round-count sweep. Their reports distinguish optimizer limitations from proven physical constraints. Increase budgets and realizations before drawing experimental-design conclusions.

A two-round counterexample demonstrates added expressivity directly: fill a site
with OFF in round-1 branch 0; incorporate ON with probability 0.6 in round-2
branch 1; leave other transitions zero. The four outputs are `[0, 0, 0, 0.6]`.
Their rectangular contrast is nonzero in both probability and hazard space, so
neither an additive cube nor an ON-only model can represent this collection.
This proves an expressivity gain from OFF competition, not universal encoding.

Add `--validate-known` to export full-size oracle-initialized reconstruction
checks. These are explicitly distinguished from blind random-start fits.
`replicated_config.json` supplies the bounded two-realization, three-restart
benchmark used for the technical report; its per-model figures are disabled to
keep batch runs compact. Full figures are provided by the quick benchmark.

ON-only and ON/OFF exports also include `precision.csv`, measuring reconstruction
error after rounding occupancy q and ON share t to 4, 16, 64, or 256 levels.
This preserves valid transition probabilities. The reported programming-bit
bound counts available parameter settings; it does not count distinguishable
output datasets, and the assumed precision coordinates are explicitly q/t.

## Learned shared basis-library size K

Set **Shared basis-library size K** (1–256) and click **Visualize analysis** to
fit a library specifically to the current targets. Both its vector entries and
the branch coefficients are optimized, together with the address assignment.
Each color has its own library, shared across all its rounds and branches.
K does not change the number of rounds or the address capacity.

Enter comma-separated values such as `1,2,4,8,16,Unrestricted` in **K sweep** and
click **Fit RMSE vs K**. Selecting the **RMSE vs K** plot only displays cached results; it never starts or queues fitting. Click **Fit RMSE vs K** explicitly to run the sweep. **Visualize analysis** fits only the selected model at the selected library size. Results appear progressively, with requested K values on the axis and completed-value counts shown above the graph. Each point reports the
encoding RMSE of the best random start at that K. For images this is the
root-mean-square error over all sites and all colors (not an average of channel
RMSEs); measurement noise is excluded. Select a fitted K in the size selector
and choose **Image Monte Carlo** to compare its image without fitting again.
An unfitted K needs another visualization run. A new run replaces the previous
sweep. `Unrestricted` uses the original independently programmed mixtures.
PCA is a fixed reference at the original round-count rank, not a K-sized basis
library. Local minima can make empirical curves non-monotonic.

For ON-only, each learned vector is a site-wise ON-probability profile. For
ON/OFF, each vector specifies a valid ON/OFF/unoccupied probability triple at
each site. Each branch learns nonnegative mixing weights shared across sites,
plus an empty-treatment weight; the weights sum to one. Thus branch mixtures
remain valid, and every population in a branch sees the same combined mixture.
Mixing these effective probability profiles is an explicit simulation
assumption, not a calibrated rule for mixing pipetted concentrations.

The additive reference learns K signed site vectors and round coefficients;
its fitted contrast matrix has rank at most K, with a separate baseline.
A differentiable (piecewise at ties) per-site scaling guarantees [0,1] output
for every address. Signed contrasts are an abstract additive encoding model,
not nonnegative reagent concentrations. All library parameters and resulting
predictions are retained in memory; the Python API exposes
`fit(..., basis_k=K)` and returns `library_parameters` containing learned vectors
and coefficients. The CLI's existing experiments remain unrestricted.

## Concentrations and two-stage 25 nL recipes

**25 nL stock preparation + branch transfers** is enabled by default in the GUI.
Choose **Stocks, dilution & capture…** to set the final diluted volume of each
K-vector well, the reference staple amount per site, the final branch volume,
the maximum vector volume allowed in a branch, and the assumed capture C50.
ON and OFF stock concentrations are read from the RGB generator's replacement
and base-stock settings when fitting begins. The scalar stock concentration
for each reagent type applies to every site, as in the current RGB settings.

Each vector is prepared in a single well by adding integer 25 nL aliquots of
ON/OFF stock for every analog site and filling to the chosen volume with buffer.
The reference amount defaults to the RGB body target × final volume, converted
to fmol. It defines the continuous full-site dose scale, not a compulsory equal
amount at every site. Changing dilution volume at a fixed reference amount does
not implicitly increase the amount of staple dispensed. This parameterization
is then projected onto legal droplets, so the realized per-site concentrations
come from actual ON/OFF volumes and their respective stock concentrations.

Branches receive integer 25 nL aliquots of the prepared vector wells. Their
per-site concentrations are calculated from delivered moles divided by final
branch volume. All transfers together must fit the branch-transfer budget, and
withdrawals across ALL rounds/branches cannot exceed the volume in any vector
well. Remaining branch volume is reserved for origami/buffer and other reagents;
set the transfer budget accordingly. This models analog transfers, not a complete
buffer/scaffold recipe or a calibrated liquid-handler picklist.

With no experimental calibration, the explicitly labeled working model is:
`q = 1 − exp(−ln(2) × (C_ON + C_OFF) / C50)`,
`u = q × C_ON/(C_ON + C_OFF)`, and `v = q × C_OFF/(C_ON + C_OFF)`.
Zero total concentration gives no incorporation. ON and OFF have equal capture
rates in this model; occupied sites remain irreversible. C50 defaults to the
current body-target concentration only as a starting assumption, NOT a measured
calibration. Change it to explore sensitivity. Volumes are never interpreted
directly as occupancy probabilities.

The optimizer learns a continuous concentration model, projects both preparation
and branch transfers onto the droplet grid, then performs a bounded deterministic
one-droplet local search. It selects the best restart by REALIZED recipe error.
This is a heuristic discrete fit, not proof of the optimal quantized recipe.
The **Transfer recipe** view shows both sets of actual nL transfers and compares
continuous versus realized RMSE. Recipe arrays, concentrations, remaining well
volumes, settings, and effective u/v are retained in each fitted result.

The Monte Carlo image samples the final probabilities obtained from those
realized u/v values. The RMSE-vs-K curves for ON-only and ON/OFF likewise use
realized recipe errors. Increasing K also increases the total prepared material
when volume per vector well is fixed; these curves reflect that resource change
as well as library expressivity. Additive (signed vectors) and PCA remain labeled
ideal references, not physical 25 nL recipes. Disable the recipe option and refit
to view ideal additive Monte Carlo examples. PCA's clipped hypothetical reference
preview remains available. CLI defaults remain the original probability models;
Python callers can pass `RecipeSettings` via `fit(..., recipe_settings=..., basis_k=K)`.

## Inspecting every learned vector

Choose **K vectors** after fitting. The heatmaps show every vector as a row and
every analog site as a column. For image targets, **Channel for detail plots**
selects that color's library; changing the fitted K or model updates the view.
Click a row, or use the **Vector** selector, to inspect its site-by-site profile
beneath the overview. Vector numbering starts at 1. Pan/zoom and optional plot
saving use the existing toolbar.

With 25 nL recipes enabled, the view shows actual ON and OFF concentrations in
the prepared, diluted wells, using matched nM color scales. Ideal ON/OFF fits
show ON/OFF/unoccupied probabilities; ON-only shows ON probability. Additive
fits show signed vectors with a zero-centered color scale. PCA shows its fitted
principal components and identifies them as a rank-based reference rather than
a K-vector stock library. Unrestricted fits have no shared K library; select a
finite fitted K to use this view.

### PCA-guided library initialization

**Try PCA-seeded library (+1 fit)** is enabled by default in the analysis subtab.
For each color and finite K, it adds one PCA-informed starting point to the
configured number of random restarts. PCA is computed on that color's targets.
The physical library begins with the target mean and clipped mean-plus/minus
principal variations; extra entries use PCA-reconstructed target profiles.
ON-only profiles use a per-round occupancy transform; ON/OFF profiles allocate
nonnegative ON, OFF, and vacant fractions. The additive reference starts with
signed principal directions. These are initial guesses, not direct PCA recipes.

All vector profiles, branch weights, and permitted address assignments are then
optimized using the existing model. Physical fits undergo the same 25 nL
projection and discrete refinement. The lowest final error wins across random
and PCA-seeded candidates. **PCA seed comparison** displays the best random
restart and the PCA-seeded trial's final RMSE for the selected channel. The
summary identifies the retained initialization. Turning the option off restores
random-only initialization. The separate `pca` reference model is unchanged.

This method does not guarantee a globally optimal library, and PCA cannot remove
shared-branch restrictions. Each color retains the better candidate within its
available address set; since address allocation across colors is sequential,
changing an earlier color's solution can change later colors' available addresses.

### Extra split–pool rounds

The **Extra split–pool rounds** control adds rounds beyond the minimum required
for unique addresses, up to the current implementation limit of 12 total rounds.
For example, 120 total RGB designs require seven rounds; two extra rounds give
nine rounds and 18 shared branch mixtures across all colors. Every extra
round contains branches 0 and 1, followed by pooling, just like earlier rounds.
It lengthens the shared synthesis sequence rather than assigning different
sequence lengths to individual branch wells. It also increases the digital code
length and available address space; all colors still receive unique addresses.

Changing the control does not trigger fitting. Click **Visualize analysis** or
**Fit RMSE vs K** to apply it. All K values in a sweep use the same round count.
Recipes, Monte Carlo, and trajectory/synthesis plots use the fitted round count.
Additional rounds share the same available K-stock volumes, and the existing
capacity constraints remain active. The PCA reference rank also follows the new
round count, subject to data dimensions. Synthetic and imported targets support
extra rounds as well.

### Independent-well synthesis

Choose **Synthesis mode → Independent wells** to give every unique design its
own destination well. This mode fits ON-only and ON/OFF libraries plus the PCA
reference; signed additive split–pool fitting is excluded. K remains **per
color**, so K=128 means up to 128 prepared vector stocks for each RGB channel.
The stock library is shared across that color's wells, while every destination
has its own optimized nonnegative transfer amounts. The configured destination
volume and transfer budget use the same settings as split–pool branch wells.

All vector contributions are combined into ON/OFF concentrations first. The
uncalibrated capture law is applied **once** to each destination, followed by
binomial finite-count sampling for the Monte Carlo image. There are no branch
histories or extra rounds. Every design receives a unique simulated barcode
identifier, but barcode preparation/incorporation chemistry is not modeled.
Completed populations may be pooled after assembly/capture.

Both stock preparation and destination transfers retain 25 nL quantization,
well-volume limits, and the shared-stock depletion constraint. The optimizer
uses independent destination coefficients, random starts, and optional PCA
initialization. When K is at least the number of designs in that color, it also
tries a one-vector-per-design initialization using the inverse capture model.
The best realized recipe wins; finite preparation/dispensing precision and
concentration limits can still prevent zero error.

**Image Monte Carlo**, **K vectors**, **Transfer recipe**, **Well trajectory**,
and **Full synthesis** support this mode. Trajectories show one mixing/capture
step. Full synthesis shows all destination recipes and pooling only after
capture. Extra-round and address-assignment controls are disabled; use
**Visualize analysis** or **Fit RMSE vs K** to explicitly refit after switching
modes. Old plots remain labeled with their fitted mode until refitting.

### Comparing independent wells with the normal RGB preview

Independent wells now expose an explicit **Recipe capture response** selector:

- **RGB ratio (complete occupancy)**, the default independent-well comparison:
  expected brightness is ON concentration / (ON + OFF concentration), with zero
  signal when both are absent. This matches the normal RGB preview's molar-ratio
  assumption. It is an ideal complete-occupancy reference, not a calibrated
  claim that diluted staples always incorporate. ON-only mixtures under this
  assumption are fully ON wherever any staple is present.
- **Concentration-dependent capture (C50)**: brightness is the ON fraction times
  `1 - exp(-ln(2) * total concentration / C50)`. The existing default C50 equals
  the per-site reference concentration, so the default continuous independent
  fit has a roughly 50% maximum occupancy before quantization. This response is
  intentionally different from the normal RGB preview. Split–pool mode defaults to this concentration-dependent response; its
  saturating ratio alternative is described below.

For current RGB designs, independent ON/OFF fitting also evaluates the original
ON/OFF site transfers as a fixed candidate when K can provide one vector per
design. It copies the already quantized recipe into dedicated vector wells,
dilutes them, and dispenses each only to its corresponding destination. It checks
well capacity, 25 nL increments, source availability, and the configured per-site
reference amount. The optimizer cannot replace that candidate with a worse
recipe. If it is infeasible, the summary explains why; increase preparation
volume/reference amount as appropriate. The ratio response preserves the original
molar fractions through dilution (using the same stock concentrations). C50
capture need not preserve their brightness. Finite-count noise remains separate.

### Split–pool response and recipe safeguards

**Recipe capture response** now applies to both synthesis modes; each mode
remembers its own selection. Split–pool still defaults to uncalibrated C50
capture. **RGB ratio (complete occupancy)** is a saturating per-round reference:
a nonzero ON/OFF exposure fills the remaining empty sites in that mixture's molar
ratio. Zero-dose sites remain empty. Previously occupied ON/OFF sites are never
overwritten, so earlier OFF incorporation still blocks later ON capture. With
complete occupancy, the first nonzero exposure at a site determines its outcome;
adding rounds does not undo that. This is not the independent-well RGB solution.

For C50 split–pool fits, the summary shows the continuous-reference per-round
capture ceiling and the optimistic multi-round ON ceiling. At a 50% per-round
ceiling, seven rounds could reach 1-(1-0.5)^7 = 99.21875% ON in the absence of
OFF blocking and other constraints. Shared stock capacity, branch restrictions,
and quantization can lower achievable brightness; the displayed ceiling is not
a fit-quality guarantee. No biochemical calibration is implied.

Physical fitting in both modes now compares each starting quantized recipe with
the optimized/projected/refined result and retains the start if it has lower
realized RMSE. The summary reports that fallback. This protects known feasible
starting recipes from being lost during continuous optimization and rounding;
it does not guarantee a global optimum or preserve fits from previous runs.
Copying one original RGB recipe per design remains restricted to independent
wells, since split–pool designs share branch exposures.

The RGB page contains **Image generator** and **Split–pool encoding analysis** subtabs. Analysis is initialized when first selected, stays alive when switching tabs, and retains controls, fitted results, and in-progress work. Closing the application cancels its worker and polling callbacks. The analysis subtab scrolls vertically to accommodate its controls and plots.

### Analysis dashboard and spatial vector canvases

A normal **Visualize analysis** run now opens **Dashboard**, showing the image
(or matrix) Monte Carlo preview, spatial K-vector library, selected vector's
ON/OFF source preparation doses, fluidic additions for the selected design,
complete synthesis/pooling, and model accuracy together in the scrollable tab.
Each section has its own plot toolbar. Changing color, design location, model,
or selected vector refreshes the related views without refitting. K sweeps
still start only from the explicit sweep button.

K-vector images use the actual site row/column mapping, preserving missing sites;
the usual 40-site canvas appears as eight rows by five columns. Coordinate-only
layouts use their physical X/Y positions. Synthetic matrices without spatial
metadata use a clearly labeled compact schematic layout (40 positions also map
to 8×5 for display only). ON/OFF concentration or probability panels share a
scale; signed references use a diverging scale. Clicking a vector selects its
larger canvas and preparation doses. Libraries are paged eight vectors at a time
with **Previous 8**, **Next 8**, or the Vector selector, so all K values remain
accessible without compressing hundreds of canvases into an unreadable plot.
The plot dropdown still provides focused views of individual analyses.

### Canvas construction: visual addition of K vectors

Select a color and X/Y design, then use **Canvas construction** or its dashboard
section. **Previous**, **Next**, the step number, and **Final canvas** navigate
all nonzero K-vector additions and capture steps along that design's assigned
barcode path. Each step shows spatial ON and OFF equations:

`mixture before + delivered K contribution = combined mixture`

For physical recipes, the delivered contribution is the diluted stock
concentration multiplied by its actual transfer volume and divided by the final
destination volume. All three canvases share the same concentration scale.
The lower row shows pre-round ON occupancy, the expected ON canvas if the
current mixture were captured, and the target. The site layout matches the RGB
canvas; matrix data uses the labeled schematic layout.

K order is only a visual listing: vectors within a well are mixed before capture,
not applied as separate sequential reactions. Partial-mixture occupancy previews
are hypothetical. Actual ON/OFF state advances only on **capture complete**.
Split–pool rounds retain prior occupancy and OFF blocking; independent wells
have one final capture step. Ideal ON-only/ON/OFF libraries show weighted
probability contributions instead of nM. PCA and signed additive references do
not invent physical K-addition steps. The last capture matches the fitted
expected prediction; no refitting or Monte Carlo resampling is required to step
through the construction. Changing the selected design/color resets to its
initial canvas.

### Responsiveness

The dashboard reuses existing canvas-construction images when stepping and
redraws only sections affected by the changed selection. Monte Carlo images are
not rebuilt when only a construction step changes. Incoming model updates are
batched before refreshing the dashboard. Optimization logging reuses the
objective value already computed for the accepted iterate rather than repeating
the full forward/gradient calculation. These changes do not reduce the fitting
budget or simulation detail.

Independent-well ON/OFF fitting checks the validated original RGB recipe before
optimization. If its MSE is at most 1e-24 (numerical equality), it returns that
recipe immediately and labels optimization as skipped. Nonexact or infeasible
recipes still use the full configured search and comparison. Capture response,
25 nL transfer checks, stock limits, and Monte Carlo sampling remain unchanged.

### Shared RGB split–pool population

Split–pool analysis fits every RGB design together, using one K-stock library,
one address assignment, and one pair of branch recipes per round. For 40
designs per color, all 120 designs share seven rounds and 14 branch wells.
K counts the total shared library; it is not multiplied by the number of colors.
Color controls select slices of the fitted population for detail plots and image
rendering. Full synthesis shows one population and labels every nonzero K-stock
addition with its vector identity and nL volume. Old separate-color results must
be refitted. Independent-well synthesis retains its per-color library behavior.

### Inspecting one design through synthesis

Select **Channel for detail plots** and **X/Y design location** (or Design index).
The Dashboard outlines that design's sites in white with a black halo on the
complete image target, using the same placement and rotation as the image preview.
**Selected origami** shows its full site layout, original digital barcode, and
fitted analog canvas. **Full synthesis** highlights its assigned branch route in
gold, following the fitted address assignment rather than the original row order.

Each branch labels its planned digital bit (ON or OFF), and the selected route
shows the growing barcode prefix. These are logical address labels; the analysis
does not assign DNA sequences for digital-barcode additions. The original design
barcode is shown separately because optimization can assign a different address.

**Selected route evolution** shows delivered ON/OFF K-vector canvases and the
accumulated ON/OFF pattern after every capture. Every nonzero addition used by
the selected route is shown, regardless of K number; larger sets wrap into rows.
Library paging applies only to the separate K-vector library view.
**Canvas construction** offers individual addition steps: the mixture before,
the added K contribution, the combined mixture, and the resulting occupancy.
All K additions within a branch mix before a single capture event; earlier
occupied ON/OFF sites remain occupied in subsequent rounds. Changing a selection
or a display step reuses the fit and does not start analysis again.

The Dashboard combines complete synthesis and route evolution into one graphic.
Each round occupies one row: its highlighted split/pool diagram is on the left,
with the selected branch’s K-vector contributions and accumulated ON/OFF pattern
on the right. The combined toolbar saves the entire graphic. Route diagrams
show all active additions and do not filter by library page. “No K additions”
appears only for a round with no nonzero transfers.

The default **Dashboard** shows only the full target image with the selected
design highlighted and the combined **Complete synthesis and pooling** graphic.
Other views are optional in the plot selector, including Image Monte Carlo,
Selected origami, K vectors, K stock preparation, and Canvas construction.

In the combined synthesis view, each K stock appears once as a premixed ON/OFF
addition with one transfer volume. A single accumulated-pattern map follows each
capture. Cyan represents ON, orange OFF, and dark represents no delivered dose
(in additions) or unoccupied sites (after capture). Component colors describe the
mixture or site-state fractions; they are not separate dispensing operations.

### Exactly representable K-vector benchmarks

`exact_k_2`, `exact_k_4`, and `exact_k_8` start from feasible shared ON/OFF stock recipes, quantize both preparation and branch transfers to 25 nL, and simulate expected occupancy before shuffling the target rows. They use 2, 4, or 8 K stocks respectively; this does not specify matrix rank or prove that fewer stocks cannot suffice. Each default dataset has 256 targets, 40 analog sites, and eight rounds. Exactness applies to the assumed capture model, not finite-count sampling or measured chemistry.

In the UI, select a family under Synthetic benchmarks, enable 25 nL recipes in split-pooling mode, select ON/OFF and the matching K, and run analysis. Generation uses the current physical settings, capture response, and extra-round count. The matrix preview reports the known-recipe reconstruction error separately from the blind optimizer's error. Fixed assignment, another model, or fewer K vectors may exclude the generating solution. Even with matching settings, blind optimization is not guaranteed to recover it.

To export a reproducible feasibility demonstration using the benchmark's reference settings:

```bash
python -m split_pool_encoding.exact_benchmark --output results/exact_k_demo
```

The output includes target CSVs, branch-volume CSVs, complete source/branch recipes and shuffled address mappings in NPZ files (loadable without pickle), metrics, and a comparison PNG. Replay independently rebuilds occupancy from the saved transfer volumes. Default seed 17 gives RMSE below 7e-17 for all three datasets; the remaining difference is floating-point rounding. This is a known-construction check, not a fitted recovery claim.

### Compact reversible digital mapping

`digital_affine` tests binary data independently of the analog capture model. Rows are files; columns are bits. It constructs a GF(2) row-reduced basis for the differences from the first file and verifies `file = offset XOR (code @ basis mod 2)`. This is an affine function, not a lookup table. Its full n-bit transform also preserves nonpivot residual bits, making it invertible even outside the fitted collection. Residuals vanish on the fitted collection.

With eight independent binary code-writing rounds, affine dimension at most eight is sufficient for the generated code library to contain the collection. This is a criterion for this affine decoder architecture, not an impossibility result for nonlinear decoders or history-dependent chemistry. Extra address bits can distinguish duplicate decoded payloads. Distinct requested payloads may leave unused generated payloads.

Run from `picklist_generator`:

```bash
python -m split_pool_encoding.digital_affine --demo --output split_pool_encoding/results/my_digital_demo
python -m split_pool_encoding.digital_affine --input files_as_bits.csv --address-bits 8 --output digital_analysis
python -m split_pool_encoding.digital_affine --folder /path/to/files --address-bits 8 --output digital_analysis
```

Inputs must be binary CSV/NumPy matrices or a folder of equal-length, nonempty files. Folder files are sorted by name; hidden files and subdirectories are excluded. Variable lengths are rejected rather than silently padded. Names, directory structure, and original per-file length metadata are not archived; the binary matrix round trip preserves row order, repeated rows, and all content bits.

The output `.gfa` files contain an actual compact binary representation: a 20-byte header, packed offset and basis, and an ordered address manifest, with byte padding. `unpack(Path(...).read_bytes())` reconstructs the binary matrix with no external dataset or lookup table. The decoder can produce every code combination via XOR. For infeasible cases the archive uses rank-bit codes and reports failure for the requested address width; archive round-trip success does not imply eight-round feasibility.

The JSON report counts the entire decoder, ordered selection manifest, header and padding against the original packed bit matrix. Exported diagnostic CSV and JSON files are not the compressed archive. Physical molecule counts, synthesis redundancy and measurement errors are separate costs; this tool does not simulate digital chemistry or demonstrate an experimental writing protocol.

Seed-17 examples each contain 64 distinct 128-bit files (1,024 packed bytes): dimension 6 uses 196 archive bytes; dimension 8 uses 228 bytes and leaves 192 unused payloads among 256 addresses. Random files have dimension 63 and use 1,548 bytes: they neither fit the eight-bit affine architecture nor compress with this representation.

### Flexible bulk digital-writing simulator

`digital_bulk.py` models identical origami with empty payload sites and separate empty address sites. Every site has three states: empty (-1), permanently OFF-blocked (0), or ON (1). An addition fills only empty sites. Repeated and overlapping mixtures are allowed: an early blocker protects a site from later ON additions. This is an ideal complete-binding model, not the analog C50 model.

A schedule can split any live well into any number of branches (equal or specified fractions), add a reusable multibit mixture, pool any selected wells, and leave other wells separate. Populations retain their site states after pooling. The simulator tracks exact state fractions, checks requested payload coverage, and checks whether fully written molecular addresses unambiguously identify those payloads. Logical path labels are not substituted for physical barcode writing: barcode sites must actually be written. Empty payload sites are not interpreted as OFF. Address manifests contain target row indices, not file contents.

From `picklist_generator`:

```bash
# Compare candidate schedules on structured and random files; save plots and JSON.
python -m split_pool_encoding.digital_bulk --demo --search --output split_pool_encoding/results/new_bulk_search
# Binary matrix input: one m-file row per n payload bits.
python -m split_pool_encoding.digital_bulk --input targets.csv --search --output bulk_analysis
# A specific automatic schedule.
python -m split_pool_encoding.digital_bulk --input targets.csv --block-bits 4 --pooling bounded --max-pools 2 --output bulk_bounded
# Run a hand-designed schedule, including overlapping ON/OFF mixtures.
python -m split_pool_encoding.digital_bulk --input targets.csv --program schedule.json --output bulk_custom
```

The initial constructive compiler partitions payload sites into contiguous blocks. Each branch writes one observed block pattern and the stage's address digit. It then merges either all populations (`all`), only populations with identical remaining suffix sets (`exact`), or a bounded number of groups (`bounded`). Bounded merging favors similar suffix sets. All policies preserve requested-file coverage; `all` and `bounded` can produce extra cross-combinations. The maximum pool count limits groups *after merging*, not instantaneous working wells. Branching widths vary with the data. `--search` compares block widths 1, 2, and 4 and these pooling policies, preferring identifiable coverage, satisfaction of the total single-use destination-well and transfer goals, useful-product fraction, then fewer transfers. This is a bounded heuristic, not a globally optimal search. In particular, the compiler currently writes disjoint payload blocks, while the simulator also supports overlapping blocker programs; searching that larger program space remains future work.

Cost accounting includes one source preparation transfer for each active site reagent in each unique used mixture, one mixture addition per treated well, one transfer per split output, and one transfer per pooled source. Mixtures include barcode reagents. Reports separate preparation from costs with an already prepared library, report source-stock and mixture wells, and compare modeled total transfers to the payload-only baseline m*n. This baseline excludes conventional address-writing costs.

Destination wells are strictly single-use. Every split output and new pooling destination consumes a fresh physical well; vacated wells cannot be refilled, even under their old name. Continued additions to the same resident population are allowed. Prepared multibit mixture source wells remain reusable, with preparation charged only once. `destination_wells_consumed` includes the initial population well. The search and `meets_destination_well_goal` compare this total against m; `meets_single_use_well_goal` is an alias. Peak simultaneous occupancy is a separate diagnostic and never substitutes for consumed wells in the optimization goal. Buffer additions, volume limits, stock depletion, cross-talk and binding errors are not modeled, so neither the transfer totals nor the schedules are Echo-ready protocols. Mixture preparation assumes one transfer can supply each site reagent at its required dose.

Outputs include targets, explicit schedules, operation traces, coverage/address manifests, complete metrics, and PNGs showing well usage, transfer costs, mixture patterns and requested-file abundance. Useful-product fraction and a finite-read missing-file union bound expose schedules that meet well/transfer goals only by making mostly unwanted products. The bound assumes independent random molecule reads, not an experimentally calibrated PAINT sampling process. Exact simulation is capped by `--max-states` (default 100,000); exceeding it is reported as a failure, not silently approximated.

Custom schedule format:

```json
{
  "payload_bits": 1,
  "address_bits": 2,
  "mixtures": {
    "early_off": [0, 0, -1],
    "early_open": [-1, 1, -1],
    "late_off": [0, -1, 0],
    "late_on": [1, -1, 1]
  },
  "operations": [
    {"op": "split", "source": "start", "outputs": ["a", "b"]},
    {"op": "add", "well": "a", "mixture": "early_off"},
    {"op": "add", "well": "b", "mixture": "early_open"},
    {"op": "pool", "sources": ["a", "b"], "destination": "p"},
    {"op": "split", "source": "p", "outputs": ["c", "d"]},
    {"op": "add", "well": "c", "mixture": "late_off"},
    {"op": "add", "well": "d", "mixture": "late_on"},
    {"op": "pool", "sources": ["c", "d"], "destination": "final"}
  ]
}
```

This program computes payload = address-bit-1 AND address-bit-2 using irreversible OFF blocking. Mixture entries are payload sites followed by address sites. Splits optionally accept a `fractions` array of positive numbers summing to one. Every new destination name must never have been used before; the initial population is in `start`. The final product consists of all wells remaining when the schedule ends.

### Reversible encoding for cheaper bulk writing

`reversible_bulk.py` searches for one small shared reversible circuit that maps the real payloads to easier-to-write payloads of exactly the same bit length. It supports NOT, SWAP, CNOT (control then target), and Toffoli (two controls then target). Every gate is self-inverse, so decoding applies the gates in reverse order. The circuit is a function over all possible input strings, not a file-specific lookup table. Payload bytes are never embedded in the decoder.

```bash
# Random 32-file × 16-bit experiment; default decoder budget is 25% of original bytes.
python -m split_pool_encoding.reversible_bulk --random --evaluations 100 --output reversible_experiment
# Your own binary files, represented one per CSV row.
python -m split_pool_encoding.reversible_bulk --input targets.csv --evaluations 500 --decoder-bytes 32 --output reversible_input
```

Each candidate is evaluated with the same bulk scheduler menu (block sizes 1, 2, 4; full, bounded, and exact suffix pooling). The score requires identifiable coverage and a minimum useful fraction (`--minimum-yield`, default 1%), then prioritizes satisfying both single-use-well and transfer goals, reducing the worst normalized cost, reducing summed normalized costs, and yield. A bounded elite search mutates, inserts, replaces and deletes gates. The published result cannot worsen either well consumption or modeled transfers relative to the baseline selected by this same evaluator. It retains identity if no qualifying improvement is found. Intermediate search candidates may be worse, and the default gate limit is 12. This evaluator's cost priorities differ from the earlier yield-prioritized bulk-only demo; before and after here always use the same evaluator.

The compact `.rbc` decoder format has an 8-byte header plus bit-packed instructions: 2 opcode bits and one site index per operand, with final byte padding. `--decoder-bytes` is a hard cap on the real serialized file, including header. The default cap is one quarter of the original packed payload bytes; small datasets that cannot accommodate the header require an explicit budget. `--max-states` defaults to 4,096. Failed baseline schedule candidates are recorded. All results remain constrained by the existing disjoint-block scheduling heuristic; the search does not yet optimize the larger overlapping-blocker program space.

Exports include original and encoded matrices, the standalone decoder, before/after synthesis schedules and plots, a comparison figure, search history and a detailed report. Tests decode the saved encoded CSV using only the saved circuit, then independently select payloads from the simulated final populations by their written addresses and decode those back to the original files. The address manifest preserves target row order and contains no payload lookup table. Decoder bytes are reported separately from physical wells, fluidic transfers, and address/order metadata; no claim of bit compression or overall digital archive compression is made.

To decode an n-bit matrix after reading:

```python
from pathlib import Path
from split_pool_encoding.reversible_bulk import unpack_circuit, apply
width, gates = unpack_circuit(Path('decoder.rbc').read_bytes())
original = apply(measured_encoded_bits, gates, inverse=True)
```

These are ideal noiseless simulations. Error correction and readout errors require separate evaluation: reversible operations can propagate measurement errors. The bulk simulator's source preparation, single-use destination-well, and incomplete physical-volume assumptions also apply.
