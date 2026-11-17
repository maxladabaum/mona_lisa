# Split–pool technical report

Implemented and exercised from the RGB analysis window and command-line package. Full-size results below use 256 × 40 targets, two dataset seeds (17, 18), three random starts per fit, six alternating cycles, and up to 80 L-BFGS iterations per cycle. Values are mean ± population SD of the best-start RMSE across the two dataset realizations. This is a bounded experiment, not a global-optimum certificate.

| Dataset | PCA | Additive | ON-only | ON/OFF |
|---|---:|---:|---:|---:|
| random_b8 | 0.23746 ± 0.00095 | 0.25803 ± 0.00104 | 0.25740 ± 0.00108 | 0.25182 ± 0.00129 |
| low_rank_2_b8 | 0.00000 ± 0.00000 | 0.01425 ± 0.00176 | 0.01765 ± 0.00272 | 0.01468 ± 0.00199 |
| low_rank_4_b8 | 0.00000 ± 0.00000 | 0.03087 ± 0.00133 | 0.03192 ± 0.00205 | 0.03067 ± 0.00253 |
| low_rank_8_b8 | 0.00000 ± 0.00000 | 0.05171 ± 0.00214 | 0.05220 ± 0.00222 | 0.05116 ± 0.00251 |
| binary_cube_b8 | 0.00000 ± 0.00000 | 0.06160 ± 0.00155 | 0.06083 ± 0.01159 | 0.04902 ± 0.02286 |
| on_off_b8 | 0.01065 ± 0.00047 | 0.02709 ± 0.00641 | 0.01396 ± 0.00024 | 0.01756 ± 0.00789 |
| smooth_b8 | 0.01616 ± 0.00001 | 0.12816 ± 0.00106 | 0.14238 ± 0.00146 | 0.12904 ± 0.00266 |
| peaks_b8 | 0.01270 ± 0.00092 | 0.07108 ± 0.00316 | 0.07292 ± 0.00462 | 0.06015 ± 0.00311 |
| correlated_b8 | 0.15083 ± 0.00281 | 0.21695 ± 0.00176 | 0.22030 ± 0.00151 | 0.21768 ± 0.00052 |
| clusters_b8 | 0.02178 ± 0.00002 | 0.13661 ± 0.00737 | 0.13669 ± 0.00808 | 0.11615 ± 0.00467 |

Independent random targets remain difficult. Low-rank distributions can fit much better, but low rank is not sufficient to ensure binary-cube compatibility. OFF incorporation has a constructive expressivity advantage (see README), yet fitted performance varies by dataset and search basin.

The generated additive cube and ON/OFF benchmarks have exact solutions by construction. Blind residuals therefore demonstrate an optimizer limitation, not nonrepresentability. Full-size oracle-start validation gives MSE below 1e-20 for both, with the scrambled assignment recovered. Independent small-case random-start recovery, probability conservation, finite-difference gradient checks, and an irreversible-blocking counterexample are tested.

With three bits, per-position fitting improves the binary-cube ON/OFF RMSE from 0.00568 to 1.89e-7, taking about 0.19 seconds versus 0.024 seconds for joint fitting under the short benchmark budget. On the scrambled ON/OFF-generated dataset, both remain near 0.041: optimizing positions independently does not resolve a poor global assignment. These are single-seed comparisons, not a general performance ranking.

The 2–8-round sweep is in results/scaling_quick. Model parameters and mixture counts grow linearly with rounds and sites, while address populations and assignment matrices grow exponentially. Benchmark runtimes exclude export plotting and are hardware-dependent.

Fixed versus flexible assignments, worst-case errors, tolerance fractions, and per-file distributions are recorded in metrics.csv and per_file.csv. All restart losses are retained. Sampling exports cover N=50 through 1000; precision.csv covers 4–256 q/t programming levels. These experiments explore accuracy tradeoffs but do not establish the number of distinguishable datasets. A model-parameter count is not an information-capacity measurement.

At eight rounds the ON/OFF programming manifold has dimension at most 1280 in a 10240-dimensional target space; arbitrary exact encoding is generically impossible even with free assignment. Final-round OFF incorporation is unidentifiable from final ON observations. Physical usefulness requires an application-specific error tolerance, calibrated incorporation precision, and additional noise measurements.

Validation: 49 relevant numerical/UI/RGB synthesis tests pass. The existing test_rgb_targets_are_independent_and_persist test fails because its prepared run list is empty; this also reproduces with the split–pool UI additions removed. The local Python readline extension crashes pytest startup; tests were run with that optional extension disabled.

Reproduce using `python -m split_pool_encoding --config split_pool_encoding/replicated_config.json --output results/replicated --validate-known`. Quick example figures, full parameter archives, reports, and all CSVs are under results/. Generated artifacts are git-ignored; code, configurations, and this report are retained.
