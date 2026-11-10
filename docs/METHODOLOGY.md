# Methodology audit and validation

## Preserved experiment
Original MLP/LSTM architecture, Adam/MSE, input length 24, horizon 6, step 1, seed 42, grid, five-fold GroupKFold inner selection, LOPO outer evaluation and persistence baseline are retained. Inner calls use 15 epochs and patience 2; outer calls use 25 and patience 3. Historical results come from saved outputs, not new training. TensorFlow 2.19.0 is explicitly recorded in source output. The supporting Word document overstates gap removal and absence of leakage; code takes precedence.

## Historical versus corrected modes
Historical mode retains exact reindexing, partial interpolation within long gaps, NaN dropping, index-based windows with one omitted final start, random 90/10 final window validation and scaling before that split. The held-out outer patient remains excluded from fitting and selection. Corrected mode preserves observations on the union of real timestamps and grid, interpolates only missing runs of six or fewer grid points, retains long-gap NaNs, verifies five-minute continuity over every input-to-target span, and includes the final valid start. Empty windows have stable (0,24,1)/(0,1) shapes. Final validation uses the first five-fold GroupKFold split of training patients; scaling fits only its fitting windows. A held-out patient without corrected windows raises an explicit error instead of generating metrics.

Off-grid records are interpolation anchors rather than exact grid samples. Strict gap handling can greatly reduce usable data; fewer than five eligible training patients will make GroupKFold fail and requires reviewing eligibility, not inventing predictions. Retrospective interpolation can still use future observations and is not a causal deployment pipeline. Scaling statistics weight repeated values in overlapping windows, matching the original approach.

The new baseline summary retains full precision until printing instead of rounding patient metrics before aggregation; historical saved numbers remain unchanged. Prediction lengths are asserted equal instead of silently truncated in aggregate baseline/error analysis. Mean and sample SD are across patient metrics, not pooled errors. Baseline per-patient numerical results are not saved, so all-patient superiority cannot be verified. The variation plot conditions on persistence error itself, omits variation >=80 mg/dL, and provides no clinical validation.

## Data handling
No NumPy recordings, raw timestamp previews or individual glucose trajectories are included. Only the aggregate error figure is exported unchanged. Source patient labels are array positions. Dataset origin, participant consent and redistribution permissions remain undocumented. No licence was added and no publication was performed.

## Checks performed
All code cells parse as Python; notebook JSON, cell fields and local figure references were checked. Synthetic checks passed for exactly one valid 30-sample window, target six steps after the last input, NaNs between input and target blocking a window, complete interpolation of six missing grid points and no partial fill of a seven-point gap. Synthetic test data are not scientific results. Original model and baseline summary values and all 12 selected MLP configurations were checked in saved outputs.

Not performed: full LOPO execution, original-data preprocessing audit, corrected metrics, clean dependency installation or independent clinical evaluation. Import checks are recorded separately. Historical metrics must never be reported as corrected-mode metrics. Before publication, verify dataset provenance, participant/data-sharing permissions and the licence decision.

Reference for interpolation semantics: [pandas documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html).

Import checks passed for TensorFlow, NumPy, pandas, scikit-learn and matplotlib. Installed TensorFlow is 2.20.0, different from the historical 2.19.0 requirement; the specified clean environment was not installed or validated.
