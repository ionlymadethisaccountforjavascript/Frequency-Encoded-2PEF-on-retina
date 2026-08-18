# Function guide

Every source function has a detailed in-code docstring. This index collects those explanations in one place.

## `src/fe2pef_retina/baselines.py`

### `load_emission_matrix` (line 15)

Load a detector-by-species emission mixing matrix from CSV.

What happens in this function:
1. Long-format rows are read with detector, species, response, and provenance.
2. Requested species are kept in the exact project order.
3. A pivot table forms detector rows and species columns.
4. Nonnegative finite responses are validated before return.

### `conventional_single_detector_image` (line 46)

Create an unfiltered conventional TPEF intensity image.

What happens in this function:
1. Each concentration map is multiplied by its relative brightness.
2. Species contributions are summed because one detector cannot label their origin.
3. The result is useful for visualization but cannot uniquely recover species maps.

### `simulate_emission_filtered_channels` (line 65)

Simulate conventional multi-detector emission-filtered TPEF.

What happens in this function:
1. Emission filters mix species into detector channels through a response matrix.
2. Optical throughput reduces signal to represent filter and beam-splitter losses.
3. Temporal averaging produces one mean count estimate per detector and pixel.
4. Shot noise, background shot noise, and read noise are approximated analytically.
5. Known mean background may be subtracted while stochastic noise remains.

## `src/fe2pef_retina/cli.py`

### `build_parser` (line 12)

Build the command-line parser and its experiment subcommands.

What happens in this function:
1. A top-level parser is created for the ``retinafe`` command.
2. Demo, validation, sweep, and all-in-one subcommands are registered.
3. Shared config and output arguments are defined consistently.
4. The parser is returned for use by ``main`` and tests.

### `main` (line 43)

Execute the selected RetinaFE experiment from command-line arguments.

What happens in this function:
1. Arguments are parsed and the output directory is selected.
2. The requested experiment function is called.
3. ``all`` writes each experiment into a dedicated subdirectory.
4. A JSON summary is printed for scripts and continuous integration.
5. Zero is returned on successful completion.

## `src/fe2pef_retina/config.py`

### `validate` (line 189)

Validate physical and numerical settings before a simulation starts.

What happens in this function:
1. It checks positive grid, time, frequency, and power values.
2. It checks that modulation depths and overlap lie in valid intervals.
3. It checks that enough modulation cycles occur during one pixel dwell.
4. It emits warnings for weak experimental designs instead of hiding them.

### `_tupleify` (line 236)

Convert YAML lists into tuples where immutable dataclasses expect tuples.

What happens in this function:
1. Lists are recursively converted into tuples.
2. Dictionaries are recursively traversed without changing their keys.
3. Scalar values pass through unchanged.

### `load_config` (line 252)

Load a YAML experiment file into validated nested dataclasses.

What happens in this function:
1. YAML is parsed from disk and lists are converted to tuples.
2. Each nested dictionary is passed to its matching dataclass constructor.
3. The resulting project configuration is validated before it is returned.

## `src/fe2pef_retina/experiments.py`

### `_resolve_project_path` (line 51)

Resolve a configuration path relative to the project root.

What happens in this function:
1. Absolute paths are returned unchanged.
2. Relative paths are joined to the supplied project root.
3. The result is resolved for stable provenance output.

### `_make_phantom` (line 64)

Generate the concentration maps requested by an experiment configuration.

What happens in this function:
1. The configured phantom kind selects a dedicated generator.
2. Morphology parameters are forwarded without hidden changes.
3. A metadata dictionary records labels or generator assumptions.
4. Current built-in generators produce two species; user maps can extend this later.

### `_simulate_fe` (line 111)

Dispatch FE simulation to analytic or direct time-domain mode.

What happens in this function:
1. The configuration's simulation mode is inspected.
2. Analytic mode uses lock-in estimator variance for fast sweeps.
3. Time-domain mode creates noisy detector waveforms in pixel chunks.
4. Both modes return channel arrays with identical axis conventions.

### `run_demo` (line 154)

Run the main retinal FE-2PEF reconstruction comparison.

What happens in this function:
1. Configuration, response tables, and a synthetic retinal phantom are loaded.
2. A Gaussian PSF blurs each species concentration map.
3. FE lock-in channels are simulated and unmixed by calibrated NNLS and blind NNMF.
4. A conventional emission-filtered baseline is simulated at reduced throughput.
5. Images, signatures, arrays, metrics, diagnostics, and provenance are saved.
6. The returned dictionary summarizes key output paths and condition numbers.

### `run_channel_validation` (line 352)

Validate the analytic FE signature against direct waveform demodulation.

What happens in this function:
1. Pure constant maps isolate one species at a time.
2. Noiseless time-domain waveforms are generated and digitally demodulated.
3. Analytic signature predictions are converted into the same count units.
4. Relative channel errors and finite-window reference leakage are saved.
5. This unit-style validation checks implementation, not biological accuracy.

### `_mean_component_nrmse` (line 452)

Return mean scale-adjusted NRMSE across all species.

What happens in this function:
1. Each estimated species map receives its best nonnegative scalar fit.
2. NRMSE is calculated against the corresponding reference map.
3. Species errors are averaged into one sweep-friendly number.

### `run_robustness_sweeps` (line 468)

Run the complete FE-versus-emission robustness study.

What happens in this function:
1. A small two-species phantom keeps the sweeps computationally affordable.
2. Additive background is varied to compare FE and emission filtering.
3. Spot separation tests spatial overlap without claiming super-resolution.
4. Brightness imbalance tests the shared-detector dynamic-range limitation.
5. FE-signature and emission-signature similarity reveal conditioning failures.
6. Photon budget, detector bandwidth, pulse overlap, and filter throughput are swept.
7. Every replicate is saved to CSV and summarized by uncertainty plots.

### `evaluate` (line 507)

Simulate and score calibrated FE and emission reconstructions.

What happens in this nested function:
1. Optional stress-test values override pulse overlap, emission mixing, or throughput.
2. FE signatures are rebuilt for the current response and detector settings.
3. Fast analytic noise simulation generates FE and emission-filtered channels.
4. Calibrated NNLS recovers both sets of species maps under matched detector noise.
5. Mean NRMSE and the FE signature condition number are returned.

## `src/fe2pef_retina/forward.py`

### `convolve_concentration_maps` (line 17)

Convolve each species map with a Gaussian detection/excitation PSF.

What happens in this function:
1. Spatial PSF widths are converted from micrometres to array pixels.
2. Every species is blurred independently to prevent artificial cross-species mixing.
3. Two-dimensional and three-dimensional maps are both supported.
4. The output retains the same shape and total-array convention as the input.

### `effective_signature_with_detector` (line 47)

Apply detector bandwidth attenuation to an FE signature matrix.

What happens in this function:
1. Each named channel is mapped to its physical frequency.
2. A first-order detector gain is evaluated at that frequency.
3. Each signature row is multiplied by its channel gain.
4. The result is the matrix that should be used for calibrated inversion.

### `simulate_fe_channels_analytic` (line 79)

Simulate demodulated FE channels with a lock-in noise approximation.

What happens in this function:
1. The effective channel signature is multiplied by every spatial concentration vector.
2. Relative fluorescence is converted to detected counts per temporal sample.
3. DC signal sets the Poisson shot-noise level for every lock-in channel.
4. Gaussian lock-in estimates are drawn using analytic estimator variance.
5. Known mean background can be subtracted, while its shot noise remains.

### `simulate_fe_channels_time_domain` (line 146)

Simulate the detector waveform and digital lock-in projection directly.

What happens in this function:
1. Species maps are converted into three pathway-amplitude maps: 11, 12, and 22.
2. Two intensity-modulated laser waveforms are evaluated across one pixel dwell.
3. The quadratic fluorescence waveform is built for pixel chunks to limit memory.
4. Optional Poisson shot noise and Gaussian read noise are added before demodulation.
5. Digital lock-in references recover DC, fundamentals, and mixed-frequency channels.
6. Detector bandwidth attenuation is applied to the demodulated amplitudes.

## `src/fe2pef_retina/io.py`

### `ensure_directory` (line 14)

Create an output directory and return it as a Path.

What happens in this function:
1. The supplied path is converted to ``Path``.
2. Missing parent directories are created.
3. Existing directories are accepted without error.

### `_json_ready` (line 28)

Convert dataclasses, arrays, paths, and tuples into JSON-compatible values.

What happens in this function:
1. Dataclasses are expanded into dictionaries.
2. Paths and NumPy scalar types are converted to plain Python values.
3. Containers are recursively traversed.
4. Remaining scalar values pass through unchanged.

### `save_json` (line 53)

Write a human-readable JSON file with stable formatting.

What happens in this function:
1. Parent directories are created if needed.
2. The payload is converted into JSON-compatible values.
3. UTF-8 text is written with indentation and sorted keys.

### `save_array_bundle` (line 68)

Save named NumPy arrays in a compressed NPZ archive.

What happens in this function:
1. Parent directories are created.
2. Every value is converted to a NumPy array.
3. Compressed storage preserves exact numeric outputs for reproducibility.

### `save_rows_csv` (line 82)

Write a list of metric or sweep rows to CSV.

What happens in this function:
1. Parent directories are created.
2. Dictionaries become a pandas DataFrame.
3. Rows are written without an extra index column.

## `src/fe2pef_retina/lockin.py`

### `reference_waveform` (line 12)

Generate the phase-aware reference for one lock-in channel.

What happens in this function:
1. Laser modulation phases are reconstructed from frequency and phase settings.
2. Fundamental channels use sine references matching the intensity tags.
3. Mixed and harmonic channels use cosine references from product identities.
4. Signs are selected so ideal pathway amplitudes are positive.

### `build_reference_matrix` (line 47)

Build normalized linear weights for simultaneous lock-in demodulation.

What happens in this function:
1. One reference waveform is generated for every requested channel.
2. DC weights perform a temporal mean.
3. AC weights use twice the reference divided by sample count.
4. Matrix multiplication then demodulates many pixels simultaneously.

### `demodulate_time_series` (line 75)

Demodulate one or many temporal fluorescence signals.

What happens in this function:
1. The final signal axis is interpreted as time.
2. Normalized lock-in weights are generated for all channels.
3. A tensor contraction projects each signal onto every reference.
4. The first output axis indexes channels.

### `orthogonality_matrix` (line 98)

Measure finite-window orthogonality of the requested references.

What happens in this function:
1. Raw reference waveforms are assembled as matrix rows.
2. Every row is normalized to unit norm.
3. The Gram matrix reveals leakage caused by insufficient or noninteger cycles.
4. An ideal acquisition has an approximately identity Gram matrix.

## `src/fe2pef_retina/metrics.py`

### `normalize_map` (line 9)

Normalize one nonnegative image to the interval [0, 1].

What happens in this function:
1. Nonfinite values are replaced with zero.
2. The minimum is removed to avoid negative display offsets.
3. The map is divided by its maximum when the maximum is nonzero.

### `nrmse` (line 24)

Compute root-mean-square error normalized by reference RMS.

What happens in this function:
1. Arrays are converted to floating point and shape equality is checked.
2. RMS error is divided by the RMS magnitude of the reference.
3. A small denominator protects all-zero reference maps.

### `pearson_correlation` (line 42)

Compute Pearson correlation between flattened image intensities.

What happens in this function:
1. Images are flattened and mean-centred.
2. Their dot product is divided by the product of norms.
3. Constant images return zero instead of an undefined value.

### `ssim` (line 59)

Compute structural similarity after independent [0, 1] normalization.

What happens in this function:
1. Both images are normalized to remove arbitrary NMF scale.
2. The data range is fixed to one.
3. Structural similarity evaluates spatial fidelity beyond pixel-wise error.

### `centroid_um` (line 73)

Calculate the intensity-weighted centroid in physical coordinates.

What happens in this function:
1. Negative intensities are clipped because concentration is nonnegative.
2. Grid indices are weighted by image intensity.
3. Coordinates are centred and converted from pixels to micrometres.
4. An all-zero image returns NaN coordinates.

### `localization_error_um` (line 93)

Return Euclidean distance between reference and estimated centroids.

What happens in this function:
1. Physical centroids are calculated for both images.
2. Their coordinate difference is measured with the Euclidean norm.
3. Failed all-zero centroids return infinity to flag reconstruction failure.

### `support_crosstalk` (line 113)

Measure leakage of another component into a reference component's support.

What happens in this function:
1. The reference support is defined by values above 20% of its maximum.
2. Mean contaminating signal inside and outside that support is calculated.
3. Their ratio reports how strongly the other component leaks into the region.
4. The metric is heuristic and should be accompanied by NRMSE and correlation.

### `component_metrics` (line 133)

Calculate a standard metric row for every recovered species map.

What happens in this function:
1. Reference and estimated component counts are checked.
2. Estimated scale is fitted by nonnegative least squares in closed form.
3. NRMSE, correlation, SSIM, and localization error are calculated.
4. Rows are returned as dictionaries ready for a CSV table.

## `src/fe2pef_retina/noise.py`

### `first_order_detector_gain` (line 8)

Return amplitude attenuation of a first-order low-pass detector.

What happens in this function:
1. DC is passed without attenuation.
2. The frequency-to-bandwidth ratio is formed.
3. The standard first-order magnitude response is returned.

### `poisson_gaussian_samples` (line 22)

Draw detector samples with shot noise, read noise, and optional clipping.

What happens in this function:
1. Negative expectations are clipped because Poisson rates cannot be negative.
2. Poisson samples model photon-counting shot noise.
3. Gaussian samples model additive read/electronic noise.
4. Optional saturation clips the final detector output.

### `lockin_noise_std` (line 46)

Approximate noise of an orthogonal lock-in estimator.

What happens in this function:
1. Poisson variance is approximated by the mean count per temporal sample.
2. Read-noise variance is added independently.
3. Averaging over the dwell reduces variance by the sample count.
4. AC quadrature projection introduces a factor of two relative to DC averaging.

## `src/fe2pef_retina/optics.py`

### `photon_energy_joule` (line 13)

Return the energy of one photon at the requested wavelength.

What happens in this function:
1. Nanometres are converted to metres.
2. The relation E = h*c/lambda is evaluated.
3. The result is returned in joules per photon.

### `effective_one_photon_wavelength_nm` (line 28)

Convert a two-photon pathway into its one-photon-equivalent wavelength.

What happens in this function:
1. If one wavelength is supplied twice, a degenerate two-photon pathway is used.
2. Photon energies are added through reciprocal wavelengths.
3. The equivalent one-photon wavelength is returned for spectral interpretation.

### `gaussian_pulse_peak_power_w` (line 47)

Estimate peak power for a Gaussian pulse train.

What happens in this function:
1. Average power is divided by repetition rate to obtain pulse energy.
2. Pulse FWHM is converted from femtoseconds to seconds.
3. A Gaussian-area factor converts pulse energy to peak power.
4. The result is an approximation and does not include optical losses.

### `relative_two_photon_dose` (line 69)

Return a relative two-photon excitation dose for scaling studies.

What happens in this function:
1. Peak power is estimated from the pulsed laser parameters.
2. A Gaussian focal area is computed from the supplied spatial sigma.
3. The square of peak intensity is multiplied by pulse duty time.
4. The value is relative; absolute fluorescence needs calibrated cross-sections.

### `gaussian_sigma_from_fwhm` (line 98)

Convert a Gaussian full width at half maximum into standard deviation.

What happens in this function:
1. The analytic Gaussian conversion factor is evaluated.
2. The result uses the same units as the supplied FWHM.

### `approximate_2pef_lateral_fwhm_um` (line 111)

Estimate lateral 2PEF FWHM with a commonly used diffraction approximation.

What happens in this function:
1. The wavelength is converted from nanometres to micrometres.
2. A 0.37*lambda/NA approximation is evaluated.
3. The result is intended for sensitivity analysis, not safety certification.

### `approximate_2pef_axial_fwhm_um` (line 125)

Estimate axial 2PEF FWHM with a paraxial approximation.

What happens in this function:
1. Wavelength is converted into micrometres.
2. Refractive index and numerical aperture set the axial scale.
3. The approximation should be replaced by measured PSF data when available.

## `src/fe2pef_retina/phantoms.py`

### `_coordinate_grid` (line 12)

Create centred physical-coordinate arrays for a spatial grid.

What happens in this function:
1. Each array index is shifted so the field centre is zero.
2. Pixel spacing converts coordinates into micrometres.
3. ``meshgrid`` returns one coordinate array per spatial dimension.

### `gaussian_spot` (line 28)

Generate an N-dimensional Gaussian concentration spot.

What happens in this function:
1. A physical coordinate grid is created in micrometres.
2. Squared distance from the requested centre is accumulated.
3. A Gaussian profile is evaluated and scaled by amplitude.

### `two_spot_phantom` (line 54)

Create two species represented by separated Gaussian spots.

What happens in this function:
1. Two centres are placed symmetrically around the field centre.
2. Each centre becomes one species-specific Gaussian concentration map.
3. The output has shape ``(2, *shape)`` for direct use by the forward model.

### `overlap_phantom` (line 77)

Create two spatially overlapping but non-identical concentration maps.

What happens in this function:
1. Smooth random fields create a shared biological-looking texture.
2. Independent smooth fields create species-specific structure.
3. ``overlap_fraction`` mixes shared and independent components.
4. Each output map is normalized to a maximum of one.

### `rpe_mosaic_phantom` (line 107)

Generate an RPE-like cell mosaic with granular and diffuse species maps.

What happens in this function:
1. Random seed points define Voronoi-like polygonal RPE cells.
2. A2E-like signal is represented by intracellular granular deposits.
3. FAD-like signal is represented by diffuse, cell-varying cytoplasmic signal.
4. Shared morphology introduces configurable spatial overlap.
5. The generator is a morphology prior, not measured molecular ground truth.

### `rpe_volume_phantom` (line 182)

Create a small 3-D two-species retinal proof-of-concept volume.

What happens in this function:
1. A 2-D RPE mosaic is created as the lateral morphology.
2. Species are assigned different axial Gaussian profiles.
3. The maps are normalized and returned as ``(species, z, y, x)``.
4. This volume demonstrates software scaling, not a validated retinal atlas.

### `load_concentration_maps` (line 215)

Load user-supplied concentration maps from NPY or NPZ files.

What happens in this function:
1. NPY arrays are loaded directly.
2. NPZ archives must contain a ``concentration_maps`` array.
3. The first dimension is interpreted as species.
4. Values are checked for finite nonnegative concentrations.

## `src/fe2pef_retina/plotting.py`

### `_display_projection` (line 15)

Convert a scalar 2-D or 3-D field into a displayable 2-D image.

What happens in this function:
1. A two-dimensional input is returned without modification.
2. A three-dimensional volume is maximum-intensity projected along its first axis.
3. Other dimensionalities are rejected so plotting errors are explicit.
4. The numerical arrays saved by experiments remain unchanged; only the figure is projected.

### `save_image_grid` (line 33)

Save a compact grid of independently scaled scalar images.

What happens in this function:
1. The number of rows is calculated from image count and requested columns.
2. Each image is normalized for morphology-focused visual comparison.
3. Unused axes are hidden and optional color bars are added.
4. The figure is saved with tight bounding and then closed.

### `save_signature_heatmap` (line 70)

Save a labelled heatmap of a channel-by-species signature matrix.

What happens in this function:
1. Matrix values are displayed without column normalization.
2. Channel and species labels are placed on the axes.
3. Numeric values are printed in every cell for auditability.
4. A color bar and tight layout are added before saving.

### `save_metric_bar_chart` (line 104)

Save a method comparison bar chart using mean species NRMSE.

What happens in this function:
1. Metric rows are grouped by reconstruction method.
2. Mean NRMSE and standard deviation across species are calculated.
3. A single bar chart summarizes lower-is-better reconstruction error.
4. The underlying values remain available in the companion CSV file.

### `save_sweep_plot` (line 127)

Save one line plot for a robustness sweep with replicate uncertainty.

What happens in this function:
1. Replicate rows are grouped by method and x-axis value.
2. Mean and standard deviation of the requested metric are calculated.
3. Each method is drawn as a separate line with an uncertainty band.
4. The plot is saved and the figure is closed to avoid memory leakage.

## `src/fe2pef_retina/spectra.py`

### `load_pathway_responses` (line 23)

Load normalized two-colour pathway responses for selected species.

What happens in this function:
1. A CSV table is read and required columns are checked.
2. Requested species are selected in the exact requested order.
3. Nonnegative and finite response values are enforced.
4. Provenance remains attached so illustrative values cannot be mistaken for data.

### `pathway_response_matrix` (line 54)

Convert a response table into a pathways-by-species matrix.

What happens in this function:
1. Columns sigma_11, sigma_12, and sigma_22 are extracted.
2. Species brightness multiplies every pathway for that species.
3. The returned matrix has rows [11, 12, 22] and columns as species.

### `channel_frequency_hz` (line 68)

Map a named lock-in channel to its physical modulation frequency.

What happens in this function:
1. Fundamental, harmonic, sum, and difference names are recognized.
2. The corresponding nonnegative frequency is calculated.
3. DC maps to zero hertz.

### `build_frequency_signature_matrix` (line 94)

Build the analytic FE-2PEF channel-by-species mixing matrix.

What happens in this function:
1. Each species contributes to I1^2, I1*I2, and I2^2 pathways.
2. Sinusoidal laser modulation is expanded into DC and sideband terms.
3. Laser powers, modulation depths, and pulse overlap scale each coefficient.
4. Rows follow the requested lock-in channels and columns follow species order.
5. Signs are chosen to match the phase-aware references in ``lockin.py``.

### `condition_number` (line 138)

Return the 2-norm condition number of a mixing matrix.

What happens in this function:
1. The matrix is converted to floating point.
2. Singular values are used through NumPy's condition-number routine.
3. Large values flag fluorophore signatures that are hard to distinguish.

### `normalize_signature_columns` (line 151)

Normalize each fluorophore signature to unit Euclidean norm.

What happens in this function:
1. A norm is calculated for every species column.
2. Zero columns are rejected because they cannot be unmixed.
3. Normalization removes arbitrary brightness scale for geometry comparisons.

## `src/fe2pef_retina/unmixing.py`

### `calibrated_nnls_unmix` (line 28)

Recover nonnegative concentration maps using a known signature matrix.

What happens in this function:
1. Spatial dimensions are flattened while channel order is preserved.
2. SciPy's nonnegative least-squares solver is applied independently per pixel.
3. Negative concentrations are forbidden by the optimizer rather than clipped later.
4. Recovered maps are reshaped to ``(species, *spatial_shape)``.

### `_nmf_objective` (line 52)

Evaluate the regularized NMF objective used for model selection.

What happens in this function:
1. Frobenius reconstruction error measures data mismatch.
2. L1 concentration penalty promotes sparse fluorophore maps.
3. L2 concentration penalty stabilizes ill-conditioned inversion.
4. The three nonnegative terms are added into one scalar objective.

### `regularized_nmf` (line 76)

Factor nonnegative FE channels into signatures and concentration maps.

What happens in this function:
1. Signal values are clipped at zero because NNMF requires nonnegative data.
2. Random positive signatures and concentrations initialize each restart.
3. Multiplicative updates follow the regularized equations used by Heuke et al.
4. Signature columns are normalized after each update to control scale ambiguity.
5. Convergence is checked through relative objective improvement.
6. The lowest-objective restart is returned.

### `align_components` (line 171)

Align arbitrary NMF component order to known reference maps for evaluation.

What happens in this function:
1. Recovered and reference maps are flattened and mean-centred.
2. Absolute correlation becomes the component-assignment score.
3. The Hungarian algorithm finds the maximum-total-correlation permutation.
4. Reordered maps and the permutation indices are returned.
5. This alignment is only for benchmarking; real blind data lack ground truth.

### `reshape_nmf_concentrations` (line 203)

Reshape flattened NMF concentration vectors into spatial maps.

What happens in this function:
1. The expected pixel count is calculated from the requested spatial shape.
2. The NMF concentration matrix is checked against that pixel count.
3. Components are reshaped while preserving their first-axis order.

## `scripts/prepare_external_maps.py`

### `normalize_species_maps` (line 11)

Normalize each nonnegative species map independently to a maximum of one.

What happens in this function:
1. The first array axis is interpreted as fluorophore species.
2. Nonfinite or negative values are rejected.
3. Each species is divided by its own maximum when nonzero.
4. Relative concentration patterns are retained while absolute units are discarded.

### `main` (line 32)

Read NPY maps, normalize them, and write a standard NPZ archive.

What happens in this function:
1. Input and output file arguments are parsed.
2. The input array is loaded from NPY format.
3. Species maps are validated and normalized.
4. The output archive uses the key ``concentration_maps``.

## `scripts/run_all.py`

### `main` (line 16)

Execute every included analysis using the default configurations.

What happens in this function:
1. Channel-equation validation is run first.
2. The main retinal reconstruction comparison is run second.
3. Robustness sweeps are run with six random replicates per condition.
4. Output subdirectories are printed for inspection.

## `scripts/run_demo.py`

### `main` (line 12)

Run the default demo and print the output directory.

What happens in this function:
1. Project-relative config and output paths are constructed.
2. The end-to-end demo experiment is executed.
3. A compact completion message is printed.
4. Zero is returned for shell compatibility.
