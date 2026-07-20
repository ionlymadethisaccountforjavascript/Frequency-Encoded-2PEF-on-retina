"""
Step 7: Retina Environment Framework
=====================================
Builds the biological substrate (tissue geometry + fluorophore distribution +
optical properties) that Phase 1 excitation/emission simulation samples from.

Fluorophore A -> A2E        (bis-retinoid, accumulates in RPE lipofuscin granules)
Fluorophore B -> Lipofuscin (broader autofluorescent granule population, RPE layer)

Real A2E / lipofuscin two-photon excitation is efficient in the ~730-960nm range,
with lipofuscin showing a broad excitation/emission profile and A2E showing a
blue-shifted, narrower band relative to bulk lipofuscin. We don't hard-code real
absorption spectra here (that would require literature calibration), but we DO
encode the qualitative facts that matter for the pathway-mixing coefficients in
forward_model.py:
  - A2E and lipofuscin heavily spectrally overlap -> single filterless detector
    genuinely cannot separate them without the frequency-multiplexing trick.
  - Both are strongly localized to the RPE layer (not photoreceptors/ganglion
    layers), so their spatial density maps should be concentrated there.
  - Tissue scattering/absorption attenuates both excitation and emitted light,
    which we model as depth-dependent multiplicative factors.
"""

import numpy as np


class RetinaEnvironment:
    """
    Computational grid representing a cross-section (or en-face patch) of
    retina, with per-layer optical properties and per-pixel fluorophore
    densities for two spectrally-overlapping species (A2E, lipofuscin).
    """

    # Layer order from vitreous side -> choroid side (simplified, en-face RPE
    # patch imaging typically targets the RPE/photoreceptor outer segment
    # interface, so we keep this coarse rather than a full 10-layer retina).
    LAYERS = [
        "nerve_fiber_ganglion",
        "inner_plexiform_nuclear",
        "outer_plexiform_nuclear",
        "photoreceptor_segments",
        "rpe",              # <- A2E / lipofuscin live here
        "choroid",
    ]

    def __init__(self, grid_shape=(64, 64), pixel_size_um=1.0, seed=0, layout="granular",
                 separation_um=25.0, blob_sigma_um=4.0):
        """
        layout:
          "granular"          -- original mode: lipofuscin as a granular random
                                  field with A2E as a spatially-correlated
                                  ~10-40% sub-fraction of each granule. Realistic
                                  RPE texture, but the two species are highly
                                  spatially correlated (good for texture
                                  realism, poor for illustrating separation).
          "separated_blobs"   -- matches the reference deck's methodology
                                  (Step 1, slide 9/18): two fluorophores placed
                                  `separation_um` apart, each with an
                                  independent Gaussian PSF footprint
                                  (sigma=`blob_sigma_um`). Spatially distinct
                                  (with partial overlap in between), which is
                                  the standard toy scenario for demonstrating
                                  that conventional TPEF blends overlapping
                                  emitters while frequency encoding separates
                                  them by excitation spectrum.
        """
        self.ny, self.nx = grid_shape
        self.pixel_size_um = pixel_size_um
        self.rng = np.random.default_rng(seed)

        # ---- Tissue geometry -------------------------------------------------
        # For an en-face RPE-focused scan (what a lock-in multiphoton retinal
        # imager actually acquires), we treat the grid as a single en-face plane
        # sitting within the RPE layer, and store a scalar "layer" label plus a
        # depth-within-layer value used for attenuation bookkeeping.
        self.layer_label = np.full(grid_shape, "rpe", dtype=object)
        self.depth_um = self._make_depth_map(grid_shape)

        # ---- Optical properties (scattering / absorption / refractive index) -
        # Values are illustrative, in the right qualitative ballpark for
        # posterior segment tissue at NIR excitation wavelengths, not literature
        # ground truth. mu_s: scattering coeff (1/mm), mu_a: absorption (1/mm).
        self.optical_props = {
            "rpe": dict(mu_s=45.0, mu_a=1.8, n=1.38),
            "photoreceptor_segments": dict(mu_s=30.0, mu_a=0.5, n=1.40),
            "choroid": dict(mu_s=60.0, mu_a=4.0, n=1.37),
        }

        # ---- Fluorophore density maps -----------------------------------------
        self.layout = layout
        if layout == "separated_blobs":
            self.density_A, self.density_B = self._make_fluorophore_maps_blobs(
                grid_shape, separation_um, blob_sigma_um)
        else:
            self.density_A, self.density_B = self._make_fluorophore_maps(grid_shape)

    # -------------------------------------------------------------------
    def _make_depth_map(self, shape):
        """Smoothly varying depth (um) within the RPE layer, used only for
        attenuation weighting -- not a full 3D volume."""
        y = np.linspace(-1, 1, shape[0])
        x = np.linspace(-1, 1, shape[1])
        yy, xx = np.meshgrid(y, x, indexing="ij")
        base = 10.0 + 2.0 * np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
        base += self.rng.normal(0, 0.3, size=shape)
        return np.clip(base, 5.0, 15.0)

    def _make_fluorophore_maps(self, shape):
        """
        Simulate density profiles of A2E and lipofuscin within RPE.

        Biologically: lipofuscin granules accumulate with age across most of
        the RPE mosaic (fairly broad/uniform-ish with granular texture), while
        A2E is a specific constituent *within* lipofuscin granules and tends to
        track a subset of that same granular pattern rather than being
        spatially independent. We model B (lipofuscin) as a granular random
        field, and A (A2E) as a correlated-but-not-identical fraction of it,
        which is the realistic "these two are hard to unmix spatially, only
        their nonlinear frequency-mixing signatures differ" scenario that
        motivates the whole frequency-multiplexed approach.
        """
        ny, nx = shape

        # Granule centers (RPE cells are roughly hexagonally packed, ~14um
        # apart; we approximate with a jittered grid).
        spacing = max(3, min(ny, nx) // 12)
        centers = []
        for gy in range(0, ny, spacing):
            for gx in range(0, nx, spacing):
                jy = gy + self.rng.normal(0, spacing * 0.15)
                jx = gx + self.rng.normal(0, spacing * 0.15)
                centers.append((jy, jx))
        centers = np.array(centers)

        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        lipofuscin = np.zeros(shape)
        a2e = np.zeros(shape)

        for (cy, cx) in centers:
            r2 = (yy - cy) ** 2 + (xx - cx) ** 2
            sigma = spacing * self.rng.uniform(0.35, 0.55)
            granule = np.exp(-r2 / (2 * sigma ** 2))
            granule_intensity = self.rng.gamma(shape=2.0, scale=1.0)  # age/load variability
            lipofuscin += granule_intensity * granule
            # A2E fraction of each granule varies granule-to-granule (~10-40%
            # of lipofuscin fluorescence is attributable to A2E-like bisretinoids
            # per typical RPE biochemistry summaries) plus independent noise so
            # it is correlated but not a fixed multiple of B.
            a2e_fraction = self.rng.uniform(0.10, 0.40)
            a2e += granule_intensity * a2e_fraction * granule * self.rng.uniform(0.7, 1.3)

        # normalize to [0, 1] density units
        lipofuscin = lipofuscin / (lipofuscin.max() + 1e-9)
        a2e = a2e / (a2e.max() + 1e-9)

        return a2e, lipofuscin

    def _make_fluorophore_maps_blobs(self, shape, separation_um, blob_sigma_um):
        """
        Reference-deck methodology (Step 1): two fluorophores placed
        `separation_um` apart, each rendered as a Gaussian PSF footprint of
        width `blob_sigma_um`, centered in the grid. Converts um -> pixels
        using self.pixel_size_um. A2E and lipofuscin get independent
        granular brightness texture within their own footprint (not derived
        from each other), so the two channels are spatially distinct while
        still both being present in the shared central overlap region --
        exactly the "overlapping emitters" scenario conventional TPEF can't
        resolve but frequency encoding can.
        """
        ny, nx = shape
        sep_px = separation_um / self.pixel_size_um
        sigma_px = blob_sigma_um / self.pixel_size_um

        cy, cx = ny / 2.0, nx / 2.0
        cA = (cy, cx - sep_px / 2.0)
        cB = (cy, cx + sep_px / 2.0)

        yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")

        def gaussian_blob(center, sigma, rng, texture=True):
            r2 = (yy - center[0]) ** 2 + (xx - center[1]) ** 2
            blob = np.exp(-r2 / (2 * sigma ** 2))
            if texture:
                # subtle granular texture within the footprint, independent
                # per fluorophore so the two maps aren't scalar multiples
                noise = rng.gamma(shape=4.0, scale=0.25, size=shape)
                blob = blob * (0.7 + 0.3 * noise)
            return blob

        rngA = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        rngB = np.random.default_rng(self.rng.integers(0, 2**31 - 1))

        a2e = gaussian_blob(cA, sigma_px, rngA, texture=False)
        lipofuscin = gaussian_blob(cB, sigma_px * 1.15, rngB, texture=False)  # slightly broader footprint

        a2e = a2e / (a2e.max() + 1e-9)
        lipofuscin = lipofuscin / (lipofuscin.max() + 1e-9)
        return a2e, lipofuscin

    # -------------------------------------------------------------------
    def attenuation(self, layer="rpe"):
        """
        Simple depth-dependent excitation/emission attenuation using
        Beer-Lambert-like exponential decay from scattering+absorption.
        Returns a (ny, nx) multiplicative attenuation map in [0,1].
        """
        props = self.optical_props[layer]
        mu_t_per_mm = props["mu_s"] + props["mu_a"]  # total attenuation coeff
        depth_mm = self.depth_um / 1000.0
        return np.exp(-mu_t_per_mm * depth_mm)

    def summary(self):
        return {
            "grid_shape": (self.ny, self.nx),
            "pixel_size_um": self.pixel_size_um,
            "mean_depth_um": float(self.depth_um.mean()),
            "mean_attenuation_rpe": float(self.attenuation("rpe").mean()),
            "density_A_range": (float(self.density_A.min()), float(self.density_A.max())),
            "density_B_range": (float(self.density_B.min()), float(self.density_B.max())),
        }
