
"""FE-2PEF demo runner.

Steps:
- Build retina plane (from retinasim if available or synthetic)
- Define fluorophore distributions (several species localized in different patches)
- Define two lasers (omega1, omega2) with modulation frequencies f1,f2
- Generate time traces and per-pixel 2PEF signals for FE approach and for emission-discrimination (spectral filters)
- Demodulate FE signals and unmix via NNMF
- Calculate simple SNR/throughput metrics and save plots in output/
"""
import os, numpy as np, matplotlib.pyplot as plt
from src.fe2pef import optics, aotf_sim, demod, nnmf_demo, stimulation_wrapper as stim

OUT = os.path.join(os.getcwd(), 'output')
os.makedirs(OUT, exist_ok=True)

def two_photon_yield(power_mW, cross_section=1.0):
    """Simple model: 2PEF roughly scales with P^2 (instantaneous) and a cross-section.
    power_mW can be time-varying; for spatial weighting, we'll multiply by local concentration.
    """
    return cross_section * (power_mW**2)

def simulate_frame(retina, channels, t, pixel_size_um=0.5):
    # retina: 2D array of per-pixel concentrations for each fluor (shape: n_species x H x W)
    # channels: list of AOTFChannel
    # t: time vector (1D)
    # returns: simulated detector trace per pixel (H x W x len(t))
    n_species = retina.shape[0]
    H, W = retina.shape[1:]
    powers = channels.powers_at(t)  # shape: (n_channels, len(t))
    # For demo: each species has a simple excitation sensitivity w.r.t channel wavelength (matrix)
    # Create a replaceable excitation matrix (n_species x n_channels)
    # Example: species 0 is more sensitive to channel 0 etc.
    exc_matrix = np.array([[1.0, 0.2, 0.0],
                           [0.1, 1.0, 0.3],
                           [0.0, 0.3, 1.0]])
    n_channels = powers.shape[0]
    # Clamp matrix size
    if exc_matrix.shape[1] < n_channels:
        # pad with small sensitivity
        pad = np.zeros((n_species, n_channels - exc_matrix.shape[1])) + 0.05
        exc_matrix = np.hstack([exc_matrix, pad])
    # Compute instantaneous yield per pixel & time
    detector = np.zeros((H, W, len(t)), dtype=float)
    for s in range(n_species):
        sens = exc_matrix[s, :n_channels]  # length n_channels
        # combined power for species: weighted sum over channels (then squared for two-photon reaction)
        # approximate instantaneous excitation power per species as sum(c_i * P_i(t))
        inst_power = np.tensordot(sens, powers, axes=(0,0))  # shape (len(t),)
        # yield per pixel = concentration * two_photon_yield(inst_power)
        yield_t = two_photon_yield(inst_power)  # len(t)
        # Multiply by spatial concentration map
        for ti in range(len(t)):
            detector[:, :, ti] += retina[s] * yield_t[ti]
    return detector

def main():
    # Build retina
    rp = stim.RetinaProvider(use_retinasim=True)
    base = rp.get_plane(shape=(128,128))  # base brightness
    H, W = base.shape
    # Create three fluorophore species with spatial localization
    species_maps = np.zeros((3, H, W))
    # species 0 centered on top-left, 1 on center, 2 on bottom-right (modulated overlaps)
    species_maps[0] = np.roll(base.copy(), shift=( -20, -20), axis=(0,1)) * 1.0
    species_maps[1] = np.roll(base.copy(), shift=(0, 0), axis=(0,1)) * 0.9
    species_maps[2] = np.roll(base.copy(), shift=(20, 20), axis=(0,1)) * 0.7

    # Simple optics: PSF blur on excitation (two-photon focal volume)
    psf = optics.gaussian_psf(size=31, sigma_um=1.2, pixel_size_um=1.0)
    for s in range(species_maps.shape[0]):
        species_maps[s] = optics.apply_scattering(species_maps[s])
    # define lasers / channels
    ch1 = aotf_sim.AOTFChannel('ch470', wavelength_nm=470, max_power_mW=80.0, mod_freq_hz=12000.0)
    ch2 = aotf_sim.AOTFChannel('ch640', wavelength_nm=640, max_power_mW=80.0, mod_freq_hz=15000.0)
    ch3 = aotf_sim.AOTFChannel('ch561', wavelength_nm=561, max_power_mW=60.0, mod_freq_hz=18000.0)
    sim = aotf_sim.AOTFSimulator([ch1, ch2, ch3])

    # time vector for one acquisition block
    fs = 200000.0  # sampling frequency (Hz)
    T = 0.02       # seconds per block (short for demo)
    t = np.arange(0, T, 1.0/fs)

    detector = simulate_frame(species_maps, sim, t)
    # Collapse time by summing to get unmodulated image (this is what emission-discrimination would collect)
    unmodulated = detector.sum(axis=2)
    # Demodulate at channel frequencies
    pixel_freqs = [ch1.mod_freq_hz, ch2.mod_freq_hz, ch3.mod_freq_hz]
    Himg = H*W
    demod_maps = np.zeros((len(pixel_freqs), H, W), dtype=float)
    from src.fe2pef import demod as demodmod
    for i in range(H):
        for j in range(W):
            sig = detector[i,j,:]
            comps = demodmod.demod_fft(t, sig, pixel_freqs)
            for k,f in enumerate(pixel_freqs):
                demod_maps[k, i, j] = np.abs(comps[f])

    # Unmix using NNMF (flatten spatial)
    V = demod_maps.reshape(len(pixel_freqs), -1)  # channels x pixels
    Wmat, Hmat = nnmf_demo.run_nnmf(V, n_components=3, max_iter=300)
    recon = (Wmat @ Hmat).reshape(demod_maps.shape)

    # Simple SNR metric: ratio of mean signal in masked region to std outside
    masks = [species_maps[s] > (0.2 * species_maps[s].max()) for s in range(species_maps.shape[0])]
    metrics = {}
    for k in range(3):
        sig_region = recon[k][masks[k]]
        noise_region = recon[k][~masks[k]]
        metrics[f'component_{k}'] = {
            'mean_sig': float(sig_region.mean()),
            'std_noise': float(noise_region.std()),
            'snr': float(sig_region.mean() / (noise_region.std()+1e-12))
        }

    # Save simple outputs
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,3, figsize=(12,8))
    axs[0,0].imshow(unmodulated, cmap='magma'); axs[0,0].set_title('Unmodulated (sum over t)')
    for k in range(3):
        axs[0, k].imshow(demod_maps[k], cmap='inferno'); axs[0,k].set_title(f'Demod @ ch{k+1}')
    for k in range(3):
        axs[1, k].imshow(recon[k].reshape(H,W), cmap='viridis'); axs[1,k].set_title(f'Recon comp {k} (SNR {metrics[f"component_{k}"]["snr"]:.2f})')
    plt.tight_layout()
    out = os.path.join(OUT, 'demo_fig.png')
    plt.savefig(out, dpi=150)
    print('Saved demo figure to', out)
    print('Metrics:', metrics)

if __name__ == '__main__':
    main()
