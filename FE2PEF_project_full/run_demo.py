\
    # run_demo.py - lightweight FE-2PEF vs emission-discrimination comparison demo
    import os, sys, numpy as np, matplotlib.pyplot as plt
    from scipy.signal import fftconvolve
    import json

    # If retinasim is available, try to import and use it
    try:
        import retinasim
        HAVE_RETINASIM = True
    except Exception as e:
        HAVE_RETINASIM = False

    # Simple 2D retina-like fluorophore map: two fluorophores spatially mixed
    nx, ny = 128, 128
    x = np.linspace(-1,1,nx)
    X, Y = np.meshgrid(x,x)
    # Two fluorophores: F1 centered left, F2 centered right with overlap
    F1 = np.exp(-((X+0.4)**2 + (Y)**2)/ (0.12))
    F2 = np.exp(-((X-0.3)**2 + (Y+0.05)**2)/ (0.07))
    # Normalize
    F1 = F1 / F1.max()
    F2 = F2 / F2.max()

    # Assign brightness and emission spectra (toy)
    alpha1 = 1.0
    alpha2 = 0.7
    emission1 = lambda lam: np.exp(-((lam-520)/40)**2)   # nm
    emission2 = lambda lam: np.exp(-((lam-600)/50)**2)

    # Simulate PSF (2D gaussian)
    def psf2d(sigma, size=31):
        ax = np.linspace(-(size-1)/2., (size-1)/2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2+yy**2)/(2.*sigma**2))
        return kernel / kernel.sum()

    psf = psf2d(sigma=2.5, size=25)

    # Optical excitation model for FE-2PEF: two lasers ω1, ω2 modulated at f1,f2
    f1, f2 = 11000.0, 15000.0  # Hz (typical AOTF modulation band)
    A1, A2 = 0.6, 0.5           # modulation depths
    t = np.linspace(0,1,2048)
    sig1 = 1 + A1 * np.cos(2*np.pi*f1*t)
    sig2 = 1 + A2 * np.cos(2*np.pi*f2*t)

    # Compute expected demod amplitudes analytically for three channels: 2f1, 2f2, f1+f2
    # For each fluorophore, assume coefficients c1,c2,c12 linking absorption strengths to lasers.
    c1 = np.array([1.0, 0.2, 0.05]) # F1 response to [laser1^2, laser2^2, cross]
    c2 = np.array([0.1, 1.2, 0.07]) # F2 response

    # Compute pixel-wise photon counts (toy units) before noise and filtering
    base_counts = 200.0
    img_ch1 = fftconvolve(alpha1*F1, psf, mode='same') + fftconvolve(alpha2*F2, psf, mode='same')

    # Channel demod amplitudes (toy mapping)
    demod_map = np.zeros((nx,ny,3))
    demod_map[:,:,0] = c1[0]*fftconvolve(F1, psf, mode='same') + c2[0]*fftconvolve(F2, psf, mode='same')  # 2f1
    demod_map[:,:,1] = c1[1]*fftconvolve(F1, psf, mode='same') + c2[1]*fftconvolve(F2, psf, mode='same')  # 2f2
    demod_map[:,:,2] = c1[2]*fftconvolve(F1, psf, mode='same') + c2[2]*fftconvolve(F2, psf, mode='same')  # f1+f2

    # Add Poisson noise and simulate demod amplitude measurement
    rng = np.random.default_rng(12345)
    counts = base_counts * (demod_map.sum(axis=2)/demod_map.sum())  # coarse
    noisy_counts = rng.poisson(lam=np.clip(counts, 0, None))

    # For emission discrimination simulate spectral filter loss (bandpass around 520 and 600 nm)
    def bandpass_filter_center(lam_center, width, lam):
        return np.exp(-((lam-lam_center)**2)/(2*(width**2)))

    # Simulate SNR metric on single pixel (center)
    cx, cy = nx//2, ny//2
    signal_fe = demod_map[cx,cy,:].sum()
    noise_fe = np.std(demod_map[cx,cy,:] + rng.normal(0,0.1,3))
    snr_fe = signal_fe / (noise_fe + 1e-9)

    # Emission discrimination: filter reduces detected photons
    lam = np.linspace(400,700,301)
    filt1 = bandpass_filter_center(520,40,lam)
    filt2 = bandpass_filter_center(600,50,lam)
    # Compute transmitted fraction for each fluorophore (toy integrals)
    tx1 = np.trapz(emission1(lam) * filt1, lam) / np.trapz(emission1(lam), lam)
    tx2 = np.trapz(emission2(lam) * filt2, lam) / np.trapz(emission2(lam), lam)
    # SNR after filtering (toy)
    signal_em = (alpha1*F1[cx,cy]*tx1 + alpha2*F2[cx,cy]*tx2)
    noise_em = np.sqrt(signal_em + 5.0)  # shot + detector noise
    snr_em = signal_em / (noise_em + 1e-9)

    print('Toy SNR FE-2PEF demod (center pixel):', snr_fe)
    print('Toy SNR emission-discrimination (center pixel):', snr_em)

    # Plot maps
    plt.figure(figsize=(10,4))
    plt.subplot(1,3,1)
    plt.title('Fluorophore F1+F2 (true mix)')
    plt.imshow(F1+F2, origin='lower')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.title('FE demod amplitude (sum channels)')
    plt.imshow(demod_map.sum(axis=2), origin='lower')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.title('PSF-blurred total intensity (toy)')
    plt.imshow(img_ch1, origin='lower')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
