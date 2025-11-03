import numpy as np

def add_awgn(H, snr_db, rng):
    sig_power = np.mean(np.abs(H)**2)
    snr_lin = 10.0**(snr_db/10.0)
    noise_power = max(sig_power / snr_lin, 1e-20)
    n = (np.sqrt(noise_power/2.0) *
         (rng.standard_normal(H.shape) + 1j*rng.standard_normal(H.shape)))
    return H + n

def apply_antenna_offsets(H, phase_static_rad, gain_static_std, rng):
    Nsub, Nrx, Ntx = H.shape
    Hc = H.copy()
    for r in range(Nrx):
        phase = rng.normal(0.0, phase_static_rad)
        gain  = 1.0 + rng.normal(0.0, gain_static_std)
        Hc[:, r, :] *= gain * np.exp(1j*phase)
    return Hc

def apply_snapshot_jitter(H, phase_jitter_rad, gain_jitter_std, rng):
    Nsub, Nrx, Ntx = H.shape
    Hc = H.copy()
    for r in range(Nrx):
        dphi = rng.normal(0.0, phase_jitter_rad)
        dg   = 1.0 + rng.normal(0.0, gain_jitter_std)
        Hc[:, r, :] *= dg * np.exp(1j*dphi)
    return Hc

def apply_cfo(H, cfo_ppm, cfo_jitter_ppm, subcarrier_spacing_hz, rng):
    Nsub, Nrx, Ntx = H.shape
    ppm = rng.normal(loc=cfo_ppm, scale=cfo_jitter_ppm)
    k = np.arange(Nsub) - (Nsub//2)
    tau = ppm * 1e-6 / max(subcarrier_spacing_hz, 1.0)
    phase_ramp = np.exp(1j * 2.0*np.pi * k * subcarrier_spacing_hz * tau)
    Hc = H.copy()
    for r in range(Nrx):
        for t in range(Ntx):
            Hc[:, r, t] = Hc[:, r, t] * phase_ramp
    return Hc

def quantize_complex(H, bits):
    if bits is None or bits <= 0: return H
    peak = np.max(np.abs(H))
    if peak < 1e-12: return H
    Hn = H / peak
    levels = 2**bits; q = (levels/2 - 1)
    real_q = np.round(np.real(Hn)*q)/q
    imag_q = np.round(np.imag(Hn)*q)/q
    return (real_q + 1j*imag_q) * peak
