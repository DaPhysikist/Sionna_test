#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, os, h5py, numpy as np
from utils.csi_utils import (
    add_awgn, apply_antenna_offsets, apply_snapshot_jitter,
    apply_cfo, quantize_complex
)

def main(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.inp, 'r') as fi, h5py.File(args.out, 'w') as fo:
        fc = fi.attrs.get('fc', None)
        bw = fi.attrs.get('bw', None)
        nsub_attr = fi.attrs.get('nsub', None)

        pos_keys = [k for k in fi.keys() if k.startswith('pos_')]
        for pk in pos_keys:
            gi = fi[pk]
            Hre = gi['H_freq_real'][:]
            Him = gi['H_freq_imag'][:]
            H0  = Hre + 1j*Him
            Nsub, Nrx, Ntx = H0.shape

            if bw is not None and (nsub_attr is not None and nsub_attr>0):
                sc_spacing = bw / float(nsub_attr)
            else:
                sc_spacing = (args.bw / float(Nsub)) if args.bw else 312_500.0

            H_static = apply_antenna_offsets(
                H0,
                phase_static_rad=np.deg2rad(args.phase_static_deg),
                gain_static_std=args.gain_static_std,
                rng=rng
            )

            K = args.samples
            csi_re = np.zeros((K, Nsub, Nrx, Ntx), dtype=np.float32)
            csi_im = np.zeros_like(csi_re)
            for s in range(K):
                Hs = apply_snapshot_jitter(
                        H_static,
                        phase_jitter_rad=np.deg2rad(args.phase_deg),
                        gain_jitter_std=args.gain_std,
                        rng=rng
                     )
                Hs = apply_cfo(
                        Hs,
                        cfo_ppm=args.cfo_ppm,
                        cfo_jitter_ppm=args.cfo_jitter_ppm,
                        subcarrier_spacing_hz=sc_spacing,
                        rng=rng
                     )
                Hs = add_awgn(Hs, snr_db=args.snr, rng=rng)
                Hs = quantize_complex(Hs, bits=args.quant_bits)

                csi_re[s,...] = np.real(Hs)
                csi_im[s,...] = np.imag(Hs)

            go = fo.create_group(pk)
            go.create_dataset('csi_re', data=csi_re, compression='gzip')
            go.create_dataset('csi_im', data=csi_im, compression='gzip')
            if 'gt_pos' in gi:
                go.create_dataset('gt_pos', data=gi['gt_pos'][:])

            go.attrs['snr_db']          = float(args.snr)
            go.attrs['phase_deg']       = float(args.phase_deg)
            go.attrs['phase_static_deg']= float(args.phase_static_deg)
            go.attrs['gain_std']        = float(args.gain_std)
            go.attrs['gain_static_std'] = float(args.gain_static_std)
            go.attrs['cfo_ppm']         = float(args.cfo_ppm)
            go.attrs['cfo_jitter_ppm']  = float(args.cfo_jitter_ppm)
            go.attrs['quant_bits']      = int(args.quant_bits)
            go.attrs['subcarrier_spacing_hz'] = float(sc_spacing)

            print(f"[OK] {pk}: {K} snapshots @ SNR={args.snr} dB")

    print("Saved:", args.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input channel HDF5")
    ap.add_argument("--out", default="data/csi_quick.h5", help="output CSI HDF5")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--snr", type=float, default=18.0)
    ap.add_argument("--phase_deg", type=float, default=5.0)
    ap.add_argument("--gain_std", type=float, default=0.05)
    ap.add_argument("--phase_static_deg", type=float, default=10.0)
    ap.add_argument("--gain_static_std", type=float, default=0.10)
    ap.add_argument("--cfo_ppm", type=float, default=2.0)
    ap.add_argument("--cfo_jitter_ppm", type=float, default=0.3)
    ap.add_argument("--quant_bits", type=int, default=12)
    ap.add_argument("--bw", type=float, default=None)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    main(args)
