#!/usr/bin/env python3
import argparse, os, h5py, numpy as np
C = 299_792_458.0

def grid_positions(room_x, room_y, grid_res, rx_h):
    xs = np.arange(1.0, room_x, grid_res)
    ys = np.arange(1.0, room_y, grid_res)
    P = []
    for x in xs:
        for y in ys:
            P.append([x, y, rx_h])
    return np.array(P, dtype=np.float32)

def friis_pathloss(db0, d0, d, n, shadow_std_db, rng):
    shadow = rng.normal(0.0, shadow_std_db)
    return db0 + 10.0 * n * np.log10(max(d,1e-3)/d0) + shadow

def random_reflections(tx, rx, n_paths, room_x, room_y, extra_loss_db, rng):
    paths = []
    for _ in range(n_paths):
        b = np.array([rng.uniform(0.0, room_x), rng.uniform(0.0, room_y), rng.uniform(1.0, 2.8)], dtype=np.float32)
        d = np.linalg.norm(tx - b) + np.linalg.norm(b - rx)
        att_db = extra_loss_db + rng.normal(0.0, 1.5)
        paths.append((d, att_db))
    paths.sort(key=lambda x: x[0])
    return paths

def steering_ula(ntx, d_lambda, aoD_rad, subfreqs):
    lam0 = C / np.mean(subfreqs)
    kd = 2.0 * np.pi * (d_lambda * lam0) / lam0
    elem_idx = np.arange(ntx)
    a = np.exp(1j * kd * elem_idx * np.sin(aoD_rad))
    return np.tile(a[None, :], (len(subfreqs), 1))

def make_Hf_for_pos(tx, rx, subfreqs, ntx, nrx, los_prob, nlos_reflections,
                    pl_exp_los, pl_exp_nlos, l0_db, d0, shadow_db, wall_loss_db,
                    room_x, room_y, rng):
    vec = rx - tx
    d_los = np.linalg.norm(vec)
    theta = np.arctan2(vec[1], vec[0])
    is_los = rng.random() < los_prob

    paths = []
    if is_los:
        pl_db = friis_pathloss(l0_db, d0, d_los, pl_exp_los, shadow_db, rng)
        paths.append((d_los, pl_db))
    else:
        pl_db = friis_pathloss(l0_db + wall_loss_db, d0, d_los, pl_exp_nlos, shadow_db, rng)
        paths.append((d_los * 1.05, pl_db + 8.0))

    refl = random_reflections(tx, rx, nlos_reflections, room_x, room_y, wall_loss_db, rng)
    for d_ref, extra_db in refl:
        n_exp = pl_exp_nlos if not is_los else (pl_exp_los + 0.3)
        plr_db = friis_pathloss(l0_db + extra_db, d0, d_ref, n_exp, shadow_db+0.5, rng)
        paths.append((d_ref, plr_db))

    paths.sort(key=lambda x: x[0])
    dists = np.array([p[0] for p in paths], dtype=np.float64)
    p_db = np.array([-x[1] for x in paths], dtype=np.float64)
    p_lin = 10.0**(p_db/10.0); p_lin /= np.max(p_lin)
    taus = dists / C

    a_tx = steering_ula(ntx=ntx, d_lambda=0.5, aoD_rad=theta, subfreqs=subfreqs)
    K = len(subfreqs)
    Hf = np.zeros((K, nrx, ntx), dtype=np.complex128)
    for tau_l, p_l in zip(taus, p_lin):
        phase = np.exp(-1j * 2.0 * np.pi * subfreqs * tau_l)  # [K]
        contrib = (phase[:, None] * a_tx) * np.sqrt(p_l)
        Hf[:, 0, :] += contrib
    return Hf, taus.astype(np.float64), p_lin.astype(np.float64), is_los

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ch_stub.h5")
    ap.add_argument("--fc", type=float, default=5.2e9)
    ap.add_argument("--bw", type=float, default=80e6)
    ap.add_argument("--nsub", type=int, default=32)
    ap.add_argument("--ntx", type=int, default=4)
    ap.add_argument("--nrx", type=int, default=1)
    ap.add_argument("--room_x", type=float, default=10.0)
    ap.add_argument("--room_y", type=float, default=12.0)
    ap.add_argument("--rx_h", type=float, default=1.2)
    ap.add_argument("--tx", nargs=3, type=float, default=[5.0,6.0,2.5])
    ap.add_argument("--grid_res", type=float, default=2.0)
    ap.add_argument("--los_prob", type=float, default=0.6)
    ap.add_argument("--nlos_reflections", type=int, default=3)
    ap.add_argument("--pl_exp_los", type=float, default=2.0)
    ap.add_argument("--pl_exp_nlos", type=float, default=2.9)
    ap.add_argument("--l0_db", type=float, default=40.0)
    ap.add_argument("--d0", type=float, default=1.0)
    ap.add_argument("--shadow_db", type=float, default=3.0)
    ap.add_argument("--wall_loss_db", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    subcarrier_spacing = args.bw / float(args.nsub)
    subfreqs = args.fc + (np.arange(args.nsub) - args.nsub//2) * subcarrier_spacing
    tx = np.array(args.tx, dtype=np.float32)
    R = grid_positions(args.room_x, args.room_y, args.grid_res, args.rx_h)

    with h5py.File(args.out, "w") as f:
        f.attrs['fc'] = args.fc; f.attrs['bw'] = args.bw
        f.attrs['nsub'] = args.nsub; f.attrs['ntx'] = args.ntx; f.attrs['nrx'] = args.nrx
        for i, rx in enumerate(R):
            Hf, taus, p_lin, is_los = make_Hf_for_pos(
                tx=tx, rx=rx, subfreqs=subfreqs, ntx=args.ntx, nrx=args.nrx,
                los_prob=args.los_prob, nlos_reflections=args.nlos_reflections,
                pl_exp_los=args.pl_exp_los, pl_exp_nlos=args.pl_exp_nlos,
                l0_db=args.l0_db, d0=args.d0, shadow_db=args.shadow_db, wall_loss_db=args.wall_loss_db,
                room_x=args.room_x, room_y=args.room_y, rng=rng
            )
            g = f.create_group(f"pos_{i:04d}")
            g.create_dataset("H_freq_real", data=np.real(Hf), compression="gzip")
            g.create_dataset("H_freq_imag", data=np.imag(Hf), compression="gzip")
            g.create_dataset("path_delays", data=taus, compression="gzip")
            g.create_dataset("path_powers", data=p_lin, compression="gzip")
            g.create_dataset("gt_pos", data=rx)
            g.attrs['is_los'] = int(is_los)
            if i % 10 == 0: print(f"pos {i:04d} written")
    print("Saved", args.out)

if __name__ == "__main__":
    main()
