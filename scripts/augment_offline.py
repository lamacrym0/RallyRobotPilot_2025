#!/usr/bin/env python3
# augment_offline.py
import argparse, os, copy, pickle, lzma
import numpy as np

# Hypothèse d'ordre des labels: [gaz, frein, gauche, droite]
LEFT_IDX, RIGHT_IDX = 2, 3

# ---------- Helpers IO ----------
def try_load_lzma_pickle(path):
    try:
        with lzma.open(path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception:
        return None

def try_load_npz(path):
    try:
        npz = np.load(path, allow_pickle=True)
        return npz
    except Exception:
        return None

def detect_npz_format(npz):
    keys = set(npz.files)
    if {"X","y"}.issubset(keys): return "xy"
    if {"speed","raycast_distances","current_controls"}.issubset(keys): return "separe"
    return None

# ---------- Transforms ----------
def apply_beam_transforms(d, rng, jitter_mul, jitter_add, mask_prob, shift_max):
    """d: (15,) numpy float. Retourne transformé."""
    d2 = d.astype(np.float32).copy()
    # jitter multiplicatif + additif
    if jitter_mul > 0:
        d2 *= (1.0 + rng.normal(0, jitter_mul, size=d2.shape)).astype(np.float32)
    if jitter_add > 0:
        d2 += rng.normal(0, jitter_add, size=d2.shape).astype(np.float32)
    # masquage
    if mask_prob > 0:
        mask = rng.random(size=d2.shape) < mask_prob
        d2[mask] = 0.0
    # décalage circulaire
    if shift_max > 0:
        s = rng.integers(-shift_max, shift_max+1)
        if s != 0:
            d2 = np.roll(d2, s)
    # clamp distances >= 0
    d2 = np.maximum(d2, 0.0)
    return d2

def maybe_mirror(d, y, mirror):
    """Inverse les 15 distances + swap gauche/droite si mirror=True."""
    if not mirror:
        return d, y
    d_m = d[::-1].copy()
    y_m = y.copy()
    y_m[[LEFT_IDX, RIGHT_IDX]] = y_m[[RIGHT_IDX, LEFT_IDX]]
    return d_m, y_m

# ---------- Augmentation pour LZMA+pickle ----------
def augment_lzma_pickle(data, num_aug, mirror_prob, jitter_mul, jitter_add, mask_prob, shift_max, seed):
    rng = np.random.default_rng(seed)
    out = []
    for e in data:
        # Ajoute l'original ? on laisse la logique "concat" au main
        for _ in range(num_aug):
            m = copy.deepcopy(e)
            # dists
            d = np.array(m.raycast_distances, dtype=np.float32)
            d = apply_beam_transforms(d, rng, jitter_mul, jitter_add, mask_prob, shift_max)
            do_mirror = rng.random() < mirror_prob
            d, y = maybe_mirror(d, np.array(m.current_controls, dtype=np.float32), do_mirror)
            # assign back
            m.raycast_distances = tuple(float(x) for x in d.tolist())
            if do_mirror:
                y = y.tolist()
                c = list(m.current_controls)
                c[LEFT_IDX], c[RIGHT_IDX] = y[LEFT_IDX], y[RIGHT_IDX]  # déjà swappé
                m.current_controls = tuple(c)
            out.append(m)
    return out

# ---------- Augmentation pour NPZ (X/y) ----------
def augment_npz_xy(X, y, num_aug, mirror_prob, jitter_mul, jitter_add, mask_prob, shift_max, seed):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    X_out = np.zeros((N*num_aug, X.shape[1]), dtype=X.dtype)
    Y_out = np.zeros((N*num_aug, y.shape[1]), dtype=y.dtype)
    k = 0
    for i in range(N):
        speed = float(X[i, 0])
        d = X[i, 1:16].astype(np.float32)
        lbl = y[i].astype(np.float32)
        for _ in range(num_aug):
            d2 = apply_beam_transforms(d, rng, jitter_mul, jitter_add, mask_prob, shift_max)
            do_mirror = rng.random() < mirror_prob
            d2, lbl2 = maybe_mirror(d2, lbl, do_mirror)
            X_out[k, 0] = speed
            X_out[k, 1:16] = d2
            Y_out[k] = lbl2
            k += 1
    return X_out, Y_out

# ---------- Augmentation pour NPZ (séparé) ----------
def augment_npz_separe(speed, dists, ctrls, num_aug, mirror_prob, jitter_mul, jitter_add, mask_prob, shift_max, seed):
    rng = np.random.default_rng(seed)
    N = dists.shape[0]
    S_out = np.zeros((N*num_aug,), dtype=speed.dtype)
    D_out = np.zeros((N*num_aug, 15), dtype=dists.dtype)
    C_out = np.zeros((N*num_aug, ctrls.shape[1]), dtype=ctrls.dtype)
    k = 0
    for i in range(N):
        s = speed[i]
        d = dists[i].astype(np.float32)
        c = ctrls[i].astype(np.float32)
        for _ in range(num_aug):
            d2 = apply_beam_transforms(d, rng, jitter_mul, jitter_add, mask_prob, shift_max)
            do_mirror = rng.random() < mirror_prob
            d2, c2 = maybe_mirror(d2, c, do_mirror)
            S_out[k] = s
            D_out[k] = d2
            C_out[k] = c2
            k += 1
    return S_out, D_out, C_out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Augmentation offline (miroir, jitter, masquage, décalage) pour .npz")
    ap.add_argument("input", help="Fichier d'entrée (.npz LZMA+pickle de ton projet, ou vrai .npz)")
    ap.add_argument("output", help="Fichier de sortie augmenté (même format que l'entrée)")
    ap.add_argument("--num-aug", type=int, default=1, help="Nombre d'augmentations générées par échantillon (>=1)")
    ap.add_argument("--concat", action="store_true", help="Inclure aussi les échantillons originaux dans la sortie")
    # paramètres des transforms
    ap.add_argument("--mirror-prob", type=float, default=0.5, help="Proba d'appliquer le miroir (0..1)")
    ap.add_argument("--jitter-mul", type=float, default=0.05, help="Bruit multiplicatif sur raycasts (écart-type)")
    ap.add_argument("--jitter-add", type=float, default=0.01, help="Bruit additif sur raycasts (écart-type)")
    ap.add_argument("--mask-prob",  type=float, default=0.05, help="Proba de masquer un rayon (mise à 0)")
    ap.add_argument("--shift-max",  type=int,   default=1,    help="Décalage circulaire max des rayons (0=off)")
    ap.add_argument("--seed",       type=int,   default=123,  help="Seed rng")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Introuvable: {args.input}")
    if args.num_aug < 1:
        raise SystemExit("--num-aug doit être >= 1")

    # 1) Essai: LZMA+pickle (tes 'record_*.npz')
    data = try_load_lzma_pickle(args.input)
    if data is not None:
        aug = augment_lzma_pickle(
            data, args.num_aug, args.mirror_prob, args.jitter_mul, args.jitter_add,
            args.mask_prob, args.shift_max, args.seed
        )
        if args.concat:
            out_list = list(data) + aug
        else:
            out_list = aug
        with lzma.open(args.output, "wb") as f:
            pickle.dump(out_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ LZMA+pickle → {args.output} | N={len(out_list)} ({'inclut originaux' if args.concat else 'augmentés seuls'})")
        return

    # 2) Sinon: vrai NPZ
    npz = try_load_npz(args.input)
    if npz is None:
        raise SystemExit("Format non reconnu (ni LZMA+pickle, ni NPZ lisible).")

    fmt = detect_npz_format(npz)
    if fmt == "xy":
        X, y = npz["X"], npz["y"]
        Xa, ya = augment_npz_xy(X, y, args.num_aug, args.mirror_prob, args.jitter_mul, args.jitter_add, args.mask_prob, args.shift_max, args.seed)
        if args.concat:
            X_out = np.concatenate([X, Xa], axis=0)
            y_out = np.concatenate([y, ya], axis=0)
        else:
            X_out, y_out = Xa, ya
        np.savez_compressed(args.output, X=X_out, y=y_out)
        print(f"✅ NPZ X/y → {args.output} | X={X_out.shape} y={y_out.shape}")
        return

    if fmt == "separe":
        speed, dists, ctrls = npz["speed"], npz["raycast_distances"], npz["current_controls"]
        Sa, Da, Ca = augment_npz_separe(speed, dists, ctrls, args.num_aug, args.mirror_prob, args.jitter_mul, args.jitter_add, args.mask_prob, args.shift_max, args.seed)
        if args.concat:
            S_out = np.concatenate([speed, Sa], axis=0)
            D_out = np.concatenate([dists, Da], axis=0)
            C_out = np.concatenate([ctrls, Ca], axis=0)
        else:
            S_out, D_out, C_out = Sa, Da, Ca
        np.savez_compressed(args.output, speed=S_out, raycast_distances=D_out, current_controls=C_out)
        print(f"✅ NPZ séparé → {args.output} | N={S_out.shape[0]}")
        return

    raise SystemExit(f"NPZ avec clés inattendues: {sorted(npz.files)}")

if __name__ == "__main__":
    main()
