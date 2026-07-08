"""
GMM-UBM Based Language Identification — Multi-Language
Dataset : FLEURS
Supports CPU (scikit-learn) and GPU (PyTorch) modes.
"""

import os
import glob
import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# ── optional imports resolved at runtime ──────────────────────────────────────
try:
    from sklearn.mixture import GaussianMixture as SklearnGMM
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GMM-UBM Language Identification (CPU / GPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # paths
    p.add_argument("--data_root",   default="/data/Database/FLEURS/downloads/data",
                   help="Root folder containing one subfolder per language")
    p.add_argument("--model_path",  default="gmm_ubm_lid_multiclass.pkl",
                   help="Where to save / load the trained model")

    # features
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--n_mfcc",      type=int, default=13,
                   help="Number of MFCC coefficients (features = n_mfcc × 3)")
    p.add_argument("--min_dur",     type=float, default=0.5,
                   help="Minimum audio duration in seconds to keep a file")

    # GMM
    p.add_argument("--n_components",       type=int,   default=64,
                   help="Number of Gaussian components in UBM and language GMMs")
    p.add_argument("--max_iter",           type=int,   default=200,
                   help="Max EM iterations for UBM training")
    p.add_argument("--n_init",             type=int,   default=3,
                   help="Number of random EM restarts; best result is kept (CPU only)")
    p.add_argument("--reg_covar",          type=float, default=1e-3,
                   help="Covariance regularisation — increase if GMM training diverges "
                        "(e.g. 1e-3 for float32 data, 1e-6 for float64)")
    p.add_argument("--relevance_factor",   type=float, default=16.0,
                   help="MAP relevance factor (higher = stay closer to UBM)")

    # hardware
    p.add_argument("--device", default="cpu",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="'auto' picks CUDA > MPS > CPU automatically")

    # mode
    p.add_argument("--mode", default="train",
                   choices=["train", "eval", "infer"],
                   help="train: full pipeline; eval: load model and evaluate; "
                        "infer: identify language of a single file")
    p.add_argument("--audio_file", default=None,
                   help="Path to audio file for --mode infer")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE RESOLUTION
# ─────────────────────────────────────────────────────────────────────────────
def resolve_device(requested: str) -> str:
    """Return 'cuda', 'mps', or 'cpu'."""
    if not _TORCH_OK:
        if requested in ("cuda", "mps"):
            raise RuntimeError("PyTorch not installed; cannot use GPU mode.")
        return "cpu"

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA GPU found.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but not available on this system.")
    return requested


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def discover_languages(root: str) -> dict:
    langs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d, "audio"))
    ])
    if not langs:
        raise RuntimeError(f"No language folders found under {root}")
    lang_map = {lang: idx for idx, lang in enumerate(langs)}
    print(f"\nDiscovered {len(lang_map)} language(s): {list(lang_map.keys())}")
    return lang_map


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(fp: str, sr: int, n_mfcc: int, min_dur: float):
    """Returns (T, n_mfcc*3) array or None."""
    try:
        y, _ = librosa.load(fp, sr=sr, mono=True)
        if len(y) < sr * min_dur:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.vstack([mfcc,
                          librosa.feature.delta(mfcc),
                          librosa.feature.delta(mfcc, order=2)]).T.astype(np.float64)
    except Exception as e:
        print(f"  [WARN] {fp}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_split(split: str, languages: dict, args):
    utt_feats, utt_labels, lang_frames = [], [], {lang: [] for lang in languages}

    for lang, label in languages.items():
        audio_dir = os.path.join(args.data_root, lang, "audio", split)
        if not os.path.isdir(audio_dir):
            print(f"  [SKIP] {audio_dir}")
            continue

        files = (glob.glob(os.path.join(audio_dir, "*.wav"))  +
                 glob.glob(os.path.join(audio_dir, "*.flac")) +
                 glob.glob(os.path.join(audio_dir, "*.mp3")))

        ok = 0
        for fp in files:
            feat = extract_features(fp, args.sample_rate, args.n_mfcc, args.min_dur)
            if feat is None:
                continue
            utt_feats.append(feat)
            utt_labels.append(label)
            lang_frames[lang].append(feat)
            ok += 1
        print(f"  [{split}] {lang}: {ok}/{len(files)} files")

    for lang in lang_frames:
        lang_frames[lang] = (np.vstack(lang_frames[lang])
                             if lang_frames[lang] else None)
    return utt_feats, utt_labels, lang_frames


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════  CPU BACKEND  (scikit-learn)  ════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
class CPUBackend:
    """Pure scikit-learn GMM-UBM."""

    def __init__(self, args):
        self.args = args

    def train_ubm(self, pooled: np.ndarray) -> SklearnGMM:
        a = self.args
        print(f"\n[CPU] Training UBM  frames={len(pooled):,}  K={a.n_components}  "
              f"reg_covar={a.reg_covar}  n_init={a.n_init}")
        ubm = SklearnGMM(n_components=a.n_components, covariance_type="diag",
                         max_iter=a.max_iter, n_init=a.n_init,
                         reg_covar=a.reg_covar,
                         verbose=1, verbose_interval=20, random_state=42)
        ubm.fit(pooled)
        print(f"  Converged: {ubm.converged_}")
        return ubm

    def map_adapt(self, ubm: SklearnGMM, data: np.ndarray) -> SklearnGMM:
        rf = self.args.relevance_factor
        log_r  = ubm._estimate_log_prob(data) + ubm._estimate_log_weights()
        log_r -= log_r.max(axis=1, keepdims=True)
        resp   = np.exp(log_r)
        resp  /= resp.sum(axis=1, keepdims=True)

        n_k   = resp.sum(axis=0)
        E_x   = (resp.T @ data) / (n_k[:, None] + 1e-10)
        alpha = n_k / (n_k + rf)

        adapted = SklearnGMM(n_components=ubm.n_components,
                             covariance_type="diag", max_iter=1,
                             reg_covar=self.args.reg_covar, random_state=42)
        adapted.fit(data[:100])
        adapted.weights_             = ubm.weights_.copy()
        adapted.means_               = alpha[:, None] * E_x + (1 - alpha[:, None]) * ubm.means_
        adapted.covariances_         = ubm.covariances_.copy()
        adapted.precisions_cholesky_ = ubm.precisions_cholesky_.copy()
        return adapted

    def score(self, feat: np.ndarray, gmm: SklearnGMM, ubm: SklearnGMM) -> float:
        return gmm.score(feat) - ubm.score(feat)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════  GPU BACKEND  (PyTorch)  ══════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
class TorchGMM:
    """
    Diagonal-covariance GMM stored as torch tensors.
    Compatible with both CUDA and MPS.
    """
    def __init__(self, weights, means, log_vars, device):
        self.weights  = torch.tensor(weights,  dtype=torch.float32, device=device)
        self.means    = torch.tensor(means,    dtype=torch.float32, device=device)
        self.log_vars = torch.tensor(log_vars, dtype=torch.float32, device=device)
        self.device   = device
        self.K, self.D = means.shape

    def log_prob(self, X: torch.Tensor) -> torch.Tensor:
        """
        X : (N, D)
        Returns mean log-likelihood over frames (scalar).
        """
        # (N, 1, D) – (1, K, D)  →  (N, K, D)
        diff     = X.unsqueeze(1) - self.means.unsqueeze(0)
        log_vars = self.log_vars.unsqueeze(0)               # (1, K, D)
        # log N(x; mu, diag(sigma²))
        log_comp = -0.5 * (
            self.D * np.log(2 * np.pi)
            + log_vars.sum(dim=2)
            + (diff ** 2 / log_vars.exp()).sum(dim=2)
        )                                                   # (N, K)
        log_mix  = log_comp + torch.log(self.weights).unsqueeze(0)
        return torch.logsumexp(log_mix, dim=1).mean()       # scalar


class GPUBackend:
    """PyTorch-based GMM-UBM for CUDA / MPS."""

    def __init__(self, args, device: str):
        self.args   = args
        self.device = device
        print(f"\n[GPU] Using device: {device}")

    # ── EM helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _e_step(X: torch.Tensor, weights, means, log_vars):
        diff     = X.unsqueeze(1) - means.unsqueeze(0)
        log_comp = -0.5 * (log_vars.unsqueeze(0).sum(2)
                           + (diff ** 2 / log_vars.unsqueeze(0).exp()).sum(2))
        log_mix  = log_comp + torch.log(weights).unsqueeze(0)
        log_resp = log_mix - torch.logsumexp(log_mix, dim=1, keepdim=True)
        return log_resp.exp()                               # (N, K)

    def train_ubm(self, pooled: np.ndarray) -> TorchGMM:
        a = self.args
        K, D = a.n_components, pooled.shape[1]
        print(f"[GPU] Training UBM  frames={len(pooled):,}  K={K}  device={self.device}")

        X = torch.tensor(pooled, dtype=torch.float32, device=self.device)

        # k-means++ init
        idx  = [np.random.randint(len(pooled))]
        dist = np.full(len(pooled), np.inf)
        for _ in range(K - 1):
            d    = np.sum((pooled - pooled[idx[-1]]) ** 2, axis=1)
            dist = np.minimum(dist, d)
            probs = dist / dist.sum()
            idx.append(np.random.choice(len(pooled), p=probs))

        means    = torch.tensor(pooled[idx], dtype=torch.float32, device=self.device)
        log_vars = torch.zeros(K, D, device=self.device)
        weights  = torch.full((K,), 1.0 / K, device=self.device)

        prev_ll = -torch.inf
        for it in range(a.max_iter):
            # E-step
            resp = self._e_step(X, weights, means, log_vars)   # (N, K)
            n_k  = resp.sum(0) + 1e-10                          # (K,)

            # M-step
            weights  = n_k / n_k.sum()
            means    = (resp.T @ X) / n_k[:, None]
            diff2    = ((X.unsqueeze(1) - means.unsqueeze(0)) ** 2)
            log_vars = torch.log((resp.unsqueeze(2) * diff2).sum(0) / n_k[:, None] + 1e-6)

            # log-likelihood for convergence check
            gmm = TorchGMM(weights.cpu().numpy(),
                           means.cpu().numpy(),
                           log_vars.cpu().numpy(),
                           self.device)
            ll = gmm.log_prob(X).item()
            if it % 20 == 0:
                print(f"  iter {it:4d}  ll={ll:.4f}")
            if abs(ll - prev_ll) < 1e-4:
                print(f"  Converged at iter {it}")
                break
            prev_ll = ll

        return TorchGMM(weights.cpu().numpy(),
                        means.cpu().numpy(),
                        log_vars.cpu().numpy(),
                        self.device)

    def map_adapt(self, ubm: TorchGMM, data: np.ndarray) -> TorchGMM:
        rf = self.args.relevance_factor
        X  = torch.tensor(data, dtype=torch.float32, device=self.device)

        resp = self._e_step(X, ubm.weights, ubm.means, ubm.log_vars)
        n_k  = resp.sum(0) + 1e-10
        E_x  = (resp.T @ X) / n_k[:, None]

        alpha = n_k / (n_k + rf)
        new_means = (alpha[:, None] * E_x
                     + (1 - alpha[:, None]) * ubm.means)

        return TorchGMM(ubm.weights.cpu().numpy(),
                        new_means.cpu().numpy(),
                        ubm.log_vars.cpu().numpy(),
                        self.device)

    def score(self, feat: np.ndarray, gmm: TorchGMM, ubm: TorchGMM) -> float:
        X = torch.tensor(feat, dtype=torch.float32, device=self.device)
        return (gmm.log_prob(X) - ubm.log_prob(X)).item()


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION & EVALUATION  (device-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
def predict_batch(utt_feats, lang_gmms, ubm, scaler, languages, backend):
    label_of = dict(languages)
    preds = []
    for feat in utt_feats:
        fs = scaler.transform(feat)
        scores = {lang: backend.score(fs, gmm, ubm)
                  for lang, gmm in lang_gmms.items()}
        preds.append(label_of[max(scores, key=scores.get)])
    return preds


def evaluate(utt_feats, true_labels, lang_gmms, ubm,
             scaler, languages, backend, split_name):
    idx_to_lang = {v: k for k, v in languages.items()}
    pred_labels = predict_batch(utt_feats, lang_gmms, ubm, scaler,
                                languages, backend)
    true_names  = [idx_to_lang[l] for l in true_labels]
    pred_names  = [idx_to_lang[l] for l in pred_labels]
    lang_names  = list(languages.keys())

    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n{'='*60}")
    print(f"  [{split_name}]   Accuracy: {acc*100:.2f}%")
    print(f"{'='*60}")
    print(classification_report(true_names, pred_names,
                                target_names=lang_names, zero_division=0))

    cm = confusion_matrix(true_labels, pred_labels)
    hdr = "          " + "  ".join(f"{l:>8}" for l in lang_names)
    print("  Confusion Matrix (rows=true, cols=pred):")
    print(hdr)
    for i, row in enumerate(cm):
        print(f"  {idx_to_lang[i]:<8}  " +
              "  ".join(f"{v:>8}" for v in row))
    return acc


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_train(args, backend, languages):
    print("\n── Loading splits ──")
    tr_feats, tr_labels, tr_frames = load_split("train", languages, args)
    dv_feats, dv_labels, _         = load_split("dev",   languages, args)
    ts_feats, ts_labels, _         = load_split("test",  languages, args)

    print("\n── Fitting scaler ──")
    all_train = np.vstack(tr_feats)
    scaler = StandardScaler()
    scaler.fit(all_train)
    print(f"  Feature dim={all_train.shape[1]}  frames={len(all_train):,}")

    pooled = scaler.transform(all_train)
    ubm = backend.train_ubm(pooled)

    print("\n── MAP-adapting language GMMs ──")
    lang_gmms = {}
    for lang in languages:
        if tr_frames[lang] is None:
            print(f"  [SKIP] {lang} — no data"); continue
        data = scaler.transform(tr_frames[lang])
        print(f"  {lang}: {len(data):,} frames")
        lang_gmms[lang] = backend.map_adapt(ubm, data)

    evaluate(dv_feats, dv_labels, lang_gmms, ubm, scaler, languages, backend, "DEV")
    evaluate(ts_feats, ts_labels, lang_gmms, ubm, scaler, languages, backend, "TEST")

    # serialise (move tensors to CPU numpy for portability)
    def gmm_to_dict(g):
        if isinstance(g, TorchGMM):
            return {"type": "torch",
                    "weights":  g.weights.cpu().numpy(),
                    "means":    g.means.cpu().numpy(),
                    "log_vars": g.log_vars.cpu().numpy()}
        return {"type": "sklearn", "gmm": g}

    payload = dict(
        ubm=gmm_to_dict(ubm),
        lang_gmms={l: gmm_to_dict(g) for l, g in lang_gmms.items()},
        scaler=scaler,
        languages=languages,
        n_mfcc=args.n_mfcc,
        backend="gpu" if not isinstance(backend, CPUBackend) else "cpu",
    )
    with open(args.model_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n── Model saved → {args.model_path} ──")


# ─────────────────────────────────────────────────────────────────────────────
# INFER  (single file)
# ─────────────────────────────────────────────────────────────────────────────
def run_infer(args, backend):
    if not args.audio_file:
        raise ValueError("--audio_file required for --mode infer")
    with open(args.model_path, "rb") as f:
        m = pickle.load(f)

    def dict_to_gmm(d, device):
        if d["type"] == "torch":
            return TorchGMM(d["weights"], d["means"], d["log_vars"], device)
        return d["gmm"]

    device = getattr(backend, "device", "cpu")
    ubm       = dict_to_gmm(m["ubm"], device)
    lang_gmms = {l: dict_to_gmm(g, device) for l, g in m["lang_gmms"].items()}
    scaler    = m["scaler"]

    feat = extract_features(args.audio_file, args.sample_rate,
                            args.n_mfcc, args.min_dur)
    if feat is None:
        print("Error: could not extract features."); return

    fs = scaler.transform(feat)
    scores = {lang: backend.score(fs, gmm, ubm)
              for lang, gmm in lang_gmms.items()}
    print("\nLLR Scores:")
    for lang, sc in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {lang:>10}: {sc:+.4f}")
    print(f"\nPredicted language: {max(scores, key=scores.get)}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = build_parser().parse_args()
    device = resolve_device(args.device)
    print(f"\n{'='*60}")
    print(f"  GMM-UBM Language Identification")
    print(f"  device={device}  mode={args.mode}  K={args.n_components}")
    print(f"{'='*60}")

    # choose backend
    if device == "cpu":
        if not _SKLEARN_OK:
            raise RuntimeError("scikit-learn not installed.")
        backend = CPUBackend(args)
    else:
        if not _TORCH_OK:
            raise RuntimeError("PyTorch not installed for GPU mode.")
        backend = GPUBackend(args, device)

    if args.mode in ("train", "eval"):
        languages = discover_languages(args.data_root)

    if args.mode == "train":
        run_train(args, backend, languages)
    elif args.mode == "eval":
        # load saved model and evaluate on dev / test
        with open(args.model_path, "rb") as f:
            m = pickle.load(f)
        print(f"Loaded model from {args.model_path}")
        # quick re-eval on test split
        ts_feats, ts_labels, _ = load_split("test", languages, args)
        def dict_to_gmm(d, dev):
            if d["type"] == "torch":
                return TorchGMM(d["weights"], d["means"], d["log_vars"], dev)
            return d["gmm"]
        dev = getattr(backend, "device", "cpu")
        ubm       = dict_to_gmm(m["ubm"], dev)
        lang_gmms = {l: dict_to_gmm(g, dev) for l, g in m["lang_gmms"].items()}
        evaluate(ts_feats, ts_labels, lang_gmms, ubm,
                 m["scaler"], m["languages"], backend, "TEST")
    elif args.mode == "infer":
        run_infer(args, backend)


if __name__ == "__main__":
    main()