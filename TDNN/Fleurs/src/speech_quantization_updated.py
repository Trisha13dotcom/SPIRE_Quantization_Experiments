"""
speech_quantization.py
======================
Shared quantization framework for the Quantized Speech Analysis intern programme.

Implements all six quantization schemes defined in the Shared Framework document.
All interns should use this module to ensure identical quantization across all tasks
(Language ID, Domain ID, Accent, Dialect, Gender Classification, and MTL).

Usage -- Python API
-------------------
    from speech_quantization import quantize, load_and_quantize, QuantizationScheme

    # Quantize a pre-loaded waveform (numpy array, already normalised to [-1, +1])
    x_q = quantize(x, scheme=1)

    # Load a file, normalise, and quantize in one step
    x_q, sr = load_and_quantize("audio.wav", scheme=3, bit_depth=4)

Usage -- Kaldi wav.scp batch pipeline
--------------------------------------
    # From Python
    from speech_quantization import process_wav_scp

    out_scp = process_wav_scp(
        scp_path  = "data/train/wav.scp",
        out_dir   = "/data/quantized/train",
        scheme    = 1,
        bit_depth = 1,
        in_dir    = "/data/corpora",   # common root; directory structure is mirrored
    )
    # out_scp is the path to the new wav.scp pointing at the quantized files.

    # From the command line (one scheme/depth combination)
    python speech_quantization.py \\
        --scp   data/train/wav.scp \\
        --out   /data/quantized/train \\
        --indir /data/corpora \\
        --scheme 3 --bit-depth 4 \\
        --jobs 8

    # Pre-process ALL scheme/depth combinations in one go
    python speech_quantization.py \\
        --scp   data/train/wav.scp \\
        --out   /data/quantized/train \\
        --indir /data/corpora \\
        --all-schemes --jobs 8

wav.scp format
--------------
    Standard Kaldi two-column format (whitespace-separated):
        <utterance-id>  <audio-path>
    Lines starting with '#' and blank lines are silently skipped.
    Pipe commands (paths ending in '|') are NOT supported -- resolve to plain files first.

Output layout
-------------
    Given --indir /data/corpora and a file at /data/corpora/hindi/spk1/utt1.wav,
    the quantized file is saved to:
        <out_dir>/scheme1/<relative_path_from_indir>/utt1.wav
        => /data/quantized/train/scheme1/hindi/spk1/utt1.wav

    If --indir is not provided, the full absolute path is mirrored:
        /data/quantized/train/scheme1/data/corpora/hindi/spk1/utt1.wav

    A new wav.scp is written alongside each scheme directory:
        <out_dir>/scheme1/wav.scp   (utterance-id => quantized file path)

Schemes
-------
    1  =>  1-bit sign quantization           (shared 1-bit baseline across all tasks)
    2  =>  Uniform mid-tread                 (2 / 4 / 8-bit)
    3  =>  mu-law companding + uniform       (2 / 4 / 8-bit)   ITU-T G.711, mu=255
    4  =>  A-law companding + uniform        (2 / 4 / 8-bit)   ITU-T G.711, A=87.6
    5  =>  Logarithmic (base-2) quantization (2 / 4 / 8-bit)   diagnostic reference
    6  =>  16-bit full precision             (no quantization)  upper bound

Notes
-----
- Input waveforms MUST be normalised to [-1, +1] before quantization.
- Noise / augmentation must be applied BEFORE quantization (see Shared Framework).
- At 1-bit, Schemes 1, 3, and 4 all produce the identical sign signal.
  Differences between companding schemes are only observable from 2-bit upward.
- All quantized files are written as 32-bit float WAV so that the full
  reconstruction range is preserved regardless of scheme.

Dependencies
------------
    numpy, scipy, soundfile, tqdm  (pip install numpy scipy soundfile tqdm)
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import IntEnum
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

MU = 255      # ITU-T G.711 mu-law parameter
A  = 87.6     # ITU-T G.711 A-law parameter
VALID_BIT_DEPTHS = {1, 2, 4, 8, 16}
VALID_SCHEMES    = {1, 2, 3, 4, 5, 6}


class QuantizationScheme(IntEnum):
    """Symbolic names for the six quantization schemes."""
    SIGN_1BIT      = 1   # 1-bit sign quantization
    UNIFORM        = 2   # Uniform mid-tread (linear PCM)
    MU_LAW         = 3   # mu-law companding + uniform
    A_LAW          = 4   # A-law companding + uniform
    LOGARITHMIC    = 5   # Logarithmic (base-2), diagnostic
    FULL_PRECISION = 6   # 16-bit reference (no quantization)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise(x: np.ndarray) -> np.ndarray:
    """
    Peak-normalise a waveform to [-1, +1].

    All quantization schemes require the signal to be in this range first.
    If the signal is silent (max amplitude == 0) it is returned unchanged.

    Parameters
    ----------
    x : np.ndarray
        Raw waveform samples (any dtype).

    Returns
    -------
    np.ndarray
        Float64 array in [-1, +1].
    """
    x    = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x))
    if peak == 0.0:
        return x
    return x / peak


# ---------------------------------------------------------------------------
# Companding functions (public -- useful for analysis)
# ---------------------------------------------------------------------------

def mu_law_compress(x: np.ndarray, mu: float = MU) -> np.ndarray:
    """
    Apply mu-law compression (ITU-T G.711, mu=255).

        F(x) = sgn(x) * ln(1 + mu*|x|) / ln(1 + mu),   x in [-1, +1]

    Parameters
    ----------
    x  : np.ndarray   Signal in [-1, +1].
    mu : float        Compression parameter (default 255).

    Returns
    -------
    np.ndarray   Compressed signal in [-1, +1].
    """
    x = np.clip(x, -1.0, 1.0)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


def mu_law_expand(y: np.ndarray, mu: float = MU) -> np.ndarray:
    """Inverse of mu-law compression."""
    y = np.clip(y, -1.0, 1.0)
    return np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu))) / mu


def a_law_compress(x: np.ndarray, a: float = A) -> np.ndarray:
    """
    Apply A-law compression (ITU-T G.711, A=87.6).

        F(x) = sgn(x) * A*|x| / (1 + ln A),            |x| <= 1/A
        F(x) = sgn(x) * (1 + ln(A*|x|)) / (1 + ln A),  1/A < |x| <= 1

    Parameters
    ----------
    x : np.ndarray   Signal in [-1, +1].
    a : float        Compression parameter (default 87.6).

    Returns
    -------
    np.ndarray   Compressed signal in [-1, +1].
    """
    x   = np.clip(x, -1.0, 1.0)
    ax  = np.abs(x)
    lna = np.log(a)
    out = np.where(
        ax <= 1.0 / a,
        a * ax / (1.0 + lna),
        (1.0 + np.log(a * ax + 1e-12)) / (1.0 + lna),
    )
    return np.sign(x) * np.clip(out, 0.0, 1.0)


def a_law_expand(y: np.ndarray, a: float = A) -> np.ndarray:
    """Inverse of A-law compression."""
    y   = np.clip(y, -1.0, 1.0)
    ay  = np.abs(y)
    lna = np.log(a)
    threshold = 1.0 / (1.0 + lna)
    out = np.where(
        ay < threshold,
        ay * (1.0 + lna) / a,
        np.exp(ay * (1.0 + lna) - 1.0) / a,
    )
    return np.sign(y) * out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _uniform_midtread(x: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Symmetric uniform mid-tread quantizer.

    Step size Delta = 2^(1-B). Output clipped to [-1 + Delta/2, +1 - Delta/2].

    Parameters
    ----------
    x         : np.ndarray   Signal in [-1, +1].
    bit_depth : int          Number of bits B (2, 4, or 8).

    Returns
    -------
    np.ndarray   Quantized signal as float64.
    """
    delta = 2.0 ** (1 - bit_depth)
    x_q   = delta * np.round(x / delta)
    limit = 1.0 - delta / 2.0
    return np.clip(x_q, -limit, limit)


def _log2_quantize(x: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Logarithmic (base-2) quantizer -- diagnostic reference (Scheme 5).

    Places reconstruction levels logarithmically: high resolution near zero,
    coarse resolution at large amplitudes (opposite emphasis to mu-law / A-law).

    Parameters
    ----------
    x         : np.ndarray   Signal in [-1, +1].
    bit_depth : int          Number of bits (2, 4, or 8).

    Returns
    -------
    np.ndarray   Quantized signal as float64.
    """
    x    = np.clip(x, -1.0, 1.0)
    sign = np.sign(x)
    mag  = np.abs(x)

    # Smallest representable magnitude
    eps      = 2.0 ** -(2 ** (bit_depth - 1))
    mag_safe = np.where(mag < eps, eps, mag)

    # Normalise log-magnitude to [0, 1]
    # log2(mag) in [log_min, 0]; log_min is the floor (e.g. -128 for 8-bit).
    # Dividing by log_min (which is negative) maps log_mag=0 -> 0 (mag=1)
    # and log_mag=log_min -> 1 (mag=eps), giving log_norm in [0, 1].
    log_mag  = np.log2(mag_safe)
    log_min  = -float(2 ** (bit_depth - 1))
    log_norm = np.clip(log_mag / log_min, 0.0, 1.0)

    # Uniform quantization in the log domain
    levels = 2 ** (bit_depth - 1)
    log_q  = np.round(log_norm * (levels - 1)) / (levels - 1)

    # Reconstruct magnitude
    # log_q=0 => log_norm=0 => log_mag=0 => mag=1 (maximum, NOT silence)
    # Silence only when sign==0 (x was exactly 0)
    mag_q = 2.0 ** (log_q * log_min)
    return np.where(sign == 0, 0.0, sign * mag_q)


# ---------------------------------------------------------------------------
# Main quantization API
# ---------------------------------------------------------------------------

def quantize(
    x: np.ndarray,
    scheme: int,
    bit_depth: int = 1,
    mu: float = MU,
    a: float = A,
    normalise_input: bool = True,
) -> np.ndarray:
    """
    Quantize a waveform using one of the six shared schemes.

    Peak-normalisation to [-1, +1] is applied automatically before quantization
    (controlled by `normalise_input`, which defaults to True).  You can pass an
    already-normalised array and set normalise_input=False to skip the extra pass,
    but leaving it True is always safe and is the recommended default.

    Parameters
    ----------
    x               : np.ndarray
        Input waveform.  Shape: (N,) or (channels, N).
    scheme          : int
        Quantization scheme (1-6).  Use QuantizationScheme enum for clarity.
    bit_depth       : int
        Target bit depth.
        - Scheme 1 always uses 1-bit (parameter ignored).
        - Schemes 2-5 accept 2, 4, or 8.
        - Scheme 6 always uses 16-bit (parameter ignored).
    mu              : float
        mu-law parameter (default 255, ITU-T G.711).
    a               : float
        A-law parameter (default 87.6, ITU-T G.711).
    normalise_input : bool
        If True (default), peak-normalise x to [-1, +1] before quantizing.
        Set to False only if you have already called normalise() on the array
        and want to avoid the extra pass.

    Returns
    -------
    np.ndarray
        Quantized waveform, same shape as input, dtype float64.
        - Scheme 1: values in {-1.0, +1.0}.
        - Schemes 2-5: float reconstruction levels.
        - Scheme 6: normalised float64 copy of the input.

    Raises
    ------
    ValueError
        If scheme or bit_depth is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> from speech_quantization import quantize
    >>> x = np.random.randn(16000)          # raw, unnormalised
    >>> x_1bit      = quantize(x, scheme=1)                    # normalises automatically
    >>> x_mulaw_4bt = quantize(x, scheme=3, bit_depth=4)       # normalises automatically
    >>> x_ref       = quantize(x, scheme=6)                    # normalises automatically
    """
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"scheme must be one of {VALID_SCHEMES}, got {scheme}.")

    x = np.asarray(x, dtype=np.float64)
    if normalise_input:
        x = normalise(x)
    x = np.clip(x, -1.0, 1.0)   # safety clip after normalisation

    if scheme == 1:
        return np.where(x > 0, 1.0, -1.0)

    if scheme == 6:
        return x.copy()

    if bit_depth not in (2, 4, 8):
        raise ValueError(
            f"bit_depth must be 2, 4, or 8 for schemes 2-5, got {bit_depth}."
        )

    if scheme == 2:
        return _uniform_midtread(x, bit_depth)

    if scheme == 3:
        return _uniform_midtread(mu_law_compress(x, mu), bit_depth)

    if scheme == 4:
        return _uniform_midtread(a_law_compress(x, a), bit_depth)

    if scheme == 5:
        return _log2_quantize(x, bit_depth)


# ---------------------------------------------------------------------------
# Convenience: load, normalise, and quantize in one call
# ---------------------------------------------------------------------------

def load_and_quantize(
    filepath: str,
    scheme: int,
    bit_depth: int = 1,
    target_sr: Optional[int] = 16000,
    mu: float = MU,
    a: float = A,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, normalise it, and apply quantization.

    Parameters
    ----------
    filepath  : str    Path to the audio file (WAV, FLAC, OGG, etc.).
    scheme    : int    Quantization scheme (1-6).
    bit_depth : int    Target bit depth.
    target_sr : int or None
        Resample to this rate.  Pass None to keep the native rate.
    mu        : float  mu-law parameter.
    a         : float  A-law parameter.

    Returns
    -------
    (x_q, sr) : Tuple[np.ndarray, int]
        x_q -- quantized mono waveform (float64, shape (N,)).
        sr  -- sample rate of the returned signal.
    """
    x, sr = sf.read(filepath, dtype="float64", always_2d=False)

    if x.ndim == 2:
        x = x.mean(axis=1)

    if target_sr is not None and sr != target_sr:
        try:
            from scipy.signal import resample_poly
            g         = gcd(target_sr, sr)
            up, down  = target_sr // g, sr // g
            x         = resample_poly(x, up, down)
            sr        = target_sr
        except ImportError:
            raise ImportError(
                "scipy is required for resampling.  "
                "Install it with: pip install scipy"
            )

    x   = normalise(x)
    x_q = quantize(x, scheme=scheme, bit_depth=bit_depth, mu=mu, a=a, normalise_input=False)
    return x_q, sr


# ---------------------------------------------------------------------------
# Batch helper: generate all scheme/depth combinations at once
# ---------------------------------------------------------------------------

def quantize_all_schemes(
    x: np.ndarray,
    bit_depths: Tuple[int, ...] = (2, 4, 8),
    mu: float = MU,
    a: float = A,
    normalise_input: bool = True,
) -> dict:
    """
    Apply all quantization schemes and return a dictionary of results.

    Useful for bit-depth ablation experiments (Exp X.3 in each problem).
    Peak-normalisation is applied once before any scheme is run so that all
    output arrays share exactly the same normalised input as their starting point.

    Parameters
    ----------
    x               : np.ndarray   Input waveform (any range).
    bit_depths      : tuple        Bit depths for Schemes 2-5 (default: 2, 4, 8).
    mu              : float        mu-law parameter.
    a               : float        A-law parameter.
    normalise_input : bool
        If True (default), peak-normalise x once before running all schemes.
        Set to False only if x is already normalised to [-1, +1].

    Returns
    -------
    dict   Keys: "scheme1", "scheme2_2bit", "scheme3_4bit", "scheme6", etc.
           Values: quantized numpy arrays.

    Example
    -------
    >>> x = np.random.randn(16000)         # raw array — normalised automatically
    >>> results = quantize_all_schemes(x)
    >>> results["scheme1"]       # 1-bit sign
    >>> results["scheme3_4bit"]  # mu-law + 4-bit
    >>> results["scheme6"]       # full precision
    """
    # Normalise once so every scheme shares the same starting point.
    x = np.asarray(x, dtype=np.float64)
    if normalise_input:
        x = normalise(x)

    # Pass normalise_input=False to each quantize() call — already done above.
    out = {}
    out["scheme1"] = quantize(x, scheme=1, mu=mu, a=a, normalise_input=False)
    for scheme in (2, 3, 4, 5):
        for bd in bit_depths:
            out[f"scheme{scheme}_{bd}bit"] = quantize(
                x, scheme=scheme, bit_depth=bd, mu=mu, a=a, normalise_input=False
            )
    out["scheme6"] = quantize(x, scheme=6, mu=mu, a=a, normalise_input=False)
    return out


# ---------------------------------------------------------------------------
# wav.scp I/O helpers
# ---------------------------------------------------------------------------

def _scheme_key(scheme: int, bit_depth: int) -> str:
    """
    Return a short string used as the subdirectory name for a scheme/depth pair.

    Examples
    --------
    scheme=1            =>  "scheme1"
    scheme=3, bits=4    =>  "scheme3_4bit"
    scheme=6            =>  "scheme6"
    """
    if scheme == 1:
        return "scheme1"
    if scheme == 6:
        return "scheme6"
    return f"scheme{scheme}_{bit_depth}bit"


def read_wav_scp(scp_path: str) -> List[Tuple[str, str]]:
    """
    Parse a Kaldi-style wav.scp file into a list of (uttid, filepath) pairs.

    Lines beginning with '#' and blank lines are skipped.
    Pipe commands (path ending with '|') raise a ValueError so the caller
    knows to resolve them to plain files before calling this function.

    Parameters
    ----------
    scp_path : str   Path to the wav.scp file.

    Returns
    -------
    List[Tuple[str, str]]   [(utterance_id, audio_filepath), ...]
    """
    entries = []
    with open(scp_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)   # split on first whitespace
            if len(parts) != 2:
                raise ValueError(
                    f"{scp_path}:{lineno}: expected 2 columns, got: {line!r}"
                )
            uttid, path = parts
            if path.endswith("|"):
                raise ValueError(
                    f"{scp_path}:{lineno}: pipe commands are not supported. "
                    f"Resolve '{path}' to a plain file path first."
                )
            entries.append((uttid, path))
    return entries


def _output_path(
    src_path: str,
    out_dir: str,
    scheme_key: str,
    in_dir: Optional[str],
) -> str:
    """
    Compute the destination path for a quantized file, mirroring the
    source directory structure under out_dir/scheme_key/.

    If in_dir is given, the path relative to in_dir is reproduced.
    Otherwise the full absolute source path is mirrored.

    Parameters
    ----------
    src_path   : str            Absolute (or resolvable) path to the source file.
    out_dir    : str            Root output directory.
    scheme_key : str            E.g. "scheme1" or "scheme3_4bit".
    in_dir     : str or None    Common root of the source corpus.

    Returns
    -------
    str   Full path where the quantized file should be written.
    """
    src = Path(src_path).resolve()
    base = Path(out_dir) / scheme_key

    if in_dir is not None:
        root = Path(in_dir).resolve()
        try:
            rel = src.relative_to(root)
        except ValueError:
            raise ValueError(
                f"Source file '{src}' is not under in_dir '{root}'. "
                "Either fix --indir or omit it to mirror the full path."
            )
        return str(base / rel)
    else:
        # Mirror the absolute path (strip leading '/')
        return str(base / str(src).lstrip("/"))


# ---------------------------------------------------------------------------
# Core worker: process a single utterance
# ---------------------------------------------------------------------------

def _process_one(
    uttid: str,
    src_path: str,
    dst_path: str,
    scheme: int,
    bit_depth: int,
    target_sr: Optional[int],
    mu: float,
    a: float,
    skip_existing: bool,
) -> Tuple[str, str, Optional[str]]:
    """
    Load, quantize, and save one utterance.

    Returns
    -------
    (uttid, dst_path, error_message_or_None)
    """
    if skip_existing and Path(dst_path).exists():
        return uttid, dst_path, None

    try:
        x_q, sr = load_and_quantize(
            src_path,
            scheme=scheme,
            bit_depth=bit_depth,
            target_sr=target_sr,
            mu=mu,
            a=a,
        )
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        # Always write as 32-bit float WAV to preserve the full dynamic range
        sf.write(dst_path, x_q.astype(np.float32), sr, subtype="FLOAT")
        return uttid, dst_path, None
    except Exception as exc:
        return uttid, dst_path, str(exc)


# ---------------------------------------------------------------------------
# Public batch API: process a full wav.scp
# ---------------------------------------------------------------------------

def process_wav_scp(
    scp_path: str,
    out_dir: str,
    scheme: int,
    bit_depth: int = 1,
    in_dir: Optional[str] = None,
    target_sr: Optional[int] = 16000,
    mu: float = MU,
    a: float = A,
    jobs: int = 4,
    skip_existing: bool = True,
    fail_on_error: bool = False,
) -> str:
    """
    Quantize every utterance listed in a Kaldi wav.scp and write results to disk.

    Mirrors the source directory structure under out_dir/<scheme_key>/.
    Writes a new wav.scp at out_dir/<scheme_key>/wav.scp mapping utterance IDs
    to their quantized file paths.

    Parameters
    ----------
    scp_path      : str    Path to the input wav.scp.
    out_dir       : str    Root directory for all quantized output.
    scheme        : int    Quantization scheme (1-6).
    bit_depth     : int    Bit depth for schemes 2-5 (ignored for 1 and 6).
    in_dir        : str or None
        Common root of the source corpus.  When given, the path relative to
        in_dir is reproduced under out_dir/<scheme_key>/. When None, the full
        absolute source path is mirrored.
    target_sr     : int or None
        Resample audio to this sample rate before quantizing.  None = keep native.
    mu            : float  mu-law parameter (default 255).
    a             : float  A-law parameter (default 87.6).
    jobs          : int    Number of parallel worker processes (default 4).
    skip_existing : bool   Skip files that already exist on disk (default True).
    fail_on_error : bool   Raise immediately on the first per-file error (default
                           False -- errors are logged and written to a .failed file).

    Returns
    -------
    str   Path to the output wav.scp for the processed scheme.

    Raises
    ------
    FileNotFoundError   If scp_path does not exist.
    ValueError          If the scp contains pipe commands.
    RuntimeError        If fail_on_error=True and any file fails.
    """
    if not Path(scp_path).exists():
        raise FileNotFoundError(f"wav.scp not found: {scp_path}")

    entries    = read_wav_scp(scp_path)
    skey       = _scheme_key(scheme, bit_depth)
    scheme_dir = Path(out_dir) / skey
    scheme_dir.mkdir(parents=True, exist_ok=True)

    out_scp_path    = str(scheme_dir / "wav.scp")
    failed_log_path = str(scheme_dir / "failed.log")

    log.info(
        "Processing %d utterances | scheme=%s | jobs=%d | scp=%s => %s",
        len(entries), skey, jobs, scp_path, out_scp_path,
    )

    # Build work list
    work = []
    for uttid, src_path in entries:
        dst_path = _output_path(src_path, out_dir, skey, in_dir)
        work.append((uttid, src_path, dst_path, scheme, bit_depth,
                     target_sr, mu, a, skip_existing))

    # --- Run (parallel or sequential) ---
    results = []
    errors  = []

    try:
        from tqdm import tqdm
        progress = tqdm(total=len(work), unit="utt", desc=skey)
    except ImportError:
        progress = None

    if jobs == 1:
        for args in work:
            uttid, dst, err = _process_one(*args)
            if progress:
                progress.update(1)
            if err:
                errors.append((uttid, dst, err))
                if fail_on_error:
                    raise RuntimeError(f"Failed on {uttid}: {err}")
            else:
                results.append((uttid, dst))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(_process_one, *args): args[0] for args in work}
            for fut in as_completed(futures):
                uttid, dst, err = fut.result()
                if progress:
                    progress.update(1)
                if err:
                    errors.append((uttid, dst, err))
                    if fail_on_error:
                        raise RuntimeError(f"Failed on {uttid}: {err}")
                else:
                    results.append((uttid, dst))

    if progress:
        progress.close()

    # Write output wav.scp (sorted by uttid for reproducibility)
    results.sort(key=lambda t: t[0])
    with open(out_scp_path, "w", encoding="utf-8") as fh:
        for uttid, dst in results:
            fh.write(f"{uttid} {dst}\n")

    # Write failed log if any
    if errors:
        with open(failed_log_path, "w", encoding="utf-8") as fh:
            for uttid, dst, err in errors:
                fh.write(f"{uttid}\t{dst}\t{err}\n")
        log.warning(
            "%d/%d utterances FAILED. See %s", len(errors), len(work), failed_log_path
        )
    else:
        log.info("All %d utterances processed successfully.", len(work))

    log.info("Output wav.scp written to: %s", out_scp_path)
    return out_scp_path


def process_wav_scp_all_schemes(
    scp_path: str,
    out_dir: str,
    in_dir: Optional[str] = None,
    bit_depths: Tuple[int, ...] = (2, 4, 8),
    target_sr: Optional[int] = 16000,
    mu: float = MU,
    a: float = A,
    jobs: int = 4,
    skip_existing: bool = True,
) -> Dict[str, str]:
    """
    Pre-process a wav.scp for ALL scheme/depth combinations.

    Iterates over Scheme 1, Schemes 2-5 x bit_depths, and Scheme 6.
    Each combination is written to its own subdirectory with its own wav.scp.
    Use this to fully pre-process a corpus once before running experiments.

    Parameters
    ----------
    scp_path    : str           Input wav.scp.
    out_dir     : str           Root output directory.
    in_dir      : str or None   Common source root for directory mirroring.
    bit_depths  : tuple         Bit depths for Schemes 2-5 (default: 2, 4, 8).
    target_sr   : int or None   Target sample rate (default 16000).
    mu          : float         mu-law parameter.
    a           : float         A-law parameter.
    jobs        : int           Parallel workers per scheme (default 4).
    skip_existing: bool         Skip already-processed files (default True).

    Returns
    -------
    Dict[str, str]
        Mapping from scheme key (e.g. "scheme3_4bit") to its wav.scp path.

    Example
    -------
    >>> scp_map = process_wav_scp_all_schemes(
    ...     scp_path = "data/train/wav.scp",
    ...     out_dir  = "/data/quantized/train",
    ...     in_dir   = "/data/corpora",
    ...     jobs     = 8,
    ... )
    >>> scp_map["scheme1"]       # => "/data/quantized/train/scheme1/wav.scp"
    >>> scp_map["scheme3_4bit"]  # => "/data/quantized/train/scheme3_4bit/wav.scp"
    """
    combinations = [(1, 1), (6, 16)]   # (scheme, bit_depth) for fixed schemes
    for s in (2, 3, 4, 5):
        for bd in bit_depths:
            combinations.append((s, bd))
    combinations.sort()

    scp_map = {}
    for scheme, bit_depth in combinations:
        skey = _scheme_key(scheme, bit_depth)
        log.info("--- Starting %s ---", skey)
        out_scp = process_wav_scp(
            scp_path      = scp_path,
            out_dir       = out_dir,
            scheme        = scheme,
            bit_depth     = bit_depth,
            in_dir        = in_dir,
            target_sr     = target_sr,
            mu            = mu,
            a             = a,
            jobs          = jobs,
            skip_existing = skip_existing,
        )
        scp_map[skey] = out_scp

    log.info("All schemes complete. %d wav.scp files written.", len(scp_map))
    return scp_map


# ---------------------------------------------------------------------------
# Quantization info / metadata
# ---------------------------------------------------------------------------

SCHEME_INFO = {
    1: {
        "name"       : "1-bit Sign Quantization",
        "standard"   : "Threshold at 0; output in {+1, -1}",
        "bit_depths" : [1],
        "description": (
            "Binarises the signal using zero as the single decision boundary. "
            "Preserves only the zero-crossing sequence. All amplitude discarded. "
            "Shared 1-bit baseline across all five tasks."
        ),
    },
    2: {
        "name"       : "Uniform Mid-Tread Quantization",
        "standard"   : "Linear PCM; Delta = 2^(1-B)",
        "bit_depths" : [2, 4, 8],
        "description": (
            "Symmetric uniform quantizer. Step size Delta = 2^(1-B). "
            "Quantization error uniformly distributed across the amplitude range."
        ),
    },
    3: {
        "name"       : "mu-Law Companding + Uniform Quantization",
        "standard"   : f"ITU-T G.711; mu={MU}",
        "bit_depths" : [2, 4, 8],
        "description": (
            "Logarithmic compander allocating more levels to low-amplitude regions. "
            "Matches perceptual sensitivity of the human auditory system. "
            "At 1-bit reduces to sgn(x) -- identical to Scheme 1."
        ),
    },
    4: {
        "name"       : "A-Law Companding + Uniform Quantization",
        "standard"   : f"ITU-T G.711; A={A}",
        "bit_depths" : [2, 4, 8],
        "description": (
            "Piecewise compander: linear near origin, logarithmic elsewhere. "
            "More uniform SNR across dynamic range than mu-law. "
            "At 1-bit reduces to sgn(x) -- identical to Scheme 1."
        ),
    },
    5: {
        "name"       : "Logarithmic (Base-2) Quantization",
        "standard"   : "Diagnostic non-linear reference",
        "bit_depths" : [2, 4, 8],
        "description": (
            "Spaces reconstruction levels logarithmically: high resolution near zero, "
            "coarse at large amplitudes -- opposite emphasis to mu-law / A-law. "
            "Diagnostic reference to isolate the effect of perceptual level-placement."
        ),
    },
    6: {
        "name"       : "16-bit Full Precision",
        "standard"   : "No quantization; upper bound",
        "bit_depths" : [16],
        "description": (
            "Original 16-bit PCM signal with no quantization applied. "
            "Clean upper bound for all ablation curves across all five tasks."
        ),
    },
}


def describe_scheme(scheme: int) -> None:
    """Print a human-readable description of a quantization scheme."""
    if scheme not in SCHEME_INFO:
        raise ValueError(f"scheme must be one of {VALID_SCHEMES}, got {scheme}.")
    info = SCHEME_INFO[scheme]
    print(f"Scheme {scheme}: {info['name']}")
    print(f"  Standard   : {info['standard']}")
    print(f"  Bit depths : {info['bit_depths']}")
    print(f"  Description: {info['description']}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """
    Lightweight sanity checks -- run with:  python speech_quantization.py --test
    """
    import tempfile

    rng  = np.random.default_rng(42)
    # Intentionally raw (unnormalised) -- quantize() must handle this itself.
    x_raw = rng.standard_normal(16000) * 3.7   # amplitude >> 1
    x_norm = normalise(x_raw)                  # reference normalised version

    # --- Auto-normalisation: quantize() on a raw array must equal
    #     quantize() on the pre-normalised array (normalise_input=True is default) ---
    assert np.allclose(
        quantize(x_raw,  scheme=2, bit_depth=4),
        quantize(x_norm, scheme=2, bit_depth=4, normalise_input=False),
    ), "Auto-normalisation inside quantize() produced a different result than manual normalise()"
    print("Auto-normalisation            : PASS")

    # Use the normalised version as the reference signal for the remaining checks
    # (signs, ranges, and error bounds are all defined relative to [-1,+1] input).
    x = x_norm

    # --- Scheme 1: output must be in {-1, +1} ---
    x1 = quantize(x, scheme=1)
    assert set(np.unique(x1)).issubset({-1.0, 1.0}), "Scheme 1: values not in {-1,+1}"
    print("Scheme 1 (1-bit sign)         : PASS")

    # --- Scheme 2: clipped to valid range; quantization error <= delta/2 ---
    for bd in (2, 4, 8):
        delta = 2.0 ** (1 - bd)
        limit = 1.0 - delta / 2.0
        x2    = quantize(x, scheme=2, bit_depth=bd)
        assert np.all(np.abs(x2) <= limit + 1e-9), f"Scheme 2 ({bd}-bit): clip violated"
        assert x2.shape == x.shape,                f"Scheme 2 ({bd}-bit): shape mismatch"
        assert np.max(np.abs(x2 - np.clip(x, -limit, limit))) <= delta / 2 + 1e-9, \
            f"Scheme 2 ({bd}-bit): quantization error exceeds delta/2"
    print("Scheme 2 (uniform)            : PASS")

    # --- Scheme 3: mu-law sign is preserved; shape unchanged ---
    y3 = mu_law_compress(x)
    assert np.allclose(np.sign(y3), np.sign(x)), "mu-law sign preservation failed"
    for bd in (2, 4, 8):
        x3 = quantize(x, scheme=3, bit_depth=bd)
        assert x3.shape == x.shape, f"Scheme 3 ({bd}-bit): shape mismatch"
    print("Scheme 3 (mu-law + uniform)   : PASS")

    # --- Scheme 4: A-law sign is preserved; shape unchanged ---
    y4 = a_law_compress(x)
    assert np.allclose(np.sign(y4), np.sign(x)), "A-law sign preservation failed"
    for bd in (2, 4, 8):
        x4 = quantize(x, scheme=4, bit_depth=bd)
        assert x4.shape == x.shape, f"Scheme 4 ({bd}-bit): shape mismatch"
    print("Scheme 4 (A-law + uniform)    : PASS")

    # --- Scheme 5: shape, amplitude range, sign preserved ---
    for bd in (2, 4, 8):
        x5 = quantize(x, scheme=5, bit_depth=bd)
        assert x5.shape == x.shape,               f"Scheme 5 ({bd}-bit): shape mismatch"
        assert np.all(np.abs(x5) <= 1.0 + 1e-9), f"Scheme 5 ({bd}-bit): out of [-1,1]"
        assert np.all(np.sign(x5) == np.sign(x)), f"Scheme 5 ({bd}-bit): sign flipped"
    print("Scheme 5 (log base-2)         : PASS")

    # --- Scheme 5 (regression guards for the log-domain mapping) ---
    # The reconstruction must actually use the available levels: more bits must
    # yield more distinct magnitudes, and the result must NOT collapse onto the
    # 1-bit sign signal (the failure mode of the old log_norm sign bug).
    x5_2 = quantize(x, scheme=5, bit_depth=2)
    x5_4 = quantize(x, scheme=5, bit_depth=4)
    x5_8 = quantize(x, scheme=5, bit_depth=8)
    n_levels_2 = len(np.unique(np.abs(x5_2)))
    n_levels_4 = len(np.unique(np.abs(x5_4)))
    n_levels_8 = len(np.unique(np.abs(x5_8)))
    assert n_levels_2 > 1, "Scheme 5 (2-bit): only one magnitude level (collapsed to sign)"
    assert n_levels_4 > n_levels_2, "Scheme 5: 4-bit should resolve more levels than 2-bit"
    assert n_levels_8 >= n_levels_4, "Scheme 5: 8-bit should resolve at least as many as 4-bit"
    assert not np.allclose(x5_4, quantize(x, scheme=1)), \
        "Scheme 5 (4-bit) collapsed onto the 1-bit sign signal"
    print("Scheme 5 (log levels resolve) : PASS")

    # --- Scheme 6: returns the normalised input unchanged ---
    x6 = quantize(x, scheme=6)
    assert np.allclose(x, x6), "Scheme 6: not equal to normalised input"
    # Also confirm that passing a raw array still yields the normalised version
    x6_raw = quantize(x_raw, scheme=6)
    assert np.allclose(x6_raw, x_norm), "Scheme 6: auto-normalisation failed on raw input"
    print("Scheme 6 (full precision)     : PASS")

    # --- Key property: companding schemes diverge from uniform at 2-bit ---
    x2_2bit = quantize(x, scheme=2, bit_depth=2)
    x3_2bit = quantize(x, scheme=3, bit_depth=2)
    x4_2bit = quantize(x, scheme=4, bit_depth=2)
    assert not np.allclose(x2_2bit, x3_2bit), "Scheme 2 vs 3 at 2-bit should differ"
    assert not np.allclose(x2_2bit, x4_2bit), "Scheme 2 vs 4 at 2-bit should differ"
    print("Companding divergence at 2-bit: PASS")

    # --- quantize_all_schemes: correct keys; raw input normalised once ---
    results_raw  = quantize_all_schemes(x_raw)   # should auto-normalise
    results_norm = quantize_all_schemes(x_norm, normalise_input=False)
    expected_keys = (
        ["scheme1", "scheme6"] +
        [f"scheme{s}_{b}bit" for s in (2, 3, 4, 5) for b in (2, 4, 8)]
    )
    for k in expected_keys:
        assert k in results_raw, f"Missing key: {k}"
        assert np.allclose(results_raw[k], results_norm[k]), \
            f"quantize_all_schemes normalisation mismatch for {k}"
    print("quantize_all_schemes()        : PASS")

    # --- wav.scp round-trip test ---
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Create a minimal corpus tree
        corpus_dir = tmp / "corpus" / "lang1" / "spk1"
        corpus_dir.mkdir(parents=True)
        audio_path = corpus_dir / "utt1.wav"
        sf.write(str(audio_path), x.astype(np.float32), 16000)

        # Write a wav.scp
        scp_path = tmp / "wav.scp"
        with open(scp_path, "w") as fh:
            fh.write(f"# comment line\n")
            fh.write(f"\n")                          # blank line
            fh.write(f"lang1-spk1-utt1 {audio_path}\n")

        entries = read_wav_scp(str(scp_path))
        assert len(entries) == 1, f"Expected 1 entry, got {len(entries)}"
        assert entries[0][0] == "lang1-spk1-utt1"

        # _output_path: with in_dir
        out_dir  = str(tmp / "quantized")
        dst      = _output_path(str(audio_path), out_dir, "scheme1", str(tmp / "corpus"))
        expected = str(Path(out_dir) / "scheme1" / "lang1" / "spk1" / "utt1.wav")
        assert dst == expected, f"_output_path mismatch:\n  got      {dst}\n  expected {expected}"

        # _output_path: without in_dir (mirrors absolute path)
        dst_abs = _output_path(str(audio_path), out_dir, "scheme1", None)
        assert dst_abs.startswith(str(Path(out_dir) / "scheme1")), \
            f"_output_path (no indir) unexpected: {dst_abs}"

        # Full pipeline: process_wav_scp
        out_scp = process_wav_scp(
            scp_path  = str(scp_path),
            out_dir   = out_dir,
            scheme    = 1,
            bit_depth = 1,
            in_dir    = str(tmp / "corpus"),
            target_sr = 16000,
            jobs      = 1,
        )
        assert Path(out_scp).exists(), f"Output wav.scp not created: {out_scp}"
        out_entries = read_wav_scp(out_scp)
        assert len(out_entries) == 1, f"Output scp should have 1 entry"
        uttid, out_path = out_entries[0]
        assert uttid == "lang1-spk1-utt1"
        assert Path(out_path).exists(), f"Quantized file not found: {out_path}"

        # Verify the saved audio is sign-quantized
        x_saved, sr_saved = sf.read(out_path)
        assert sr_saved == 16000
        assert set(np.unique(np.sign(x_saved))).issubset({-1.0, 0.0, 1.0})
        assert set(np.unique(x_saved)).issubset({-1.0, 1.0}), \
            "Saved scheme-1 file should contain only {-1.0, +1.0}"

    print("wav.scp round-trip            : PASS")
    print("\nAll tests passed.")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "speech_quantization",
        description = "Quantize a Kaldi wav.scp corpus using the shared scheme framework.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--test", action="store_true",
        help="Run the internal self-tests and exit."
    )
    p.add_argument(
        "--scp", metavar="WAV_SCP",
        help="Path to the input Kaldi wav.scp file."
    )
    p.add_argument(
        "--out", metavar="OUT_DIR",
        help="Root directory for quantized output files."
    )
    p.add_argument(
        "--indir", metavar="IN_DIR", default=None,
        help=(
            "Common root of the source corpus. "
            "The relative path from IN_DIR is reproduced under OUT_DIR/<scheme>/. "
            "If omitted, the full absolute path is mirrored."
        ),
    )
    p.add_argument(
        "--scheme", type=int, choices=[1, 2, 3, 4, 5, 6], default=1,
        help="Quantization scheme. Ignored when --all-schemes is set."
    )
    p.add_argument(
        "--bit-depth", type=int, choices=[2, 4, 8], default=2,
        dest="bit_depth",
        help="Bit depth for schemes 2-5. Ignored for schemes 1 and 6."
    )
    p.add_argument(
        "--all-schemes", action="store_true",
        help="Process ALL scheme/depth combinations (overrides --scheme and --bit-depth)."
    )
    p.add_argument(
        "--target-sr", type=int, default=16000, dest="target_sr",
        help="Resample audio to this sample rate. Set to 0 to keep native rate."
    )
    p.add_argument(
        "--jobs", type=int, default=4,
        help="Number of parallel worker processes."
    )
    p.add_argument(
        "--no-skip", action="store_true",
        help="Re-process files even if the output already exists."
    )
    p.add_argument(
        "--fail-fast", action="store_true",
        help="Abort immediately on the first per-file error."
    )
    return p


def main(argv=None):
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.test:
        _run_tests()
        return

    if not args.scp or not args.out:
        parser.error("--scp and --out are required (unless --test is given).")

    target_sr     = args.target_sr if args.target_sr > 0 else None
    skip_existing = not args.no_skip

    if args.all_schemes:
        process_wav_scp_all_schemes(
            scp_path      = args.scp,
            out_dir       = args.out,
            in_dir        = args.indir,
            target_sr     = target_sr,
            jobs          = args.jobs,
            skip_existing = skip_existing,
        )
    else:
        process_wav_scp(
            scp_path      = args.scp,
            out_dir       = args.out,
            scheme        = args.scheme,
            bit_depth     = args.bit_depth,
            in_dir        = args.indir,
            target_sr     = target_sr,
            jobs          = args.jobs,
            skip_existing = skip_existing,
            fail_on_error = args.fail_fast,
        )


if __name__ == "__main__":
    main()
