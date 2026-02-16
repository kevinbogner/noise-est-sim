from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import random
import secrets
import warnings
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import seaborn as sns  # type: ignore
from matplotlib import colormaps  # pyright: ignore[reportMissingImports]
from matplotlib.axes import Axes  # pyright: ignore[reportMissingImports]
from matplotlib.colors import to_rgba  # pyright: ignore[reportMissingImports]
from matplotlib.lines import Line2D  # pyright: ignore[reportMissingImports]
from matplotlib.ticker import (  # pyright: ignore[reportMissingImports]
    AutoMinorLocator,
    FuncFormatter,
    MaxNLocator,
)

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
    category=Warning,
)

try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore

PALETTE_SPEC = [
    ("blue", "#0173b2"),
    ("orange", "#de8f05"),
    ("green", "#029e73"),
    ("red", "#d55e00"),
    ("purple", "#cc78bc"),
    ("brown", "#ca9161"),
    ("pink", "#fbafe4"),
    ("grey", "#949494"),
    ("yellow", "#ece133"),
    ("cyan", "#56b4e9"),
]
PALETTE_COLOR_NAMES = [name for name, _ in PALETTE_SPEC]


def use_pub_theme(font: str = "Avenir", context: str = "talk") -> None:
    """Apply a publication-style theme once for all figures."""

    palette_colors = [hex_code for _, hex_code in PALETTE_SPEC]
    sns.set_theme(context=context, style="ticks", palette=palette_colors, font=font)
    plt.rcParams.update(
        {
            "figure.dpi": 320,
            "savefig.dpi": 320,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.labelweight": "medium",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.grid": True,
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "font.family": [font, "DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "mathtext.fontset": "stix",
        }
    )


use_pub_theme()

DEFAULT_LINEWIDTH = 1.4
DEFAULT_MARKERSIZE = 5
GRID_ALPHA = 0.25

try:
    import perceval as pcvl  # type: ignore
except Exception as exc:  # pragma: no cover - Perceval is required
    raise SystemExit(
        "Perceval is required for noise_est; install perceval-quandela before running."
    ) from exc


def wilson_bounds(phat: float, n: int, z: float) -> Tuple[float, float]:
    """Wilson interval bounds for a Bernoulli rate with precomputed z.

    Returns (L, U). For n<=0, returns (0.0, 1.0).
    """
    if n <= 0:
        return (0.0, 1.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = phat + z2 / (2.0 * n)
    radicand = (phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n))
    radius = z * math.sqrt(max(0.0, radicand))
    U = (center + radius) / denom
    L = max(0.0, (center - radius) / denom)
    return (L, U)


def inv_std_normal_cdf(p: float) -> float:
    """Inverse CDF (quantile) for standard normal; Acklam's approximation.

    Accurate to ~1e-9 for p in (0,1)."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )

    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def derive_seed(key: str, batch_id: str, salt: str = "pcvl-noise-est") -> int:
    """Derive a deterministic 64-bit seed using HMAC-SHA256(key, batch_id||salt).

    If key is empty, fall back to SHA-256 over (batch_id||salt).
    Accepts ASCII or hex-encoded keys.
    """
    material = (batch_id + "|" + salt).encode("utf-8")
    if key:
        kclean = key.replace(" ", "")
        if all(c in "0123456789abcdefABCDEF" for c in kclean) and len(kclean) % 2 == 0:
            kb = bytes.fromhex(kclean)
        else:
            kb = key.encode("utf-8")
        tag = hmac.new(kb, material, hashlib.sha256).digest()
    else:
        tag = hashlib.sha256(material).digest()
    return int.from_bytes(tag[:8], "big", signed=False)


def _decode_qkd_key_material(key: str) -> bytes | None:
    """Decode ASCII/hex QKD key material once for repeated HMAC operations."""
    if not key:
        return None
    kclean = key.replace(" ", "")
    if all(c in "0123456789abcdefABCDEF" for c in kclean) and len(kclean) % 2 == 0:
        return bytes.fromhex(kclean)
    return key.encode("utf-8")


def commit_open_outcome(
    key: str,
    commander: Tuple[int, int],
    reported: Sequence[int],
    *,
    key_material: bytes | None = None,
) -> None:
    """Simulate HMAC commit&open for one round outcome.

    No-op if key is empty. Raises RuntimeError on MAC mismatch (should not happen here).
    """
    kb = key_material if key_material is not None else _decode_qkd_key_material(key)
    if kb is None:
        return
    msg = bytes([commander[0], commander[1]] + list(reported))
    tag = hmac.new(kb, msg, hashlib.sha256).digest()

    tag2 = hmac.new(kb, msg, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, tag2):
        raise RuntimeError("Commit/open verification failed for a round outcome")


@dataclass
class SimulationConfig:
    num_generals: int
    shots: int
    verify_fraction: float
    epsilon_threshold: float
    delta: float
    seed: int
    flag_rule: "FlagRule"
    engine: Literal["fast", "perceval"] = "perceval"

    windows: int = 1
    quarantine_w: int = 1
    readmit_eps_frac: float = 0.5

    epsilon0: float | None = None
    epsilon1: float | None = None
    epsilon_delta: float | None = None
    qkd_key: str = ""
    batch_id: str = "batch-0"
    pcvl_noise_model: "pcvl.NoiseModel | None" = None  # type: ignore[name-defined]
    pcvl_dark_scale: float = 1.0
    dcr_hz_list: Sequence[float] | None = None
    gate_ns_list: Sequence[float] | None = None
    p_phys_cmd: tuple[float, float] = (0.01, 0.01)
    dcr_hz_cmd: tuple[float, float] = (0.0, 0.0)
    gate_ns_cmd: tuple[float, float] = (0.0, 0.0)
    tau_leak: float = 0.135
    tau_fid: float = 0.89


class FlagRule(Enum):
    UPPER = auto()
    POINT = auto()
    LOWER = auto()
    SPLIT = auto()


@dataclass
class NodeCounters:
    n_det: int = 0
    m_det: int = 0
    n1: int = 0
    m1: int = 0
    n0: int = 0
    m0: int = 0
    reported_counts: List[List[int]] = field(default_factory=lambda: [[0, 0], [0, 0]])
    calibration_counts: List[List[int]] = field(
        default_factory=lambda: [[0, 0], [0, 0]]
    )


@dataclass
class NodeStats:
    e: float
    p_hat: float
    U: float
    L: float
    U1: float
    L1: float
    U0: float
    L0: float
    LDelta: float
    eta_hat: float
    pdark_hat: float
    flagged: bool
    classification: List[str]
    raw_flagged: bool
    raw_classification: List[str]
    raw_e1: float
    raw_e0: float
    corrected_e1: float
    corrected_e0: float
    meter_dominated: bool
    calibration_matrix: List[List[float]]
    calibration_inverse: List[List[float]]


@dataclass
class NodeSummary:
    n_det_per_l: List[int]
    mismatches_per_l: List[int]
    e_l: List[float]
    p_hat: List[float]
    u_bound: List[float]
    l_bound: List[float]
    u1_bound: List[float]
    l1_bound: List[float]
    u0_bound: List[float]
    l0_bound: List[float]
    flagged: List[int]
    e1: List[float]
    e0: List[float]
    eta_hat: List[float]
    pdark_hat: List[float]
    raw_e1: List[float]
    raw_e0: List[float]
    classifications: List[List[str]]
    raw_classifications: List[List[str]]
    meter_dominated: List[bool]
    calibration_matrix_per_l: List[List[List[float]]]
    calibration_inverse_per_l: List[List[List[float]]]

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_det_per_l": self.n_det_per_l,
            "mismatches_per_l": self.mismatches_per_l,
            "e_l": self.e_l,
            "p_hat": self.p_hat,
            "u_bound": self.u_bound,
            "l_bound": self.l_bound,
            "u1_bound": self.u1_bound,
            "l1_bound": self.l1_bound,
            "u0_bound": self.u0_bound,
            "l0_bound": self.l0_bound,
            "flagged": self.flagged,
            "e1": self.e1,
            "e0": self.e0,
            "eta_hat": self.eta_hat,
            "pdark_hat": self.pdark_hat,
            "raw_e1": self.raw_e1,
            "raw_e0": self.raw_e0,
            "classifications": self.classifications,
            "raw_classifications": self.raw_classifications,
            "meter_dominated": self.meter_dominated,
            "calibration_matrix_per_l": self.calibration_matrix_per_l,
            "calibration_inverse_per_l": self.calibration_inverse_per_l,
        }


def build_node_summary(
    counters: Sequence[NodeCounters], stats: Sequence[NodeStats]
) -> NodeSummary:
    """Convert counters and stats into the legacy summary structure."""
    n_det = [c.n_det for c in counters]
    mismatches = [c.m_det for c in counters]
    e_l = [s.e for s in stats]
    p_hat = [s.p_hat for s in stats]
    u_bound = [s.U for s in stats]
    l_bound = [s.L for s in stats]
    u1_bound = [s.U1 for s in stats]
    l1_bound = [s.L1 for s in stats]
    u0_bound = [s.U0 for s in stats]
    l0_bound = [s.L0 for s in stats]
    flagged = [i for i, s in enumerate(stats) if s.flagged]
    e1 = [s.corrected_e1 for s in stats]
    e0 = [s.corrected_e0 for s in stats]
    eta_hat = [s.eta_hat for s in stats]
    pdark_hat = [s.pdark_hat for s in stats]
    raw_e1 = [s.raw_e1 for s in stats]
    raw_e0 = [s.raw_e0 for s in stats]
    classifications = [list(s.classification) for s in stats]
    raw_classifications = [list(s.raw_classification) for s in stats]
    meter_dominated = [s.meter_dominated for s in stats]
    calibration_matrix_per_l = [
        [list(row) for row in s.calibration_matrix] for s in stats
    ]
    calibration_inverse_per_l = [
        [list(row) for row in s.calibration_inverse] for s in stats
    ]
    return NodeSummary(
        n_det_per_l=n_det,
        mismatches_per_l=mismatches,
        e_l=e_l,
        p_hat=p_hat,
        u_bound=u_bound,
        l_bound=l_bound,
        u1_bound=u1_bound,
        l1_bound=l1_bound,
        u0_bound=u0_bound,
        l0_bound=l0_bound,
        flagged=flagged,
        e1=e1,
        e0=e0,
        eta_hat=eta_hat,
        pdark_hat=pdark_hat,
        raw_e1=raw_e1,
        raw_e0=raw_e0,
        classifications=classifications,
        raw_classifications=raw_classifications,
        meter_dominated=meter_dominated,
        calibration_matrix_per_l=calibration_matrix_per_l,
        calibration_inverse_per_l=calibration_inverse_per_l,
    )


def expand_summary_to_full(
    summary: NodeSummary, total_nodes: int, active_indices: Sequence[int]
) -> NodeSummary:
    """Embed a partial summary back into the full lieutenant index space."""
    active = list(active_indices)
    n_det = [0] * total_nodes
    mismatches = [0] * total_nodes
    e_l = [0.0] * total_nodes
    p_hat = [0.0] * total_nodes
    u_bound = [1.0] * total_nodes
    l_bound = [0.0] * total_nodes
    u1_bound = [0.0] * total_nodes
    l1_bound = [0.0] * total_nodes
    u0_bound = [1.0] * total_nodes
    l0_bound = [0.0] * total_nodes
    e1 = [0.0] * total_nodes
    e0 = [0.0] * total_nodes
    eta_hat = [0.0] * total_nodes
    pdark_hat = [0.0] * total_nodes
    raw_e1 = [0.0] * total_nodes
    raw_e0 = [0.0] * total_nodes
    classifications: List[List[str]] = [[] for _ in range(total_nodes)]
    raw_classifications: List[List[str]] = [[] for _ in range(total_nodes)]
    meter_dominated = [False] * total_nodes
    calibration_matrix_per_l = [[[1.0, 0.0], [0.0, 1.0]] for _ in range(total_nodes)]
    calibration_inverse_per_l = [[[1.0, 0.0], [0.0, 1.0]] for _ in range(total_nodes)]

    for local_idx, global_idx in enumerate(active):
        n_det[global_idx] = summary.n_det_per_l[local_idx]
        mismatches[global_idx] = summary.mismatches_per_l[local_idx]
        e_l[global_idx] = summary.e_l[local_idx]
        p_hat[global_idx] = summary.p_hat[local_idx]
        u_bound[global_idx] = summary.u_bound[local_idx]
        l_bound[global_idx] = summary.l_bound[local_idx]
        u1_bound[global_idx] = summary.u1_bound[local_idx]
        l1_bound[global_idx] = summary.l1_bound[local_idx]
        u0_bound[global_idx] = summary.u0_bound[local_idx]
        l0_bound[global_idx] = summary.l0_bound[local_idx]
        e1[global_idx] = summary.e1[local_idx]
        e0[global_idx] = summary.e0[local_idx]
        eta_hat[global_idx] = summary.eta_hat[local_idx]
        pdark_hat[global_idx] = summary.pdark_hat[local_idx]
        raw_e1[global_idx] = summary.raw_e1[local_idx]
        raw_e0[global_idx] = summary.raw_e0[local_idx]
        classifications[global_idx] = list(summary.classifications[local_idx])
        raw_classifications[global_idx] = list(summary.raw_classifications[local_idx])
        meter_dominated[global_idx] = summary.meter_dominated[local_idx]
        calibration_matrix_per_l[global_idx] = [
            list(row) for row in summary.calibration_matrix_per_l[local_idx]
        ]
        calibration_inverse_per_l[global_idx] = [
            list(row) for row in summary.calibration_inverse_per_l[local_idx]
        ]

    flagged = [active[idx] for idx in summary.flagged]
    return NodeSummary(
        n_det_per_l=n_det,
        mismatches_per_l=mismatches,
        e_l=e_l,
        p_hat=p_hat,
        u_bound=u_bound,
        l_bound=l_bound,
        u1_bound=u1_bound,
        l1_bound=l1_bound,
        u0_bound=u0_bound,
        l0_bound=l0_bound,
        flagged=flagged,
        e1=e1,
        e0=e0,
        eta_hat=eta_hat,
        pdark_hat=pdark_hat,
        raw_e1=raw_e1,
        raw_e0=raw_e0,
        classifications=classifications,
        raw_classifications=raw_classifications,
        meter_dominated=meter_dominated,
        calibration_matrix_per_l=calibration_matrix_per_l,
        calibration_inverse_per_l=calibration_inverse_per_l,
    )


def build_psi_n_statevector(n: int) -> "pcvl.StateVector":  # type: ignore[name-defined]
    """
    Build |Psi_n> as a Perceval StateVector over modes [C0, C1, L1, ..., L_{n-1}].
    Encoding: bit 1 -> |1>, bit 0 -> |0> per mode. Amplitudes match Sec. A.
    """
    if pcvl is None:
        raise RuntimeError("Perceval engine requested but perceval is not installed")
    import math as _math

    if n < 3:
        raise ValueError("n must be >= 3 (Commander + >=2 Lieutenants)")

    m = 2 + (n - 1)
    _ = m
    sv = pcvl.StateVector()
    a_det = 1.0 / _math.sqrt(3.0)
    a_prob = 1.0 / _math.sqrt(3.0) / _math.sqrt(2.0 * (n - 1))

    sv += a_det * pcvl.StateVector(pcvl.BasicState([0, 0] + [1] * (n - 1)))

    sv += a_det * pcvl.StateVector(pcvl.BasicState([1, 1] + [0] * (n - 1)))

    for i in range(n - 1):
        occ_01 = [0, 1] + [0] * (n - 1)
        occ_01[2 + i] = 1
        sv += a_prob * pcvl.StateVector(pcvl.BasicState(occ_01))
        occ_10 = [1, 0] + [1] * (n - 1)
        occ_10[2 + i] = 0
        sv += a_prob * pcvl.StateVector(pcvl.BasicState(occ_10))

    sv.normalize()
    return sv


def build_processor_for_psi(
    n: int,
    p_phys_list: list[float],
    noise: "pcvl.NoiseModel | None" = None,  # type: ignore[name-defined]
    p_phys_cmd: tuple[float, float] = (0.0, 0.0),
) -> "pcvl.Processor":  # type: ignore[name-defined]
    """Identity m-mode processor with LC(loss) on Commander and lieutenants."""
    if pcvl is None:
        raise RuntimeError("Perceval engine requested but perceval is not installed")
    if n < 3:
        raise ValueError("n must be >= 3 (Commander + >=2 Lieutenants)")
    if len(p_phys_list) != (n - 1):
        raise ValueError("p_phys_list must have length n-1 (one per lieutenant)")
    m = 2 + (n - 1)
    proc = pcvl.Processor("SLOS", m, noise)
    try:
        proc.min_detected_photons_filter(0)
    except Exception:
        pass
    c_losses = (
        max(0.0, min(1.0, float(p_phys_cmd[0]))) if len(p_phys_cmd) > 0 else 0.0,
        max(0.0, min(1.0, float(p_phys_cmd[1]))) if len(p_phys_cmd) > 1 else 0.0,
    )
    for mode, loss in enumerate(c_losses):
        if loss > 0.0:
            proc.add(mode, pcvl.LC(loss))

    for i, loss in enumerate(p_phys_list):
        loss = max(0.0, min(1.0, float(loss)))
        if loss > 0.0:
            proc.add(2 + i, pcvl.LC(loss))
    return proc


def sample_rounds_perceval(
    n: int,
    shots: int,
    p_phys_list: list[float],
    q_class_list: list[float],
    *,
    seed: int | None = None,
    noise_model: "pcvl.NoiseModel | None" = None,  # type: ignore[name-defined]
    dark_scale: float = 1.0,
    dcr_hz_list: Sequence[float] | None = None,
    gate_ns_list: Sequence[float] | None = None,
    dcr_hz_cmd: tuple[float, float] | None = None,
    gate_ns_cmd: tuple[float, float] | None = None,
    p_phys_cmd: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[tuple[int, int], list[int], list[int]]]:
    """
    Sample full outcomes from Perceval for all modes, then apply classical flips.
    Returns a list of ((c0,c1), physical_bits[L], reported_bits[L]).
    """
    if pcvl is None:
        raise RuntimeError("Perceval engine requested but perceval is not installed")
    if hasattr(pcvl, "seed") and seed is not None:
        try:
            pcvl.seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass
    psi = build_psi_n_statevector(n)
    proc = build_processor_for_psi(
        n,
        list(p_phys_list),
        noise=noise_model,
        p_phys_cmd=p_phys_cmd,
    )
    proc.with_input(psi)
    sampler = pcvl.algorithm.Sampler(proc)
    res = sampler.samples(shots)
    samples = res["results"]  # type: ignore[index]
    rnd = random.Random(seed)
    L = n - 1
    q_len = len(q_class_list)
    q_probs = [float(q_class_list[i]) for i in range(q_len)]

    dark_probs_l = [0.0] * L
    if L:
        d_len = len(dcr_hz_list) if dcr_hz_list is not None else 0
        g_len = len(gate_ns_list) if gate_ns_list is not None else 0
        for i in range(L):
            dcr = (
                float(dcr_hz_list[i]) if dcr_hz_list is not None and i < d_len else 0.0
            )
            gate_ns = (
                float(gate_ns_list[i])
                if gate_ns_list is not None and i < g_len
                else 0.0
            )
            rate = dark_scale * dcr * (gate_ns * 1e-9)
            dark_probs_l[i] = (
                min(1.0, max(0.0, 1.0 - math.exp(-rate))) if rate > 0.0 else 0.0
            )

    dark_prob_c0 = 0.0
    dark_prob_c1 = 0.0
    dark_cmd_enabled0 = False
    dark_cmd_enabled1 = False
    if dcr_hz_cmd is not None and gate_ns_cmd is not None:
        d0 = max(0.0, float(dcr_hz_cmd[0])) if len(dcr_hz_cmd) > 0 else 0.0
        d1 = max(0.0, float(dcr_hz_cmd[1])) if len(dcr_hz_cmd) > 1 else 0.0
        g0 = max(0.0, float(gate_ns_cmd[0])) if len(gate_ns_cmd) > 0 else 0.0
        g1 = max(0.0, float(gate_ns_cmd[1])) if len(gate_ns_cmd) > 1 else 0.0
        dark_cmd_enabled0 = d0 > 0.0 and g0 > 0.0
        dark_cmd_enabled1 = d1 > 0.0 and g1 > 0.0
        if dark_cmd_enabled0:
            rate0 = dark_scale * d0 * (g0 * 1e-9)
            dark_prob_c0 = min(1.0, max(0.0, 1.0 - math.exp(-max(0.0, rate0))))
        if dark_cmd_enabled1:
            rate1 = dark_scale * d1 * (g1 * 1e-9)
            dark_prob_c1 = min(1.0, max(0.0, 1.0 - math.exp(-max(0.0, rate1))))

    rounds: list[tuple[tuple[int, int], list[int], list[int]]] = []
    for bs in samples:
        c0 = int(int(bs[0]) > 0)  # type: ignore[index]
        c1 = int(int(bs[1]) > 0)  # type: ignore[index]
        if c0 == 0 and dark_cmd_enabled0 and rnd.random() < dark_prob_c0:
            c0 = 1
        if c1 == 0 and dark_cmd_enabled1 and rnd.random() < dark_prob_c1:
            c1 = 1

        phys = [int(int(bs[2 + i]) > 0) for i in range(L)]  # type: ignore[index]

        for i in range(L):
            if phys[i] == 0:
                # Preserve RNG draw pattern: one dark-count draw per zero bit.
                if rnd.random() < dark_probs_l[i]:
                    phys[i] = 1
        rep = phys.copy()
        for i in range(L):
            if i < q_len and rnd.random() < q_probs[i]:
                rep[i] ^= 1
        rounds.append(((c0, c1), phys, rep))
    return rounds


def _compute_pdark_array(
    L: int,
    dark_scale: float,
    dcr_hz_list: Sequence[float] | None,
    gate_ns_list: Sequence[float] | None,
) -> np.ndarray:
    dcr = (
        np.zeros(L, dtype=np.float64)
        if dcr_hz_list is None
        else np.array(
            [float(dcr_hz_list[i]) if i < len(dcr_hz_list) else 0.0 for i in range(L)],
            dtype=np.float64,
        )
    )
    gate_ns = (
        np.zeros(L, dtype=np.float64)
        if gate_ns_list is None
        else np.array(
            [
                float(gate_ns_list[i]) if i < len(gate_ns_list) else 0.0
                for i in range(L)
            ],
            dtype=np.float64,
        )
    )
    rate = dark_scale * dcr * (gate_ns * 1e-9)
    return 1.0 - np.exp(-np.clip(rate, 0.0, None))


def sample_arrays_fast(
    n: int,
    shots: int,
    p_phys_list: Sequence[float],
    q_class_list: Sequence[float],
    *,
    seed: int | None = None,
    dark_scale: float = 1.0,
    dcr_hz_list: Sequence[float] | None = None,
    gate_ns_list: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n < 3:
        raise ValueError("n must be >= 3")
    L = n - 1
    if len(p_phys_list) != L or len(q_class_list) != L:
        raise ValueError("p_phys_list and q_class_list must have length n-1")

    rng = np.random.default_rng(seed)

    thresholds = np.array([1.0 / 3.0, 2.0 / 3.0, 5.0 / 6.0, 1.0], dtype=np.float64)
    codes = np.searchsorted(thresholds, rng.random(shots), side="right")

    c0 = np.where(
        codes == 0,
        0,
        np.where(codes == 1, 1, np.where(codes == 2, 0, 1)),
    ).astype(np.uint8)
    c1 = np.where(
        codes == 0,
        0,
        np.where(codes == 1, 1, np.where(codes == 2, 1, 0)),
    ).astype(np.uint8)

    ideal = np.zeros((shots, L), dtype=np.uint8)
    rows_00 = np.where(codes == 0)[0]
    rows_01 = np.where(codes == 2)[0]
    rows_10 = np.where(codes == 3)[0]

    if rows_00.size:
        ideal[rows_00, :] = 1
    if rows_10.size:
        ideal[rows_10, :] = 1
        j = rng.integers(0, L, size=rows_10.size)
        ideal[rows_10, j] = 0
    if rows_01.size:
        j = rng.integers(0, L, size=rows_01.size)
        ideal[rows_01, j] = 1

    phys = ideal.copy()

    p_loss = np.clip(np.asarray(p_phys_list, dtype=np.float64), 0.0, 1.0)[None, :]
    if L:
        loss_mask = (phys == 1) & (rng.random((shots, L)) < p_loss)
        phys[loss_mask] = 0

    p_dark = _compute_pdark_array(L, dark_scale, dcr_hz_list, gate_ns_list)[None, :]
    if L:
        dark_mask = (phys == 0) & (rng.random((shots, L)) < p_dark)
        phys[dark_mask] = 1

    q_class = np.clip(np.asarray(q_class_list, dtype=np.float64), 0.0, 1.0)[None, :]
    if L:
        flips = (rng.random((shots, L)) < q_class).astype(np.uint8)
    else:
        flips = np.empty((shots, 0), dtype=np.uint8)
    reported = phys ^ flips

    return c0, c1, phys, reported


def compute_commander_counts_arrays(c0: np.ndarray, c1: np.ndarray) -> Dict[str, int]:
    return {
        "00": int(np.sum((c0 == 0) & (c1 == 0))),
        "11": int(np.sum((c0 == 1) & (c1 == 1))),
        "01": int(np.sum((c0 == 0) & (c1 == 1))),
        "10": int(np.sum((c0 == 1) & (c1 == 0))),
    }


def _rounds_to_arrays(
    rounds: Sequence[Tuple[Tuple[int, int], List[int], List[int]]],
    num_lieutenants: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shots = len(rounds)
    c0 = np.empty(shots, dtype=np.uint8)
    c1 = np.empty(shots, dtype=np.uint8)
    phys = np.empty((shots, num_lieutenants), dtype=np.uint8)
    reported = np.empty((shots, num_lieutenants), dtype=np.uint8)
    for idx, ((cc0, cc1), phys_bits, rep_bits) in enumerate(rounds):
        c0[idx] = np.uint8(cc0)
        c1[idx] = np.uint8(cc1)
        phys[idx, :] = phys_bits[:num_lieutenants]
        reported[idx, :] = rep_bits[:num_lieutenants]
    return c0, c1, phys, reported


def accumulate_counters_fast(
    c0: np.ndarray,
    c1: np.ndarray,
    phys: np.ndarray,
    reported: np.ndarray,
    verify_indices: Sequence[int],
) -> List[NodeCounters]:
    idx = np.asarray(list(verify_indices), dtype=np.int64)
    L = reported.shape[1]
    C = [NodeCounters() for _ in range(L)]

    r_verify = reported[idx, :]
    p_verify = phys[idx, :]
    c0_v = c0[idx]
    c1_v = c1[idx]

    mask00 = (c0_v == 0) & (c1_v == 0)
    mask11 = (c0_v == 1) & (c1_v == 1)
    s_det = mask00 | mask11
    b_vec = np.where(mask00, 1, 0).astype(np.uint8)

    for li in range(L):
        cnt = C[li]
        r_col = r_verify[:, li]
        p_col = p_verify[:, li]
        for rb in (0, 1):
            for pb in (0, 1):
                cnt.calibration_counts[rb][pb] = int(
                    np.count_nonzero((r_col == rb) & (p_col == pb))
                )

        if not np.any(s_det):
            continue
        r_det = r_col[s_det]
        b_det = b_vec[s_det]
        cnt.n_det = int(r_det.size)
        cnt.m_det = int(np.count_nonzero(r_det != b_det))

        mask1 = b_det == 1
        mask0 = ~mask1

        if np.any(mask1):
            r1 = r_det[mask1]
            cnt.n1 = int(r1.size)
            cnt.m1 = int(np.count_nonzero(r1 == 0))
            cnt.reported_counts[1][0] = int(np.count_nonzero(r1 == 0))
            cnt.reported_counts[1][1] = int(np.count_nonzero(r1 == 1))

        if np.any(mask0):
            r0 = r_det[mask0]
            cnt.n0 = int(r0.size)
            cnt.m0 = int(np.count_nonzero(r0 == 1))
            cnt.reported_counts[0][0] = int(np.count_nonzero(r0 == 0))
            cnt.reported_counts[0][1] = int(np.count_nonzero(r0 == 1))

    return C


def _smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple centered moving-average smoother that preserves endpoints.

    Falls back to the input if there are too few points or finite samples."""

    effective_window = window if values.size >= 7 else 1
    if effective_window <= 1 or values.size <= 2:
        return values
    mask = np.isfinite(values)
    if mask.sum() <= 1:
        return values

    filled = values.astype(float, copy=True)
    if not mask.all():
        valid_idx = np.flatnonzero(mask)
        filled[~mask] = np.interp(np.flatnonzero(~mask), valid_idx, filled[mask])

    left = effective_window // 2
    right = effective_window - left - 1
    padded = np.pad(filled, (left, right), mode="edge")
    kernel = np.full(effective_window, 1.0 / effective_window, dtype=float)
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _summary_triplet(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    mask = np.isfinite(arr)
    if not mask.any():
        return (np.nan, np.nan, np.nan)
    finite = arr[mask]
    return (
        float(np.mean(finite, dtype=float)),
        float(np.min(finite)),
        float(np.max(finite)),
    )


def _mism(bit: int, truth: int) -> int:
    return bit ^ truth


def _parse_flag_rule(s: Optional[str]) -> FlagRule:
    if not s:
        return FlagRule.UPPER
    s2 = s.strip().lower()
    if s2 == "upper":
        return FlagRule.UPPER
    if s2 == "point":
        return FlagRule.POINT
    if s2 == "lower":
        return FlagRule.LOWER
    if s2 == "split":
        return FlagRule.SPLIT
    return FlagRule.UPPER


def accumulate_counters(
    rounds: Sequence[Tuple[Tuple[int, int], List[int], List[int]]],
    verify_indices: Sequence[int],
    num_lieutenants: int,
) -> List[NodeCounters]:
    """Accumulate per-node counters on S_det only (Commander outcomes 00/11)."""
    C = [NodeCounters() for _ in range(num_lieutenants)]
    for idx in verify_indices:
        (c0, c1), _physical, reported = rounds[idx]
        phys_bits = _physical
        for li in range(num_lieutenants):
            cnt = C[li]
            phys_bit = phys_bits[li]
            rep_bit = reported[li]
            cnt.calibration_counts[rep_bit][phys_bit] += 1
        if (c0, c1) in ((0, 0), (1, 1)):
            b = 1 if (c0, c1) == (0, 0) else 0
            for li in range(num_lieutenants):
                cnt = C[li]
                cnt.n_det += 1
                mismatch = _mism(reported[li], b)
                cnt.m_det += mismatch
                if b == 1:
                    cnt.n1 += 1
                    cnt.m1 += mismatch
                    cnt.reported_counts[1][reported[li]] += 1
                else:
                    cnt.n0 += 1
                    cnt.m0 += mismatch
                    cnt.reported_counts[0][reported[li]] += 1
    return C


_STRUCTURAL_NOISY = {"Flagged (noisy)", "Loss-biased (1$\to$0)"}

Color = Tuple[float, float, float]
COLOR_BY_NAME: Dict[str, Color] = {
    name: tuple(to_rgba(hex_code)[:3]) for name, hex_code in PALETTE_SPEC
}
COLOR_HEX_BY_NAME: Dict[str, str] = {name: hex_code for name, hex_code in PALETTE_SPEC}


# Helper to keep palette access consistent with analysis notebooks.
def palette_color(name: str) -> Color:
    return COLOR_BY_NAME.get(name, COLOR_BY_NAME[PALETTE_COLOR_NAMES[0]])


def palette_hex(name: str) -> str:
    return COLOR_HEX_BY_NAME.get(name, COLOR_HEX_BY_NAME[PALETTE_COLOR_NAMES[0]])


# Role colors derived from the shared palette so plots stay consistent.
LEAKAGE_COLOR: Color = COLOR_BY_NAME["blue"]
FIDELITY_COLOR: Color = COLOR_BY_NAME["green"]
ACCENT_COLOR: Color = COLOR_BY_NAME["red"]
PROBATION_COLOR: Color = COLOR_BY_NAME["purple"]
NEUTRAL_COLOR: Color = COLOR_BY_NAME["grey"]
THRESHOLD_COLOR: Color = COLOR_BY_NAME["brown"]

THRESHOLD_LINE_STYLE = {"color": THRESHOLD_COLOR, "linestyle": "--", "linewidth": 1.2}

MARKERS: Dict[str, str] = {
    "both_accept": "o",
    "baseline_only": "s",
    "both_reject": "x",
    "ours_only": "D",
    "trend": "None",
}

SERIES_MARKERS: Tuple[str, ...] = ("o", "s", "D", "^", "v", "P", "X", "h")
SERIES_COLORS: Tuple[Color, ...] = tuple(
    COLOR_BY_NAME[name] for name in PALETTE_COLOR_NAMES
)
VERIFY_TARGET_MAX = 6000


def _series_style(series_index: int) -> Tuple[Color, str]:
    color = SERIES_COLORS[series_index % len(SERIES_COLORS)]
    marker = SERIES_MARKERS[series_index % len(SERIES_MARKERS)]
    return color, marker


def _downsample_verify_targets(
    targets: Sequence[int], *, max_points: int, include: Sequence[int] | None = None
) -> List[int]:
    arr = np.array(sorted({int(v) for v in targets if v > 0}), dtype=int)
    if arr.size == 0 or arr.size <= max_points:
        return arr.tolist()

    include_set = {int(v) for v in (include or []) if v is not None}
    include_set &= set(arr.tolist())

    keep: Set[int] = set(include_set)
    keep.add(int(arr[0]))
    keep.add(int(arr[-1]))

    if len(keep) < max_points:
        grid = np.geomspace(arr[0], arr[-1], num=max_points).round().astype(int)
        for val in grid:
            nearest = int(arr[np.abs(arr - val).argmin()])
            keep.add(nearest)
            if len(keep) >= max_points:
                break

    return sorted(keep)[:max_points]


def _configure_verification_axis(
    ax: Axes,
    values: Sequence[float],
    *,
    x_max_floor: Optional[float] = None,
    start_at_zero: bool = True,
    use_data_ticks: bool = False,
    max_data_ticks: Optional[int] = None,
) -> None:
    arr = np.asarray(sorted(set(float(v) for v in values)), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    vmax = float(np.max(arr))
    vmax = min(3000.0, max(1.0, vmax))
    if x_max_floor is not None:
        vmax = max(vmax, float(x_max_floor))
    visible = arr[arr <= vmax]
    if visible.size == 0:
        visible = arr
    xmin_data = float(np.min(visible))
    xmin = 0.0 if start_at_zero else xmin_data
    xmax = vmax * 1.02 if start_at_zero else vmax
    ax.set_xlim(xmin, xmax)

    if use_data_ticks:
        ticks = visible[visible >= xmin]
        if (
            max_data_ticks is not None
            and max_data_ticks > 0
            and ticks.size > max_data_ticks
        ):
            idx = np.linspace(0, ticks.size - 1, num=max_data_ticks).round().astype(int)
            idx = np.unique(idx)
            ticks = ticks[idx]
        if ticks.size:
            ax.set_xticks(ticks)
            return

    step_candidates = [50, 100, 200, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]
    step = step_candidates[-1]
    for candidate in step_candidates:
        if (vmax - xmin) / candidate <= 8:
            step = candidate
            break
    ticks = np.arange(xmin, vmax + step, step)
    ax.set_xticks(ticks)


def _build_confusion_matrix(counts: Sequence[Sequence[int]]) -> List[List[float]]:
    matrix = [[0.0, 0.0], [0.0, 0.0]]
    for actual in (0, 1):
        col_total = float(counts[0][actual] + counts[1][actual])
        if col_total > 0.0:
            matrix[0][actual] = counts[0][actual] / col_total
            matrix[1][actual] = counts[1][actual] / col_total
        else:
            matrix[0][actual] = 1.0 if actual == 0 else 0.0
            matrix[1][actual] = 0.0 if actual == 0 else 1.0
    return matrix


def _invert_2x2(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    a, b = float(matrix[0][0]), float(matrix[0][1])
    c, d = float(matrix[1][0]), float(matrix[1][1])
    det = (a * d) - (b * c)
    if abs(det) < 1e-9:
        return [[1.0, 0.0], [0.0, 1.0]]
    inv_det = 1.0 / det
    return [[d * inv_det, -b * inv_det], [-c * inv_det, a * inv_det]]


def _matvec(matrix: Sequence[Sequence[float]], vec: Sequence[float]) -> List[float]:
    return [
        matrix[0][0] * vec[0] + matrix[0][1] * vec[1],
        matrix[1][0] * vec[0] + matrix[1][1] * vec[1],
    ]


def _apply_inverse_with_clip(
    matrix_inv: Sequence[Sequence[float]],
    observed: Sequence[int],
) -> List[float]:
    total = float(observed[0] + observed[1])
    if total <= 0.0:
        return [0.0, 0.0]
    est = _matvec(matrix_inv, [float(observed[0]), float(observed[1])])
    clipped = [max(0.0, v) for v in est]
    s = sum(clipped)
    if s <= 0.0:
        return [float(observed[0]), float(observed[1])]
    scale = total / s
    return [v * scale for v in clipped]


def _classify_from_bounds(
    *,
    has_samples: bool,
    flagged: bool,
    LDelta: float,
    eps_delta: float,
) -> List[str]:
    if not has_samples:
        return ["Inconclusive"]
    if not flagged:
        return ["Clean"]
    labels = ["Flagged (noisy)"]
    if LDelta > eps_delta:
        labels.append("Loss-biased (1$\to$0)")
    return labels


def _noise_label_set(labels: Sequence[str]) -> Set[str]:
    return _STRUCTURAL_NOISY.intersection(labels)


def counters_to_stats(
    C: Sequence[NodeCounters],
    z: float,
    eps: float,
    flag_rule: FlagRule,
    eps0: Optional[float] = None,
    eps1: Optional[float] = None,
    eps_delta: Optional[float] = None,
) -> List[NodeStats]:
    _ = flag_rule
    if eps0 is None:
        eps0 = 0.5 * eps
    if eps1 is None:
        eps1 = eps
    if eps_delta is None:
        eps_delta = 0.005

    stats: List[NodeStats] = []
    for cnt in C:
        n = cnt.n_det
        n1 = cnt.n1
        n0 = cnt.n0

        e_raw = (cnt.m_det / n) if n > 0 else 0.0
        L_raw, U_raw = wilson_bounds(e_raw, n, z) if n > 0 else (0.0, 1.0)

        raw_e1 = (cnt.m1 / n1) if n1 > 0 else 0.0
        raw_L1, raw_U1 = wilson_bounds(raw_e1, n1, z) if n1 > 0 else (0.0, 1.0)

        raw_e0 = (cnt.m0 / n0) if n0 > 0 else 0.0
        raw_L0, raw_U0 = wilson_bounds(raw_e0, n0, z) if n0 > 0 else (0.0, 1.0)

        raw_Ltot = max(raw_L1, raw_L0)
        raw_LDelta = raw_L1 - raw_U0
        raw_classification = _classify_from_bounds(
            has_samples=n > 0,
            flagged=raw_Ltot > eps,
            LDelta=raw_LDelta,
            eps_delta=float(eps_delta),
        )
        raw_flagged = raw_Ltot > eps

        cal_matrix = _build_confusion_matrix(cnt.calibration_counts)
        cal_inv = _invert_2x2(cal_matrix)

        obs1 = [cnt.reported_counts[1][0], cnt.reported_counts[1][1]]
        obs0 = [cnt.reported_counts[0][0], cnt.reported_counts[0][1]]
        corr_counts1 = _apply_inverse_with_clip(cal_inv, obs1)
        corr_counts0 = _apply_inverse_with_clip(cal_inv, obs0)

        total1 = float(sum(obs1))
        total0 = float(sum(obs0))
        corrected_e1 = (corr_counts1[0] / total1) if total1 > 0 else 0.0
        corrected_e0 = (corr_counts0[1] / total0) if total0 > 0 else 0.0

        corrected_errors = (corrected_e1 * total1) + (corrected_e0 * total0)
        e_corrected = (corrected_errors / n) if n > 0 else 0.0

        L_corr, U_corr = wilson_bounds(e_corrected, n, z) if n > 0 else (0.0, 1.0)
        L1_corr, U1_corr = wilson_bounds(corrected_e1, n1, z) if n1 > 0 else (0.0, 1.0)
        L0_corr, U0_corr = wilson_bounds(corrected_e0, n0, z) if n0 > 0 else (0.0, 1.0)
        Ltot_corr = max(L1_corr, L0_corr)
        LDelta_corr = L1_corr - U0_corr

        p_hat_corr = 0.5 * (corrected_e1 + corrected_e0)

        classification = _classify_from_bounds(
            has_samples=n > 0,
            flagged=Ltot_corr > eps,
            LDelta=LDelta_corr,
            eps_delta=float(eps_delta),
        )
        flagged = Ltot_corr > eps

        meter_dominated = _noise_label_set(raw_classification) != _noise_label_set(
            classification
        )

        stats.append(
            NodeStats(
                e=e_corrected,
                p_hat=p_hat_corr,
                U=U_corr,
                L=L_corr,
                U1=U1_corr,
                L1=L1_corr,
                U0=U0_corr,
                L0=L0_corr,
                LDelta=LDelta_corr,
                eta_hat=1.0 - corrected_e1,
                pdark_hat=corrected_e0,
                flagged=flagged,
                classification=list(classification),
                raw_flagged=raw_flagged,
                raw_classification=list(raw_classification),
                raw_e1=raw_e1,
                raw_e0=raw_e0,
                corrected_e1=corrected_e1,
                corrected_e0=corrected_e0,
                meter_dominated=meter_dominated,
                calibration_matrix=[row[:] for row in cal_matrix],
                calibration_inverse=[row[:] for row in cal_inv],
            )
        )
    return stats


_SIGMA_CACHE: Dict[int, Tuple[Set[Tuple[int, ...]], Dict[Tuple[int, ...], float]]] = {}
_SIGMA_INT_CACHE: Dict[int, Tuple[Set[int], Dict[int, float]]] = {}


def _build_sigmaid_and_pid(
    num_lieutenants: int,
) -> Tuple[Set[Tuple[int, ...]], Dict[Tuple[int, ...], float]]:
    if num_lieutenants in _SIGMA_CACHE:
        return _SIGMA_CACHE[num_lieutenants]

    sigmaid: Set[Tuple[int, ...]] = set()
    pid: Dict[Tuple[int, ...], float] = {}

    outcome_00 = (0, 0) + tuple(1 for _ in range(num_lieutenants))
    outcome_11 = (1, 1) + tuple(0 for _ in range(num_lieutenants))
    sigmaid.add(outcome_00)
    sigmaid.add(outcome_11)
    pid[outcome_00] = 1.0 / 3.0
    pid[outcome_11] = 1.0 / 3.0

    if num_lieutenants > 0:
        prob_split = (1.0 / 6.0) / float(num_lieutenants)
        for lt in range(num_lieutenants):
            base_01 = [0, 1] + [0] * num_lieutenants
            base_01[2 + lt] = 1
            t01 = tuple(base_01)
            sigmaid.add(t01)
            pid[t01] = prob_split

            base_10 = [1, 0] + [1] * num_lieutenants
            base_10[2 + lt] = 0
            t10 = tuple(base_10)
            sigmaid.add(t10)
            pid[t10] = prob_split

    _SIGMA_CACHE[num_lieutenants] = (sigmaid, pid)
    return sigmaid, pid


def _sigma_as_ints(
    num_lieutenants: int,
) -> Tuple[Set[int], Dict[int, float]]:
    if num_lieutenants in _SIGMA_INT_CACHE:
        return _SIGMA_INT_CACHE[num_lieutenants]

    sigmaid, pid = _build_sigmaid_and_pid(num_lieutenants)

    def pack(outcome: Tuple[int, ...]) -> int:
        c0, c1, *bits = outcome
        rep = 0
        for idx, bit in enumerate(bits):
            if bit:
                rep |= 1 << idx
        return (c0 << (num_lieutenants + 1)) | (c1 << num_lieutenants) | rep

    sigma_ints = {pack(outcome) for outcome in sigmaid}
    pid_int = {pack(outcome): prob for outcome, prob in pid.items()}
    _SIGMA_INT_CACHE[num_lieutenants] = (sigma_ints, pid_int)
    return sigma_ints, pid_int


def compute_leakage_and_fidelity(
    rounds: Sequence[Tuple[Tuple[int, int], List[int], List[int]]],
    verify_indices: Sequence[int],
    num_lieutenants: int,
    z: float,
) -> Dict[str, float]:
    total = len(verify_indices)
    if total <= 0:
        return {
            "q_hat": 0.0,
            "q_L": 0.0,
            "q_U": 1.0,
            "samples": 0.0,
            "allowed_samples": 0.0,
            "F_c": 0.0,
        }

    sigmaid, pid = _build_sigmaid_and_pid(num_lieutenants)

    leakage = 0
    counts: Dict[Tuple[int, ...], int] = {}
    for idx in verify_indices:
        (c0, c1), _phys, reported = rounds[idx]
        outcome = (c0, c1, *reported[:num_lieutenants])
        counts[outcome] = counts.get(outcome, 0) + 1
        if outcome not in sigmaid:
            leakage += 1

    phat = leakage / float(total)
    L_q, U_q = wilson_bounds(phat, total, z)

    allowed_total = total - leakage
    if allowed_total > 0:
        fc_sum = 0.0
        inv_allowed = 1.0 / float(allowed_total)
        for outcome, count in counts.items():
            p_id = pid.get(outcome, 0.0)
            if p_id <= 0.0:
                continue
            p_tilde = count * inv_allowed
            if p_tilde > 0.0:
                fc_sum += math.sqrt(p_tilde * p_id)
        F_c = fc_sum * fc_sum
    else:
        F_c = 0.0

    return {
        "q_hat": phat,
        "q_L": L_q,
        "q_U": U_q,
        "samples": float(total),
        "allowed_samples": float(max(0, allowed_total)),
        "F_c": F_c,
    }


def compute_leakage_and_fidelity_arrays(
    c0: np.ndarray,
    c1: np.ndarray,
    reported: np.ndarray,
    verify_indices: Sequence[int],
    num_lieutenants: int,
    z: float,
) -> Dict[str, float]:
    idx = np.asarray(list(verify_indices), dtype=np.int64)
    total = int(idx.size)
    if total <= 0:
        return {
            "q_hat": 0.0,
            "q_L": 0.0,
            "q_U": 1.0,
            "samples": 0.0,
            "allowed_samples": 0.0,
            "F_c": 0.0,
        }

    c0v = c0[idx].astype(np.uint64, copy=False)
    c1v = c1[idx].astype(np.uint64, copy=False)
    reps = reported[idx, :num_lieutenants].astype(np.uint64, copy=False)

    if num_lieutenants:
        powers = np.uint64(1) << np.arange(num_lieutenants, dtype=np.uint64)
        rep_bits = (reps * powers).sum(axis=1, dtype=np.uint64)
    else:
        rep_bits = np.zeros(total, dtype=np.uint64)

    vals = (c0v << (num_lieutenants + 1)) | (c1v << num_lieutenants) | rep_bits
    uniq, cnts = np.unique(vals, return_counts=True)

    sigma_ints, pid_int = _sigma_as_ints(num_lieutenants)
    legal_mask = np.array([int(v) in sigma_ints for v in uniq], dtype=bool)
    leakage = int(cnts[~legal_mask].sum()) if np.any(~legal_mask) else 0

    phat = leakage / float(total)
    L_q, U_q = wilson_bounds(phat, total, z)

    allowed_total = total - leakage
    if allowed_total > 0:
        inv_allowed = 1.0 / float(allowed_total)
        fc_sum = 0.0
        for v, c in zip(uniq, cnts):
            p_id = pid_int.get(int(v), 0.0)
            if p_id <= 0.0:
                continue
            p_tilde = c * inv_allowed
            if p_tilde > 0.0:
                fc_sum += math.sqrt(p_tilde * p_id)
        F_c = fc_sum * fc_sum
    else:
        F_c = 0.0

    return {
        "q_hat": phat,
        "q_L": L_q,
        "q_U": U_q,
        "samples": float(total),
        "allowed_samples": float(max(0, allowed_total)),
        "F_c": F_c,
    }


def _infer_expected_commander_bits(
    lieutenant_bits: Sequence[int],
) -> Optional[Tuple[int, int]]:
    """Strict commander inference: only unanimous lieutenant strings are usable.

    This avoids ambiguous branch flips from single-lieutenant errors.
    """
    L = len(lieutenant_bits)
    ones = int(sum(int(b) for b in lieutenant_bits))
    if ones == L:
        return (0, 0)
    if ones == 0:
        return (1, 1)
    return None


def compute_commander_diagnosis(
    rounds: Sequence[Tuple[Tuple[int, int], List[int], List[int]]],
    verify_indices: Sequence[int],
    num_lieutenants: int,
    z: float,
    eps_flag: float,
    eps_delta: float,
) -> Dict[str, object]:
    total = len(verify_indices)
    if total <= 0:
        return {
            "samples": 0,
            "s_clean": 0,
            "n1": 0,
            "m1": 0,
            "n0": 0,
            "m0": 0,
            "e1": 0.0,
            "e0": 0.0,
            "L1": 0.0,
            "U1": 1.0,
            "L0": 0.0,
            "U0": 1.0,
            "Ltot": 0.0,
            "LDelta": 0.0,
            "flagged": False,
            "loss_biased": False,
            "classification": ["Inconclusive"],
        }

    s_clean = 0
    n1 = 0
    m1 = 0
    n0 = 0
    m0 = 0
    expected_counts: Dict[str, int] = {"00": 0, "11": 0, "01": 0, "10": 0}

    for idx in verify_indices:
        (c0, c1), _phys, reported = rounds[idx]
        expected = _infer_expected_commander_bits(reported[:num_lieutenants])
        if expected is None:
            continue

        s_clean += 1
        exp_key = f"{expected[0]}{expected[1]}"
        expected_counts[exp_key] = expected_counts.get(exp_key, 0) + 1

        for exp_bit, obs_bit in ((expected[0], c0), (expected[1], c1)):
            if exp_bit == 1:
                n1 += 1
                if obs_bit == 0:
                    m1 += 1
            else:
                n0 += 1
                if obs_bit == 1:
                    m0 += 1

    e1 = (m1 / n1) if n1 > 0 else 0.0
    e0 = (m0 / n0) if n0 > 0 else 0.0
    L1, U1 = wilson_bounds(e1, n1, z) if n1 > 0 else (0.0, 1.0)
    L0, U0 = wilson_bounds(e0, n0, z) if n0 > 0 else (0.0, 1.0)
    Ltot = max(L1, L0)
    LDelta = L1 - U0

    flagged = (s_clean > 0) and (Ltot > eps_flag)
    classification = _classify_from_bounds(
        has_samples=s_clean > 0,
        flagged=flagged,
        LDelta=LDelta,
        eps_delta=eps_delta,
    )

    return {
        "samples": int(total),
        "s_clean": int(s_clean),
        "n1": int(n1),
        "m1": int(m1),
        "n0": int(n0),
        "m0": int(m0),
        "e1": float(e1),
        "e0": float(e0),
        "L1": float(L1),
        "U1": float(U1),
        "L0": float(L0),
        "U0": float(U0),
        "Ltot": float(Ltot),
        "LDelta": float(LDelta),
        "flagged": bool(flagged),
        "loss_biased": bool(flagged and (LDelta > eps_delta)),
        "classification": list(classification),
        "expected_counts": expected_counts,
    }


def compute_commander_diagnosis_arrays(
    c0: np.ndarray,
    c1: np.ndarray,
    reported: np.ndarray,
    verify_indices: Sequence[int],
    num_lieutenants: int,
    z: float,
    eps_flag: float,
    eps_delta: float,
) -> Dict[str, object]:
    idx = np.asarray(list(verify_indices), dtype=np.int64)
    total = int(idx.size)
    if total <= 0:
        return {
            "samples": 0,
            "s_clean": 0,
            "n1": 0,
            "m1": 0,
            "n0": 0,
            "m0": 0,
            "e1": 0.0,
            "e0": 0.0,
            "L1": 0.0,
            "U1": 1.0,
            "L0": 0.0,
            "U0": 1.0,
            "Ltot": 0.0,
            "LDelta": 0.0,
            "flagged": False,
            "loss_biased": False,
            "classification": ["Inconclusive"],
        }

    c0_v = c0[idx]
    c1_v = c1[idx]
    rep_v = reported[idx, :num_lieutenants]
    ones = (
        rep_v.sum(axis=1, dtype=np.int64)
        if num_lieutenants > 0
        else np.zeros(total, dtype=np.int64)
    )
    mask_00 = ones == num_lieutenants
    mask_11 = ones == 0
    clean_mask = mask_00 | mask_11

    s_clean = int(np.count_nonzero(clean_mask))
    expected_counts: Dict[str, int] = {
        "00": int(np.count_nonzero(mask_00)),
        "11": int(np.count_nonzero(mask_11)),
        "01": 0,
        "10": 0,
    }

    n1 = int(2 * expected_counts["11"])
    n0 = int(2 * expected_counts["00"])
    m1 = (
        int(np.count_nonzero(c0_v[mask_11] == 0))
        + int(np.count_nonzero(c1_v[mask_11] == 0))
        if expected_counts["11"] > 0
        else 0
    )
    m0 = (
        int(np.count_nonzero(c0_v[mask_00] == 1))
        + int(np.count_nonzero(c1_v[mask_00] == 1))
        if expected_counts["00"] > 0
        else 0
    )

    e1 = (m1 / n1) if n1 > 0 else 0.0
    e0 = (m0 / n0) if n0 > 0 else 0.0
    L1, U1 = wilson_bounds(e1, n1, z) if n1 > 0 else (0.0, 1.0)
    L0, U0 = wilson_bounds(e0, n0, z) if n0 > 0 else (0.0, 1.0)
    Ltot = max(L1, L0)
    LDelta = L1 - U0

    flagged = (s_clean > 0) and (Ltot > eps_flag)
    classification = _classify_from_bounds(
        has_samples=s_clean > 0,
        flagged=flagged,
        LDelta=LDelta,
        eps_delta=eps_delta,
    )

    return {
        "samples": int(total),
        "s_clean": int(s_clean),
        "n1": int(n1),
        "m1": int(m1),
        "n0": int(n0),
        "m0": int(m0),
        "e1": float(e1),
        "e0": float(e0),
        "L1": float(L1),
        "U1": float(U1),
        "L0": float(L0),
        "U0": float(U0),
        "Ltot": float(Ltot),
        "LDelta": float(LDelta),
        "flagged": bool(flagged),
        "loss_biased": bool(flagged and (LDelta > eps_delta)),
        "classification": list(classification),
        "expected_counts": expected_counts,
    }


def run_simulation(
    cfg: SimulationConfig,
    p_phys_list: Sequence[float],
    q_class_list: Sequence[float],
) -> Dict[str, object]:
    """Run the verification-and-estimation process and return summary results."""
    if cfg.num_generals < 3:
        raise ValueError("num_generals must be >= 3 (Commander + >=2 Lieutenants)")
    num_lieutenants = cfg.num_generals - 1
    if len(p_phys_list) != num_lieutenants or len(q_class_list) != num_lieutenants:
        raise ValueError(
            "length of p_phys_list and q_class_list must be num_generals-1"
        )

    num_verify = max(1, int(cfg.verify_fraction * cfg.shots))
    qkd_key = cfg.qkd_key
    batch_id = cfg.batch_id
    vseed = derive_seed(qkd_key, batch_id, salt="verify")
    rng_v = random.Random(vseed)
    verify_indices = rng_v.sample(range(cfg.shots), num_verify)
    z = inv_std_normal_cdf(1.0 - cfg.delta)
    if cfg.engine != "perceval":
        raise RuntimeError(
            "Fast engine disabled: set engine='perceval' to run noise_est."
        )

    nm = cfg.pcvl_noise_model
    rounds = sample_rounds_perceval(
        n=cfg.num_generals,
        shots=cfg.shots,
        p_phys_list=list(p_phys_list),
        q_class_list=list(q_class_list),
        seed=cfg.seed,
        noise_model=nm,
        dark_scale=cfg.pcvl_dark_scale,
        dcr_hz_list=cfg.dcr_hz_list,
        gate_ns_list=cfg.gate_ns_list,
        dcr_hz_cmd=cfg.dcr_hz_cmd,
        gate_ns_cmd=cfg.gate_ns_cmd,
        p_phys_cmd=cfg.p_phys_cmd,
    )
    c0_arr, c1_arr, phys_arr, rep_arr = _rounds_to_arrays(rounds, num_lieutenants)
    commander_counts = compute_commander_counts_arrays(c0_arr, c1_arr)

    key_material = _decode_qkd_key_material(qkd_key)
    for idx in verify_indices:
        c01, _phys, rep = rounds[idx]
        commit_open_outcome(qkd_key, c01, rep, key_material=key_material)

    C = accumulate_counters_fast(c0_arr, c1_arr, phys_arr, rep_arr, verify_indices)

    leak_metrics = compute_leakage_and_fidelity_arrays(
        c0_arr, c1_arr, rep_arr, verify_indices, num_lieutenants, z
    )
    eps_delta = float(cfg.epsilon_delta if cfg.epsilon_delta is not None else 0.005)
    commander_diag = compute_commander_diagnosis_arrays(
        c0_arr,
        c1_arr,
        rep_arr,
        verify_indices,
        num_lieutenants,
        z,
        float(cfg.epsilon_threshold),
        eps_delta,
    )
    commander_total = len(rounds)

    leak_ok = leak_metrics["q_U"] <= float(cfg.tau_leak)
    fid_ok = leak_ok and (leak_metrics["F_c"] >= float(cfg.tau_fid))
    batch_accept = leak_ok and fid_ok
    noisy_leakage = not leak_ok
    batch_gate = not batch_accept

    stats = counters_to_stats(
        C,
        z,
        cfg.epsilon_threshold,
        cfg.flag_rule,
        cfg.epsilon0,
        cfg.epsilon1,
        cfg.epsilon_delta,
    )

    summary = build_node_summary(C, stats)
    result = summary.as_dict()
    meter_nodes = [i for i, s in enumerate(stats) if s.meter_dominated]
    mitigation_trigger = batch_gate or bool(meter_nodes)
    result.update(
        {
            "delta": float(cfg.delta),
            "q_leak_hat": leak_metrics["q_hat"],
            "q_leak_L": leak_metrics["q_L"],
            "q_leak_U": leak_metrics["q_U"],
            "q_leak_samples": leak_metrics["samples"],
            "q_allowed_samples": leak_metrics["allowed_samples"],
            "F_c": leak_metrics["F_c"],
            "noisy_leakage": noisy_leakage,
            "leak_ok": leak_ok,
            "fid_ok": fid_ok,
            "batch_accept": batch_accept,
            "batch_gate": batch_gate,
            "meter_dominated_nodes": meter_nodes,
            "mitigation_trigger": mitigation_trigger,
            "tau_leak": float(cfg.tau_leak),
            "tau_fid": float(cfg.tau_fid),
            "commander_outcomes": commander_counts,
            "commander_total": commander_total,
            "commander_flagged": bool(commander_diag["flagged"]),
            "commander_diagnosis": commander_diag,
        }
    )
    return result


def _estimate_one_window(
    rng: random.Random,
    shots: int,
    verify_fraction: float,
    active_indices: Sequence[int],
    p_phys_list: Sequence[float],
    q_class_list: Sequence[float],
    *,
    epsilon: float,
    delta_eff: float,
    flag_rule: FlagRule,
    noise_model: "pcvl.NoiseModel | None",
    dark_scale: float,
    dcr_hz_list: Sequence[float] | None,
    gate_ns_list: Sequence[float] | None,
    dcr_hz_cmd: tuple[float, float] | None,
    gate_ns_cmd: tuple[float, float] | None,
    p_phys_cmd: tuple[float, float],
    qkd_key: str,
    epsilon0: float | None,
    epsilon1: float | None,
    epsilon_delta: float | None,
    tau_leak: float,
    tau_fid: float,
    engine: Literal["fast", "perceval"],
) -> Dict[str, object]:
    """Estimate per-node stats for a single window over active_indices only.

    Returns a dict with keys matching run_simulation's return for convenience,
    but values are length = total lieutenants, with zeros for inactive nodes.
    Also returns 'active_indices' and 'all_indices' for context.
    """
    active = list(active_indices)
    num_lieutenants_total = len(p_phys_list)

    p_act = [p_phys_list[i] for i in active]
    q_act = [q_class_list[i] for i in active]

    if dcr_hz_list is not None:
        dcr_act = [
            float(dcr_hz_list[i]) if i < len(dcr_hz_list) else 0.0 for i in active
        ]
    else:
        dcr_act = None
    if gate_ns_list is not None:
        gate_act = [
            float(gate_ns_list[i]) if i < len(gate_ns_list) else 0.0 for i in active
        ]
    else:
        gate_act = None

    num_verify = max(1, int(verify_fraction * shots))
    batch_tag = f"win-{len(active)}-{shots}"
    vseed = derive_seed(qkd_key, batch_tag, salt="verify")
    rng_v = random.Random(vseed)
    verify_indices = rng_v.sample(range(shots), num_verify)
    z = inv_std_normal_cdf(1.0 - delta_eff)
    shot_seed = rng.randrange(2**63)

    if engine != "perceval":
        raise RuntimeError(
            "Fast engine disabled: window estimation requires engine='perceval'."
        )

    rounds: List[Tuple[Tuple[int, int], List[int], List[int]]] = sample_rounds_perceval(
        n=len(active) + 1,
        shots=shots,
        p_phys_list=list(p_act),
        q_class_list=list(q_act),
        seed=shot_seed,
        noise_model=noise_model,
        dark_scale=dark_scale,
        dcr_hz_list=dcr_act,
        gate_ns_list=gate_act,
        dcr_hz_cmd=dcr_hz_cmd,
        gate_ns_cmd=gate_ns_cmd,
        p_phys_cmd=p_phys_cmd,
    )
    c0_arr, c1_arr, phys_arr, rep_arr = _rounds_to_arrays(rounds, len(active))

    key_material = _decode_qkd_key_material(qkd_key)
    for idx in verify_indices:
        c01, _phys, rep = rounds[idx]
        commit_open_outcome(qkd_key, c01, rep, key_material=key_material)

    C_act = accumulate_counters_fast(c0_arr, c1_arr, phys_arr, rep_arr, verify_indices)
    leak_metrics = compute_leakage_and_fidelity_arrays(
        c0_arr,
        c1_arr,
        rep_arr,
        verify_indices,
        len(active),
        z,
    )
    eps_delta = float(epsilon_delta if epsilon_delta is not None else 0.005)
    commander_diag = compute_commander_diagnosis_arrays(
        c0_arr,
        c1_arr,
        rep_arr,
        verify_indices,
        len(active),
        z,
        float(epsilon),
        eps_delta,
    )

    leak_ok = leak_metrics["q_U"] <= tau_leak
    fid_ok = leak_ok and (leak_metrics["F_c"] >= tau_fid)
    batch_accept = leak_ok and fid_ok
    noisy_leakage = not leak_ok
    batch_gate = not batch_accept

    stats_act = counters_to_stats(
        C_act,
        z,
        epsilon,
        flag_rule,
        epsilon0,
        epsilon1,
        epsilon_delta,
    )
    summary_active = build_node_summary(C_act, stats_act)
    summary_full = expand_summary_to_full(summary_active, num_lieutenants_total, active)
    result = summary_full.as_dict()
    result["active_indices"] = list(active)
    result["all_indices"] = list(range(num_lieutenants_total))
    meter_nodes_global = [
        idx
        for idx, flag in enumerate(result["meter_dominated"])
        if flag  # type: ignore[index]
    ]
    mitigation_trigger = batch_gate or bool(meter_nodes_global)
    result.update(
        {
            "delta": float(delta_eff),
            "q_leak_hat": leak_metrics["q_hat"],
            "q_leak_L": leak_metrics["q_L"],
            "q_leak_U": leak_metrics["q_U"],
            "q_leak_samples": leak_metrics["samples"],
            "q_allowed_samples": leak_metrics["allowed_samples"],
            "F_c": leak_metrics["F_c"],
            "noisy_leakage": noisy_leakage,
            "leak_ok": leak_ok,
            "fid_ok": fid_ok,
            "batch_accept": batch_accept,
            "batch_gate": batch_gate,
            "meter_dominated_nodes": meter_nodes_global,
            "mitigation_trigger": mitigation_trigger,
            "tau_leak": tau_leak,
            "tau_fid": tau_fid,
            "commander_flagged": bool(commander_diag["flagged"]),
            "commander_diagnosis": commander_diag,
        }
    )
    return result


def run_simulation_windows(
    cfg: SimulationConfig,
    p_phys_list: Sequence[float],
    q_class_list: Sequence[float],
) -> Dict[str, object]:
    """Run multi-window simulation with quarantine policy.

    - Nodes flagged in a window are excluded for the next cfg.quarantine_w windows.
    - Re-admission rule: two consecutive windows with U <= eps * cfg.readmit_eps_frac clears probation.
    Returns the result dict for the last window and a 'timeline' list summarizing all windows.
    """
    if cfg.num_generals < 3:
        raise ValueError("num_generals must be >= 3 (Commander + >=2 Lieutenants)")
    num_lieutenants = cfg.num_generals - 1
    if len(p_phys_list) != num_lieutenants or len(q_class_list) != num_lieutenants:
        raise ValueError(
            "length of p_phys_list and q_class_list must be num_generals-1"
        )

    rng = random.Random(cfg.seed)
    noise_model = cfg.pcvl_noise_model
    dark_scale = cfg.pcvl_dark_scale
    qkd_key = cfg.qkd_key
    eps0_cfg = cfg.epsilon0 if cfg.epsilon0 is not None else 0.5 * cfg.epsilon_threshold
    eps1_cfg = cfg.epsilon1 if cfg.epsilon1 is not None else cfg.epsilon_threshold
    eps_delta_cfg = cfg.epsilon_delta if cfg.epsilon_delta is not None else 0.005

    quarantine_remaining = [0] * num_lieutenants
    probation_streak = [0] * num_lieutenants
    probation_block = [False] * num_lieutenants

    timeline: List[Dict[str, object]] = []

    last_result: Dict[str, object] | None = None

    for w_idx in range(cfg.windows):
        # Nodes allowed to participate this window
        active_indices = [
            i
            for i in range(num_lieutenants)
            if quarantine_remaining[i] <= 0 and not probation_block[i]
        ]
        # Safety: protocol requires at least 2 lieutenants (n >= 3).
        if len(active_indices) < 2:
            active_indices = list(range(num_lieutenants))
        active_set = set(active_indices)

        delta_eff = cfg.delta

        win_res = _estimate_one_window(
            rng=rng,
            shots=cfg.shots,
            verify_fraction=cfg.verify_fraction,
            active_indices=active_indices,
            p_phys_list=p_phys_list,
            q_class_list=q_class_list,
            epsilon=cfg.epsilon_threshold,
            delta_eff=delta_eff,
            flag_rule=cfg.flag_rule,
            noise_model=noise_model,
            dark_scale=dark_scale,
            dcr_hz_list=cfg.dcr_hz_list,
            gate_ns_list=cfg.gate_ns_list,
            dcr_hz_cmd=cfg.dcr_hz_cmd,
            gate_ns_cmd=cfg.gate_ns_cmd,
            p_phys_cmd=cfg.p_phys_cmd,
            qkd_key=qkd_key,
            epsilon0=eps0_cfg,
            epsilon1=eps1_cfg,
            epsilon_delta=eps_delta_cfg,
            tau_leak=float(cfg.tau_leak),
            tau_fid=float(cfg.tau_fid),
            engine=cfg.engine,
        )

        flagged_global: List[int] = win_res["flagged"]  # type: ignore[index]

        for i in flagged_global:
            quarantine_remaining[i] = max(quarantine_remaining[i], cfg.quarantine_w)
            probation_streak[i] = 0
            probation_block[i] = True

        u1_bound: List[float] = win_res["u1_bound"]  # type: ignore[index]
        u0_bound: List[float] = win_res["u0_bound"]  # type: ignore[index]
        eps0 = float(eps0_cfg)
        eps1 = float(eps1_cfg)
        for i in range(num_lieutenants):
            if i in flagged_global:
                continue
            if i not in active_set:
                continue
            if quarantine_remaining[i] <= 0:
                good = (u1_bound[i] <= eps1 * cfg.readmit_eps_frac) and (
                    u0_bound[i] <= eps0
                )
                probation_streak[i] = (probation_streak[i] + 1) if good else 0
                if probation_streak[i] >= 2:
                    probation_block[i] = False
            else:
                probation_streak[i] = 0
                probation_block[i] = True

        quarantine_remaining = [max(0, q - 1) for q in quarantine_remaining]

        production_active = list(active_indices)

        l_bounds_win: List[float] = win_res.get("l_bound", [])  # type: ignore[index]
        if active_indices and l_bounds_win:
            baseline_l_max = float(max(l_bounds_win[i] for i in active_indices))
        else:
            baseline_l_max = float("nan")
        baseline_accept = bool(
            np.isfinite(baseline_l_max)
            and (baseline_l_max <= float(cfg.epsilon_threshold))
        )

        last_result = win_res
        timeline.append(
            {
                "window": w_idx + 1,
                "active": list(production_active),
                "flagged": list(flagged_global),
                "delta_eff": delta_eff,
                "probation_streak": list(probation_streak),
                "quarantine_remaining": list(quarantine_remaining),
                "probation_block": list(probation_block),
                "noisy_leakage": bool(win_res.get("noisy_leakage", False)),
                "batch_gate": bool(win_res.get("batch_gate", False)),
                "mitigation_trigger": bool(win_res.get("mitigation_trigger", False)),
                "meter_dominated_nodes": list(win_res.get("meter_dominated_nodes", [])),
                "F_c": float(win_res.get("F_c", 0.0)),
                "q_leak_hat": float(win_res.get("q_leak_hat", 0.0)),
                "q_leak_U": float(win_res.get("q_leak_U", 0.0)),
                "baseline_l_max": baseline_l_max,
                "baseline_accept": baseline_accept,
            }
        )

    assert last_result is not None
    last_result["timeline"] = timeline  # type: ignore[index]
    return last_result


def _ensure_dir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def percent_axis(ax: Axes, axis: str = "y", decimals: int = 0) -> None:
    fmt = FuncFormatter(lambda value, _pos: f"{value:.{decimals}f}%")
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def tidy_ticks(
    ax: Axes,
    x_major: Optional[int] = None,
    y_major: Optional[int] = None,
    add_minor: bool = True,
) -> None:
    if x_major:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_major, prune=None))
    if y_major:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_major, prune=None))
    if add_minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())


def save_all(fig: plt.Figure, path_no_ext: str, *, tight: bool = True) -> None:
    """Save both raster and vector exports, then close the figure."""

    path_root = Path(path_no_ext)
    if tight:
        fig.tight_layout()
    fig.savefig(path_root.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_root.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


LEGEND_BASE_KWARGS: Dict[str, object] = {
    "frameon": False,
    "borderpad": 0.8,
    "labelspacing": 0.6,
    "handlelength": 1.6,
    "handletextpad": 0.5,
}


def dedup_legend(
    ax: Axes, *, loc: str = "best", **kwargs: object
) -> Optional[plt.Legend]:
    handles = kwargs.pop("handles", None)
    labels = kwargs.pop("labels", None)
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    seen: Set[str] = set()
    uniq_handles: List[object] = []
    uniq_labels: List[str] = []
    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        uniq_handles.append(handle)
        uniq_labels.append(label)
    if not uniq_handles:
        return None
    params = {**LEGEND_BASE_KWARGS, **kwargs}
    params.setdefault("loc", loc)
    legend = ax.legend(uniq_handles, uniq_labels, **params)
    if legend is not None:
        legend._legend_box.align = "left"  # type: ignore[attr-defined]
    return legend


def plot_commander_detection(
    out_dir: str,
    sweep: Dict[str, object],
    *,
    highlight_p: Optional[float] = None,
    filename: str = "noise_est_commander_detection.png",
) -> Optional[str]:
    verify_sizes = np.array(sweep.get("verify_sizes", []), dtype=float)
    det_mean = np.array(sweep.get("det_mean", []), dtype=float)
    p_cmd_list = [float(v) for v in sweep.get("p_cmd_list", [])]

    if verify_sizes.size == 0 or det_mean.size == 0 or not p_cmd_list:
        return None

    _ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for idx, p_cmd in enumerate(p_cmd_list):
        color, marker = _series_style(idx)
        y_mean = np.asarray(det_mean[idx], dtype=float)
        smooth_mean = _smooth_series(y_mean, window=5)

        is_highlight = highlight_p is not None and math.isclose(
            float(p_cmd), float(highlight_p), abs_tol=1e-9
        )

        # Skip shaded uncertainty band to keep lines clean

        ax.plot(
            verify_sizes,
            smooth_mean,
            color=color,
            linewidth=1.6 if is_highlight else 1.4,
            alpha=0.95,
            marker=marker,
            markersize=6.0 if is_highlight else 5.0,
            markeredgewidth=0.8,
            markerfacecolor=color,
            markeredgecolor=color,
            markevery=max(1, verify_sizes.size // 10),
            label=rf"$p_C={p_cmd * 100:.0f}\%$",
        )

    ax.set_xlabel("Verification size $\\mathcal{S}$")
    ax.set_ylabel("Commander flag rate")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Per-Node Diagnosis: Commander Flag Rate")
    _configure_verification_axis(
        ax,
        verify_sizes,
        start_at_zero=False,
        use_data_ticks=True,
        max_data_ticks=6,
    )
    _style_axes(ax, grid="both")
    tidy_ticks(ax, y_major=6)
    dedup_legend(ax, loc="best")

    path = os.path.join(out_dir, filename)
    _save_figure(fig, path)
    return path


def _style_axes(
    ax: Axes,
    *,
    grid: Literal["x", "y", "both", "none"] = "y",
    tick_color: str = "#2b2b2b",
) -> None:
    if grid == "none":
        ax.grid(False)
    else:
        axis = "both" if grid == "both" else grid
        ax.grid(True, axis=axis, alpha=GRID_ALPHA, linewidth=0.7, linestyle="-")
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        length=4.5,
        width=0.8,
        color=tick_color,
        labelcolor=tick_color,
    )
    for spine_name in ("top", "right"):
        spine = ax.spines.get(spine_name)
        if spine is not None:
            spine.set_visible(False)
    for spine_name in ("bottom", "left"):
        spine = ax.spines.get(spine_name)
        if spine is not None:
            spine.set_linewidth(0.9)
            spine.set_color(tick_color)


def _save_figure(fig: plt.Figure, path: str) -> None:
    """Backward-compatible saver that dispatches to save_all."""

    root, _ = os.path.splitext(path)
    save_all(fig, root)


def _generate_seeds(base_seed: int, count: int) -> List[int]:
    rng = random.Random(base_seed)
    return [rng.randrange(2**63) for _ in range(count)]


def _run_independent_policy_plane_batches(
    cfg: SimulationConfig,
    p_phys_list: Sequence[float],
    q_class_list: Sequence[float],
    *,
    batch_count: int,
    seed: int,
    tag: str,
    batch_id_prefix: str,
) -> List[Dict[str, object]]:
    if batch_count <= 0:
        return []

    seeds = _generate_seeds(seed, batch_count)
    batches: List[Dict[str, object]] = []
    for idx, seed_r in enumerate(seeds, start=1):
        cfg_r = replace(
            cfg,
            seed=seed_r,
            windows=1,
            batch_id=f"{batch_id_prefix}-{idx:04d}",
        )
        res = run_simulation(cfg_r, p_phys_list, q_class_list)
        batches.append(
            {
                "window": idx,
                "policy_plane_tag": tag,
                "F_c": float(res.get("F_c", np.nan)),
                "q_leak_hat": float(res.get("q_leak_hat", np.nan)),
                "q_leak_U": float(res.get("q_leak_U", np.nan)),
                "q_leak_samples": float(res.get("q_leak_samples", np.nan)),
                "delta": float(res.get("delta", cfg_r.delta)),
            }
        )
    return batches


def _collect_policy_plane_data(
    single_result: Dict[str, object],
    timeline: Optional[Sequence[Dict[str, object]]],
    *,
    baseline_mode: str,
    epsilon_threshold: Optional[float],
) -> Dict[str, object]:
    tau_leak = float(single_result.get("tau_leak", single_result.get("tau_q", 0.05)))
    tau_fid = float(single_result.get("tau_fid", single_result.get("F_min", 0.89)))
    eps_thresh = (
        float(epsilon_threshold)
        if epsilon_threshold is not None
        else float(single_result.get("epsilon_threshold", float("nan")))
    )

    baseline_mode_norm = baseline_mode.lower().strip()

    if timeline and len(timeline) > 0:
        qhat = np.array(
            [float(t.get("q_leak_hat", np.nan)) for t in timeline], dtype=float
        )
        Uq = np.array([float(t.get("q_leak_U", np.nan)) for t in timeline], dtype=float)
        Fc = np.array([float(t.get("F_c", np.nan)) for t in timeline], dtype=float)
        Sarr = np.array(
            [float(t.get("q_leak_samples", np.nan)) for t in timeline], dtype=float
        )
        darr = np.array([float(t.get("delta", np.nan)) for t in timeline], dtype=float)
        baseline_l_max = np.array(
            [float(t.get("baseline_l_max", np.nan)) for t in timeline], dtype=float
        )
        tags = [str(t.get("policy_plane_tag", "")) for t in timeline]
    else:
        qhat = np.array([float(single_result.get("q_leak_hat", np.nan))], dtype=float)
        Uq = np.array([float(single_result.get("q_leak_U", np.nan))], dtype=float)
        Fc = np.array([float(single_result.get("F_c", np.nan))], dtype=float)
        Sarr = np.array(
            [float(single_result.get("q_leak_samples", np.nan))], dtype=float
        )
        darr = np.array([float(single_result.get("delta", np.nan))], dtype=float)
        baseline_l_max = np.array(
            [float(single_result.get("baseline_l_max", np.nan))], dtype=float
        )
        tags = [""]

    if baseline_mode_norm in {"support", "qhat", "supp"}:
        # Paper baseline protocol accepts by support compliance:
        # q_hat <= tau_leak on the sampled verification set.
        base_ok = qhat <= tau_leak
    elif baseline_mode_norm == "wilson":
        base_ok = np.array(
            [
                bool(
                    np.isfinite(l_val)
                    and np.isfinite(eps_thresh)
                    and l_val <= eps_thresh
                )
                for l_val in baseline_l_max
            ],
            dtype=bool,
        )
    else:
        # Fidelity-only baseline: accept when F_c >= tau_fid
        base_ok = Fc >= tau_fid

    ours_ok = (Fc >= tau_fid) & (Uq <= tau_leak)

    data: Dict[str, object] = {
        "Qhat": qhat,
        "Uq": Uq,
        "Fc": Fc,
        "samples": Sarr,
        "delta": darr,
        "base_ok": base_ok,
        "ours_ok": ours_ok,
        "tau_leak": tau_leak,
        "tau_fid": tau_fid,
        "baseline_mode": baseline_mode_norm,
        "indices": np.arange(1, len(Uq) + 1),
        "tags": tags,
    }
    return data


def _summarize_policy_plane(data: Dict[str, object]) -> Dict[str, object]:
    Uq = data["Uq"]
    Fc = data["Fc"]
    base_ok = data["base_ok"]
    ours_ok = data["ours_ok"]
    tau_leak = float(data["tau_leak"])
    tau_fid = float(data["tau_fid"])

    baseline_only = (~ours_ok) & base_ok
    both_accept = ours_ok & base_ok
    both_reject = (~ours_ok) & (~base_ok)
    ours_only = ours_ok & (~base_ok)

    counts = {
        "baseline_only": int(np.count_nonzero(baseline_only)),
        "both_accept": int(np.count_nonzero(both_accept)),
        "both_reject": int(np.count_nonzero(both_reject)),
        "ours_only": int(np.count_nonzero(ours_only)),
        "baseline_accept": int(np.count_nonzero(base_ok)),
        "ours_accept": int(np.count_nonzero(ours_ok)),
        "total": int(len(Uq)),
    }

    summary_rows: List[List[str]] = []
    total_float = float(max(1, counts["total"]))

    if counts["baseline_only"]:
        margins = Uq[baseline_only] - tau_leak
        median_margin = (
            float(np.nanmedian(margins)) if np.isfinite(margins).any() else float("nan")
        )
        summary_rows.append(
            [
                "Baseline-only",
                str(counts["baseline_only"]),
                f"{counts['baseline_only'] / total_float:.2f}",
                f"ΔU=+{median_margin:.3f}" if np.isfinite(median_margin) else "ΔU=--",
                f"$\\bar{{F}}={float(np.nanmean(Fc[baseline_only])):.3f}$",
            ]
        )

    if counts["both_accept"]:
        summary_rows.append(
            [
                "Both accept",
                str(counts["both_accept"]),
                f"{counts['both_accept'] / total_float:.2f}",
                f"$\\bar{{U}}={float(np.nanmean(Uq[both_accept])):.3f}$",
                f"$\\bar{{F}}={float(np.nanmean(Fc[both_accept])):.3f}$",
            ]
        )

    if counts["both_reject"]:
        deficits = tau_fid - Fc[both_reject]
        median_def = (
            float(np.nanmedian(deficits))
            if np.isfinite(deficits).any()
            else float("nan")
        )
        summary_rows.append(
            [
                "Both reject",
                str(counts["both_reject"]),
                f"{counts['both_reject'] / total_float:.2f}",
                f"ΔF=-{median_def:.3f}" if np.isfinite(median_def) else "ΔF=--",
                f"$\\bar{{U}}={float(np.nanmean(Uq[both_reject])):.3f}$",
            ]
        )

    if counts["ours_only"]:
        summary_rows.append(
            [
                "Ours-only",
                str(counts["ours_only"]),
                f"{counts['ours_only'] / total_float:.2f}",
                "-",
                "-",
            ]
        )

    if not summary_rows:
        summary_rows.append(["No data", "-", "-", "-", "-"])

    return {
        "rows": summary_rows,
        "counts": counts,
        "baseline_only_mask": baseline_only,
        "both_accept_mask": both_accept,
    }


def plot_policy_plane(
    out_dir: str,
    single_result: Dict[str, object],
    timeline: Optional[Sequence[Dict[str, object]]],
    *,
    filename: str = "noise_est_policy_plane.png",
    baseline_mode: str = "support",
    epsilon_threshold: Optional[float] = None,
    detailed_log: bool = False,
) -> Optional[str]:
    _ensure_dir(out_dir)

    data = _collect_policy_plane_data(
        single_result,
        timeline,
        baseline_mode=baseline_mode,
        epsilon_threshold=epsilon_threshold,
    )

    summary = _summarize_policy_plane(data)
    counts = summary["counts"]
    baseline_only_mask = summary["baseline_only_mask"]

    Uq = data["Uq"]
    Fc = data["Fc"]
    base_ok = data["base_ok"]
    ours_ok = data["ours_ok"]
    tau_leak = float(data["tau_leak"])
    tau_fid = float(data["tau_fid"])
    baseline_mode_norm = str(data["baseline_mode"])
    Sarr = np.asarray(data.get("samples", []), dtype=float)
    darr = np.asarray(data.get("delta", []), dtype=float)
    indices = data["indices"]
    finite_s = Sarr[np.isfinite(Sarr)]
    finite_d = darr[np.isfinite(darr)]
    S_eff = int(np.nanmedian(finite_s)) if finite_s.size else 0
    delta_eff = float(np.nanmedian(finite_d)) if finite_d.size else 0.05
    baseline_u_at_tau: float | None = None
    if baseline_mode_norm in {"support", "qhat", "supp"} and S_eff > 0:
        z_eff = inv_std_normal_cdf(1.0 - delta_eff)
        _, baseline_u_at_tau = wilson_bounds(tau_leak, S_eff, z_eff)

    diff_batches: List[int] = []
    detailed_rows: List[str] = []
    for idx, u_val, f_val, base_accept, ours_accept in zip(
        indices, Uq, Fc, base_ok, ours_ok
    ):
        u_str = f"{u_val:.6f}" if np.isfinite(u_val) else "nan"
        f_str = f"{f_val:.6f}" if np.isfinite(f_val) else "nan"
        base_label = "accept" if base_accept else "reject"
        ours_label = "accept" if ours_accept else "reject"
        if base_accept and ours_accept:
            category = "both-accept"
        elif base_accept and not ours_accept:
            category = "baseline-only"
        elif (not base_accept) and ours_accept:
            category = "ours-only"
        else:
            category = "both-reject"
        if base_accept != ours_accept:
            diff_batches.append(int(idx))
        if detailed_log:
            detailed_rows.append(
                f"{int(idx):>4} {u_str:>10} {f_str:>10} {base_label:>11} {ours_label:>8} {category:>16}"
            )

    if detailed_log and detailed_rows:
        if baseline_mode_norm in {"support", "qhat", "supp"}:
            mode_desc = "support-compliance"
        elif baseline_mode_norm == "fidelity":
            mode_desc = "fidelity-only"
        else:
            mode_desc = "Wilson"
        header = f"{'idx':>4} {'U_hat':>10} {'F_c':>10} {'baseline':>11} {'ours':>8} {'category':>16}"
        print(f"[policy-plane] Batch outcomes (baseline={mode_desc})")
        print(header)
        for row in detailed_rows:
            print(row)

    if diff_batches:
        diff_str = ", ".join(str(b) for b in diff_batches)
        print(f"[policy-plane] Baseline differs from ours on batch(es): {diff_str}")
    else:
        print("[policy-plane] Baseline matches ours on all batches.")

    print(
        f"[policy-plane] Counts — baseline accept: {counts['baseline_accept']}/{counts['total']}, "
        f"ours accept: {counts['ours_accept']}/{counts['total']}"
    )
    print(
        f"[policy-plane] Breakdown — baseline-only: {counts['baseline_only']}, "
        f"both-reject: {counts['both_reject']}, ours-only: {counts['ours_only']}, both-accept: {counts['both_accept']}"
    )

    if counts["baseline_only"] > 0:
        if baseline_mode_norm in {"support", "qhat", "supp"}:
            qhat_vals = np.array(data["Qhat"], dtype=float)
            margins = qhat_vals[baseline_only_mask] - tau_leak
            finite_margins = margins[np.isfinite(margins)]
            if finite_margins.size:
                print(
                    "[policy-plane] Support margin (q_hat - tau_leak) on baseline-only batches — "
                    f"median: {float(np.median(finite_margins)):.6f}, "
                    f"min: {float(np.min(finite_margins)):.6f}, max: {float(np.max(finite_margins)):.6f}"
                )
        else:
            margins = Uq[baseline_only_mask] - tau_leak
            finite_margins = margins[np.isfinite(margins)]
            if finite_margins.size:
                print(
                    "[policy-plane] Leakage overrun on baseline-only batches — "
                    f"median: {float(np.median(finite_margins)):.6f}, "
                    f"min: {float(np.min(finite_margins)):.6f}, max: {float(np.max(finite_margins)):.6f}"
                )

    finite_U = Uq[np.isfinite(Uq)]
    if finite_U.size:
        min_u = float(finite_U.min())
        max_u = float(finite_U.max())
    else:
        min_u = 0.0
        max_u = tau_leak

    x_ref_min = min(min_u, tau_leak)
    x_ref_max = max(max_u, tau_leak)
    if baseline_u_at_tau is not None and np.isfinite(baseline_u_at_tau):
        x_ref_min = min(x_ref_min, baseline_u_at_tau)
        x_ref_max = max(x_ref_max, baseline_u_at_tau)

    x_min = max(0.0, x_ref_min - 0.015)
    x_max = min(x_ref_max * 1.05, 1.0)
    y_tick_step = 0.02
    y_min = 0.86
    y_max = 1.0

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    from matplotlib.patches import Rectangle  # pyright: ignore[reportMissingImports]

    rect_height = max(y_max - tau_fid, 0.0)
    if rect_height > 0.0:
        # Ours acceptance region (U <= tau_leak and F_c >= F_min)
        ours_accept_width = max(min(tau_leak, x_max) - x_min, 0.0)
        if ours_accept_width > 0.0:
            accept_rect = Rectangle(
                (x_min, tau_fid),
                ours_accept_width,
                rect_height,
                facecolor=FIDELITY_COLOR,
                alpha=0.14,
                edgecolor=FIDELITY_COLOR,
                linewidth=0.4,
                hatch="///",
                zorder=0,
            )
            ax.add_patch(accept_rect)

        if baseline_u_at_tau is not None and np.isfinite(baseline_u_at_tau):
            mid_left = max(min(tau_leak, x_max), x_min)
            base_boundary = min(max(float(baseline_u_at_tau), x_min), x_max)
            base_boundary = max(base_boundary, mid_left)

            # Baseline-only support-compliance band.
            baseline_only_width = max(base_boundary - mid_left, 0.0)
            if baseline_only_width > 0.0:
                baseline_only_rect = Rectangle(
                    (mid_left, tau_fid),
                    baseline_only_width,
                    rect_height,
                    facecolor=PROBATION_COLOR,
                    alpha=0.10,
                    edgecolor=PROBATION_COLOR,
                    linewidth=0.4,
                    hatch="xx",
                    zorder=0,
                )
                ax.add_patch(baseline_only_rect)

            # Baseline reject side in U-space (often where "reject by both" appears).
            reject_width = max(x_max - base_boundary, 0.0)
            if reject_width > 0.0:
                both_reject_rect = Rectangle(
                    (base_boundary, tau_fid),
                    reject_width,
                    rect_height,
                    facecolor=ACCENT_COLOR,
                    alpha=0.09,
                    edgecolor=ACCENT_COLOR,
                    linewidth=0.4,
                    hatch="..",
                    zorder=0,
                )
                ax.add_patch(both_reject_rect)
        else:
            # Fallback when baseline boundary is not representable in U-space.
            baseline_rect = Rectangle(
                (x_min, tau_fid),
                max(x_max - x_min, 0.0),
                rect_height,
                facecolor=NEUTRAL_COLOR,
                alpha=0.10,
                edgecolor=NEUTRAL_COLOR,
                linewidth=0.0,
                hatch="//",
                zorder=0,
            )
            ax.add_patch(baseline_rect)

    ax.axvline(tau_leak, **THRESHOLD_LINE_STYLE)
    ax.axhline(tau_fid, **THRESHOLD_LINE_STYLE, zorder=6)
    if baseline_u_at_tau is not None and np.isfinite(baseline_u_at_tau):
        ax.axvline(
            baseline_u_at_tau,
            color=THRESHOLD_COLOR,
            linestyle=":",
            linewidth=1.2,
        )

    both_accept = ours_ok & base_ok
    baseline_only = (~ours_ok) & base_ok
    both_reject = (~ours_ok) & (~base_ok)

    if np.any(both_accept):
        ax.scatter(
            Uq[both_accept],
            Fc[both_accept],
            marker=MARKERS["both_accept"],
            s=54,
            facecolor=FIDELITY_COLOR,
            edgecolor="black",
            linewidths=0.6,
            label="Accept by both",
            zorder=3,
        )
    if np.any(baseline_only):
        ax.scatter(
            Uq[baseline_only],
            Fc[baseline_only],
            marker=MARKERS["baseline_only"],
            s=64,
            facecolor=PROBATION_COLOR,
            edgecolor="black",
            linewidths=0.6,
            label="Baseline-only accept",
            zorder=4,
        )
    if np.any(both_reject):
        ax.scatter(
            Uq[both_reject],
            Fc[both_reject],
            marker=MARKERS["both_reject"],
            s=70,
            color=ACCENT_COLOR,
            linewidths=1.4,
            label="Reject by both",
            zorder=3,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Leakage upper bound $U_{\alpha_{\mathrm{leak}}}(\hat q)$")
    ax.set_ylabel(r"Classical fidelity $F_c$")
    ax.set_title("Batch Health Check: Baseline vs. Our Proposed Policy")
    _style_axes(ax, grid="both")
    tidy_ticks(ax, x_major=6)
    y_ticks = np.arange(y_min, y_max + y_tick_step * 0.5, y_tick_step)
    ax.set_yticks(np.round(y_ticks, 2))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:.2f}"))

    # Threshold labels and disagreement callouts improve readability.
    ax.text(
        tau_leak,
        y_min,
        r"$\tau_{\mathrm{leak}}$",
        color=THRESHOLD_COLOR,
        fontsize=8,
        rotation=90,
        va="bottom",
        ha="right",
    )
    ax.text(
        x_min,
        tau_fid,
        r"$\tau_{\mathrm{fid}}$",
        color=THRESHOLD_COLOR,
        fontsize=8,
        va="bottom",
        ha="left",
    )
    if baseline_u_at_tau is not None and np.isfinite(baseline_u_at_tau):
        ax.text(
            baseline_u_at_tau,
            y_min,
            r"$U(\tau_{\mathrm{leak}})$",
            color=THRESHOLD_COLOR,
            fontsize=8,
            rotation=90,
            va="bottom",
            ha="left",
        )

    tags = data["tags"]
    for x_val, y_val, base_flag, ours_flag, tag in zip(Uq, Fc, base_ok, ours_ok, tags):
        if bool(base_flag) == bool(ours_flag):
            continue
        if not isinstance(tag, str) or not tag:
            continue
        tag_clean = tag.strip()
        if tag_clean.lower() in {"stress", "clean"}:
            continue
        ax.annotate(
            tag_clean,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    dedup_legend(ax, loc="lower left", fontsize=9)

    scatter_root = os.path.join(out_dir, os.path.splitext(filename)[0])
    save_all(fig, scatter_root)
    scatter_path = f"{scatter_root}.png"

    return scatter_path


def _parse_float_grid(spec: str) -> List[float]:
    """Parse comma list or start:stop:step spec into a float grid."""
    s = (spec or "").strip()
    if not s:
        return []
    if ":" in s:
        parts = [p.strip() for p in s.split(":")]
        if len(parts) != 3:
            raise SystemExit(f"Invalid range '{spec}'. Use start:stop:step")
        start, stop, step = (float(parts[0]), float(parts[1]), float(parts[2]))
        if step <= 0:
            raise SystemExit("Step must be > 0 in range spec")
        count = int(round((stop - start) / step)) + 1
        return [start + idx * step for idx in range(max(0, count))]
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_grid(spec: str) -> List[int]:
    """Parse comma lists and start:stop:step segments into an integer grid."""

    s = (spec or "").strip()
    if not s:
        return []

    values: List[int] = []
    seen: Set[int] = set()

    for part in s.split(","):
        part = part.strip()
        if not part:
            continue

        seq: List[int]
        if ":" in part:
            pieces = [p.strip() for p in part.split(":")]
            if len(pieces) != 3:
                raise SystemExit(f"Invalid range '{part}'. Use start:stop:step")
            start, stop, step = (int(pieces[0]), int(pieces[1]), int(pieces[2]))
            if step <= 0:
                raise SystemExit("Step must be > 0 in range spec")
            direction = 1 if stop >= start else -1
            step_eff = step * direction
            seq = list(range(start, stop + direction, step_eff))
            if seq:
                last = seq[-1]
                if direction > 0 and last < stop:
                    seq.append(stop)
                elif direction < 0 and last > stop:
                    seq.append(stop)
        else:
            seq = [int(part)]

        for val in seq:
            if val not in seen:
                values.append(val)
                seen.add(val)

    return values


def sweep_loss_bias_grid(
    base_cfg: SimulationConfig,
    n_list: Sequence[int],
    p_noisy_list: Sequence[float],
    *,
    p_clean: float = 0.02,
    noisy_count: int = 1,
    reps: int = 8,
    shots_override: Optional[int] = None,
    verify_frac_override: Optional[float] = None,
    verify_counts: Optional[Sequence[int]] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Run a grid of simulations and aggregate TP/FP rates for Loss-biased (1$\to$0)."""

    rng = random.Random(base_cfg.seed if seed is None else seed)
    Ns = list(n_list)
    Ps = list(p_noisy_list)
    det = np.full((len(Ns), len(Ps)), np.nan, dtype=float)
    det_min = np.full((len(Ns), len(Ps)), np.nan, dtype=float)
    det_max = np.full((len(Ns), len(Ps)), np.nan, dtype=float)

    if verify_counts and len(verify_counts) > 0:
        verify_targets = sorted(
            {min(VERIFY_TARGET_MAX, int(max(1, v))) for v in verify_counts}
        )
    else:
        shots_default = shots_override if shots_override is not None else base_cfg.shots
        verify_frac_default = (
            verify_frac_override
            if verify_frac_override is not None
            else base_cfg.verify_fraction
        )
        default_target = max(1, int(round(verify_frac_default * shots_default)))
        verify_targets = [min(VERIFY_TARGET_MAX, default_target)]

    shots_default = shots_override if shots_override is not None else base_cfg.shots
    verify_frac_default = (
        verify_frac_override
        if verify_frac_override is not None
        else base_cfg.verify_fraction
    )
    base_verify_count = max(1, int(round(verify_frac_default * shots_default)))
    base_verify_count = min(VERIFY_TARGET_MAX, base_verify_count)
    verify_targets = _downsample_verify_targets(
        verify_targets,
        max_points=12,
        include=[base_verify_count],
    )
    if verify_targets:
        det_target_idx = min(
            range(len(verify_targets)),
            key=lambda idx: abs(verify_targets[idx] - base_verify_count),
        )
    else:
        det_target_idx = 0

    det_by_S = {
        int(s_val): np.full((len(Ns), len(Ps)), np.nan, dtype=float)
        for s_val in verify_targets
    }
    det_by_S_min = {
        int(s_val): np.full((len(Ns), len(Ps)), np.nan, dtype=float)
        for s_val in verify_targets
    }
    det_by_S_max = {
        int(s_val): np.full((len(Ns), len(Ps)), np.nan, dtype=float)
        for s_val in verify_targets
    }
    fpr_by_S = {
        int(s_val): np.full((len(Ns), len(Ps)), np.nan, dtype=float)
        for s_val in verify_targets
    }
    fpr_by_S_min = {
        int(s_val): np.full((len(Ns), len(Ps)), np.nan, dtype=float)
        for s_val in verify_targets
    }
    fpr_by_S_max = {
        int(s_val): np.full((len(Ns), len(Ps)), np.nan, dtype=float)
        for s_val in verify_targets
    }

    total_cells = len(Ns) * len(Ps)
    progress = None
    cell_counter = 0
    if total_cells and tqdm is not None:
        desc = f"Loss-bias sweep |n|={len(Ns)} |p|={len(Ps)} reps={reps}"
        progress = tqdm(total=total_cells * len(verify_targets), desc=desc, leave=False)
    elif total_cells:
        print(
            f"[SWEEP] Running loss-bias grid |n|={len(Ns)} |p|={len(Ps)} reps={reps}",
            flush=True,
        )

    for i, n in enumerate(Ns):
        if n < 3:
            raise SystemExit(f"n must be >=3; got {n}")
        L = n - 1
        for j, p_noisy in enumerate(Ps):
            for s_idx, s_val in enumerate(verify_targets):
                shots_cell = (
                    shots_override if shots_override is not None else base_cfg.shots
                )
                shots_cell = max(shots_cell, s_val)
                verify_frac_cell = (
                    min(1.0, s_val / shots_cell) if shots_cell > 0 else 1.0
                )

                det_hits = 0
                det_total = 0
                fp_hits = 0
                fp_total = 0
                fp_bound_sum = 0.0
                fp_bound_count = 0
                det_rates: List[float] = []
                fpr_rates: List[float] = []

                for _ in range(reps):
                    seed_r = rng.randrange(2**63)
                    cfg_r = replace(
                        base_cfg,
                        num_generals=n,
                        seed=seed_r,
                        shots=shots_cell,
                        verify_fraction=verify_frac_cell,
                    )
                    p_phys = [float(p_clean)] * L
                    rloc = random.Random(seed_r ^ 0x9E3779B97F4A7C15)
                    noisy_total = min(max(1, noisy_count), L)
                    noisy_idx = set(rloc.sample(range(L), noisy_total))
                    for idx in noisy_idx:
                        p_phys[idx] = float(p_noisy)
                    q_class = [0.0] * L

                    det_hits_r = 0
                    det_total_r = 0
                    fp_hits_r = 0
                    fp_total_r = 0
                    fp_bound_sum_r = 0.0
                    fp_bound_count_r = 0

                    res = run_simulation(cfg_r, p_phys, q_class)
                    flagged_indices = {
                        int(val)
                        for val in res.get("flagged", [])
                        if isinstance(val, (int, float))
                    }
                    u_bounds = res.get("u_bound", [])

                    for idx in noisy_idx:
                        if idx in flagged_indices:
                            det_hits_r += 1
                        det_total_r += 1

                    for idx in range(L):
                        if idx in noisy_idx:
                            continue
                        if idx in flagged_indices:
                            fp_hits_r += 1
                        fp_total_r += 1
                        if idx < len(u_bounds):
                            bound_val = max(0.0, float(u_bounds[idx]))
                            fp_bound_sum_r += bound_val
                            fp_bound_count_r += 1

                    det_hits += det_hits_r
                    det_total += det_total_r
                    fp_hits += fp_hits_r
                    fp_total += fp_total_r
                    fp_bound_sum += fp_bound_sum_r
                    fp_bound_count += fp_bound_count_r

                    if det_total_r > 0:
                        det_rates.append(det_hits_r / det_total_r)
                    if fp_bound_count_r > 0:
                        fpr_rates.append(fp_bound_sum_r / fp_bound_count_r)
                    elif fp_total_r > 0:
                        fpr_rates.append(fp_hits_r / fp_total_r)

                det_rate = (det_hits / det_total) if det_total > 0 else np.nan
                det_mean_rep, det_rate_min, det_rate_max = _summary_triplet(det_rates)
                if np.isnan(det_rate) and np.isfinite(det_mean_rep):
                    det_rate = det_mean_rep
                if fp_bound_count > 0:
                    fpr_rate = fp_bound_sum / fp_bound_count
                elif fp_total > 0:
                    fpr_rate = fp_hits / fp_total
                else:
                    fpr_rate = np.nan
                fpr_mean_rep, fpr_rate_min, fpr_rate_max = _summary_triplet(fpr_rates)
                if np.isnan(fpr_rate) and np.isfinite(fpr_mean_rep):
                    fpr_rate = fpr_mean_rep

                det_rate_min = (
                    float(det_rate_min) if np.isfinite(det_rate_min) else np.nan
                )
                det_rate_max = (
                    float(det_rate_max) if np.isfinite(det_rate_max) else np.nan
                )
                fpr_rate_min = (
                    float(fpr_rate_min) if np.isfinite(fpr_rate_min) else np.nan
                )
                fpr_rate_max = (
                    float(fpr_rate_max) if np.isfinite(fpr_rate_max) else np.nan
                )

                if np.isfinite(det_rate_min):
                    det_rate_min = max(0.0, min(1.0, det_rate_min))
                if np.isfinite(det_rate_max):
                    det_rate_max = max(0.0, min(1.0, det_rate_max))
                if np.isfinite(fpr_rate_min):
                    fpr_rate_min = max(0.0, min(1.0, fpr_rate_min))
                if np.isfinite(fpr_rate_max):
                    fpr_rate_max = max(0.0, min(1.0, fpr_rate_max))

                if s_idx == det_target_idx:
                    det[i, j] = det_rate
                    det_min[i, j] = det_rate_min
                    det_max[i, j] = det_rate_max
                det_by_S[int(s_val)][i, j] = det_rate
                det_by_S_min[int(s_val)][i, j] = det_rate_min
                det_by_S_max[int(s_val)][i, j] = det_rate_max
                fpr_by_S[int(s_val)][i, j] = fpr_rate
                fpr_by_S_min[int(s_val)][i, j] = fpr_rate_min
                fpr_by_S_max[int(s_val)][i, j] = fpr_rate_max

                if progress is not None:
                    progress.update(1)
                elif total_cells:
                    cell_counter += 1
                    if (
                        cell_counter == total_cells * len(verify_targets)
                        or cell_counter
                        % max(1, (total_cells * len(verify_targets)) // 10)
                        == 0
                    ):
                        pct = (
                            cell_counter / (total_cells * len(verify_targets))
                        ) * 100.0
                        print(
                            f"[SWEEP] Progress: {cell_counter}/{total_cells * len(verify_targets)} cells ({pct:5.1f}%)",
                            flush=True,
                        )

    if progress is not None:
        progress.close()
    elif total_cells:
        print(
            f"[SWEEP] Completed {total_cells * len(verify_targets)} cells.",
            flush=True,
        )

    return {
        "n_grid": Ns,
        "L_grid": [n_val - 1 for n_val in Ns],
        "p_noisy_grid": Ps,
        "det_matrix": det.tolist(),
        "det_matrix_min": det_min.tolist(),
        "det_matrix_max": det_max.tolist(),
        "det_cube": [det_by_S[int(s)].tolist() for s in verify_targets],
        "det_cube_min": [det_by_S_min[int(s)].tolist() for s in verify_targets],
        "det_cube_max": [det_by_S_max[int(s)].tolist() for s in verify_targets],
        "fpr_cube": [fpr_by_S[int(s)].tolist() for s in verify_targets],
        "fpr_cube_min": [fpr_by_S_min[int(s)].tolist() for s in verify_targets],
        "fpr_cube_max": [fpr_by_S_max[int(s)].tolist() for s in verify_targets],
        "verify_counts": [int(s) for s in verify_targets],
        "reps": reps,
        "p_clean": float(p_clean),
        "noisy_count": int(noisy_count),
        "shots": int(shots_override if shots_override is not None else base_cfg.shots),
        "verify_frac": float(
            verify_frac_override
            if verify_frac_override is not None
            else base_cfg.verify_fraction
        ),
        "eps0": float(
            base_cfg.epsilon0
            if base_cfg.epsilon0 is not None
            else 0.5 * base_cfg.epsilon_threshold
        ),
        "eps1": float(
            base_cfg.epsilon1
            if base_cfg.epsilon1 is not None
            else base_cfg.epsilon_threshold
        ),
        "eps_delta": float(
            base_cfg.epsilon_delta if base_cfg.epsilon_delta is not None else 0.005
        ),
    }


def sweep_commander_detection_grid(
    base_cfg: SimulationConfig,
    p_cmd_list: Sequence[float],
    p_phys_list: Sequence[float],
    q_class_list: Sequence[float],
    *,
    verify_counts: Optional[Sequence[int]] = None,
    shots_override: Optional[int] = None,
    reps: int = 6,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    rng = random.Random(base_cfg.seed if seed is None else seed)
    p_vals = [float(v) for v in p_cmd_list]

    if verify_counts and len(verify_counts) > 0:
        verify_targets = _downsample_verify_targets(
            [min(VERIFY_TARGET_MAX, max(1, int(v))) for v in verify_counts],
            max_points=12,
        )
    else:
        shots_default = shots_override if shots_override is not None else base_cfg.shots
        default_target = max(1, int(round(base_cfg.verify_fraction * shots_default)))
        verify_targets = _downsample_verify_targets(
            [min(VERIFY_TARGET_MAX, default_target)],
            max_points=6,
        )

    if not verify_targets:
        verify_targets = [max(1, int(round(base_cfg.verify_fraction * base_cfg.shots)))]

    verify_targets = sorted(set(verify_targets))

    det_mean = np.full((len(p_vals), len(verify_targets)), np.nan, dtype=float)
    det_q1 = np.full_like(det_mean, np.nan)
    det_q3 = np.full_like(det_mean, np.nan)

    total_cells = len(p_vals) * len(verify_targets)
    progress = None
    if total_cells and tqdm is not None:
        desc = f"Commander detection sweep |p_C|={len(p_vals)} reps={reps}"
        progress = tqdm(total=total_cells, desc=desc, leave=False)
    elif total_cells:
        print(
            f"[SWEEP] Running commander detection grid |p_C|={len(p_vals)} reps={reps}",
            flush=True,
        )

    for i, p_cmd in enumerate(p_vals):
        for j, s_val in enumerate(verify_targets):
            shots_cell = (
                shots_override if shots_override is not None else base_cfg.shots
            )
            shots_cell = max(shots_cell, s_val)
            verify_frac = min(1.0, s_val / shots_cell) if shots_cell > 0 else 1.0

            det_rates: List[float] = []
            for _ in range(max(1, reps)):
                seed_r = rng.randrange(2**63)
                cfg_r = replace(
                    base_cfg,
                    seed=seed_r,
                    shots=shots_cell,
                    verify_fraction=verify_frac,
                    p_phys_cmd=(p_cmd, p_cmd),
                )
                res = run_simulation(cfg_r, p_phys_list, q_class_list)
                det_rates.append(1.0 if res.get("commander_flagged", False) else 0.0)

            if det_rates:
                arr = np.array(det_rates, dtype=float)
                det_mean[i, j] = float(np.mean(arr))
                det_q1[i, j] = float(np.quantile(arr, 0.25))
                det_q3[i, j] = float(np.quantile(arr, 0.75))

            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()
    elif total_cells:
        print(
            f"[SWEEP] Completed {total_cells} commander cells.",
            flush=True,
        )

    return {
        "verify_sizes": [int(v) for v in verify_targets],
        "p_cmd_list": [float(v) for v in p_vals],
        "det_mean": det_mean.tolist(),
        "det_q1": det_q1.tolist(),
        "det_q3": det_q3.tolist(),
        "reps": int(reps),
    }


def plot_loss_bias_sweep(
    out_dir: str,
    sweep: Dict[str, object],
    *,
    highlight_L: Optional[int] = 5,
    highlight_p: Optional[float] = 0.07,
    filename_detection: str = "noise_est_loss_bias_detection.png",
) -> Optional[str]:
    """Render only the loss-bias detection figure used by noise_est.tex."""

    Ls = list(sweep["L_grid"])  # type: ignore[index]
    Ps = list(sweep["p_noisy_grid"])  # type: ignore[index]
    det = np.array(sweep["det_matrix"], dtype=float)
    verify_counts_list = sweep.get("verify_counts")

    _ensure_dir(out_dir)

    det_cube = np.array(sweep.get("det_cube", []), dtype=float)
    det_cube_min = np.array(sweep.get("det_cube_min", []), dtype=float)
    det_cube_max = np.array(sweep.get("det_cube_max", []), dtype=float)
    det_plot = np.ma.masked_invalid(det)
    X, Y = np.meshgrid(Ps, Ls)
    line_handles: List[Line2D] = []

    fig_det, ax_det = plt.subplots(figsize=(7.2, 4.0))
    handled_detection = False
    if (
        det_cube.ndim == 3
        and verify_counts_list
        and len(verify_counts_list) == det_cube.shape[0]
        and len(Ps) == det_cube.shape[2]
    ):
        S_vals = np.array(verify_counts_list, dtype=float)
        order = np.argsort(S_vals)
        S_vals = S_vals[order]

        if len(Ls) == 1:
            det_lines = (
                det_cube[:, 0, :][order, :]
                if det_cube.ndim == 3
                else np.empty((0, len(Ps)))
            )
            det_lines_min = (
                det_cube_min[:, 0, :][order, :]
                if det_cube_min.ndim == 3
                else np.empty_like(det_lines)
            )
            det_lines_max = (
                det_cube_max[:, 0, :][order, :]
                if det_cube_max.ndim == 3
                else np.empty_like(det_lines)
            )
            for j, p_val in enumerate(Ps):
                line = det_lines[:, j]
                line_min = det_lines_min[:, j]
                line_max = det_lines_max[:, j]
                mask = np.isfinite(line)
                if det_lines_min.size:
                    mask &= np.isfinite(line_min)
                if det_lines_max.size:
                    mask &= np.isfinite(line_max)
                if not mask.any():
                    continue
                S_valid = S_vals[mask]
                mean_vals = line[mask]
                is_highlight = bool(
                    highlight_p is not None
                    and math.isclose(
                        float(p_val), float(highlight_p), rel_tol=0, abs_tol=1e-9
                    )
                )
                order_valid = np.argsort(S_valid)
                S_line = S_valid[order_valid]
                mean_line = mean_vals[order_valid]
                smooth_line = _smooth_series(mean_line, window=5)
                band_lower = band_upper = None
                if det_lines_min.size and det_lines_max.size:
                    min_vals = line_min[mask]
                    max_vals = line_max[mask]
                    if min_vals.size and max_vals.size:
                        min_line = min_vals[order_valid]
                        max_line = max_vals[order_valid]
                        smooth_min = _smooth_series(min_line, window=5)
                        smooth_max = _smooth_series(max_line, window=5)
                        band_lower = np.minimum(smooth_min, smooth_max)
                        band_upper = np.maximum(smooth_min, smooth_max)
                        span_factor = 0.25
                        band_lower = smooth_line - span_factor * (
                            smooth_line - band_lower
                        )
                        band_upper = smooth_line + span_factor * (
                            band_upper - smooth_line
                        )
                        band_lower = np.clip(band_lower, 0.0, 1.0)
                        band_upper = np.clip(band_upper, 0.0, 1.0)
                color, marker = _series_style(j)
                mark_step = max(1, S_line.size // 8)
                line_kwargs: Dict[str, object] = {
                    "linewidth": 1.6 if is_highlight else 1.4,
                    "alpha": 0.92,
                    "label": rf"$p_{{\mathrm{{L}}}}={float(p_val) * 100:.1f}\%$",
                    "marker": marker,
                    "markersize": 6.0 if is_highlight else 5.0,
                    "markeredgewidth": 0.8,
                    "color": color,
                    "markerfacecolor": color,
                    "markeredgecolor": color,
                    "markevery": mark_step,
                }
                line_obj = ax_det.plot(S_line, smooth_line, **line_kwargs)[0]
                line_handles.append(line_obj)
                # No shaded band to keep lines uncluttered
            handled_detection = True
        elif len(Ps) == 1:
            det_lines = (
                det_cube[:, :, 0][order, :]
                if det_cube.ndim == 3
                else np.empty((0, len(Ls)))
            )
            det_lines_min = (
                det_cube_min[:, :, 0][order, :]
                if det_cube_min.ndim == 3
                else np.empty_like(det_lines)
            )
            det_lines_max = (
                det_cube_max[:, :, 0][order, :]
                if det_cube_max.ndim == 3
                else np.empty_like(det_lines)
            )
            if highlight_L is not None and any(
                int(count) == int(highlight_L) for count in Ls
            ):
                target_idx = next(
                    idx
                    for idx, count in enumerate(Ls)
                    if int(count) == int(highlight_L)
                )
            else:
                target_idx = 0
            Lval = Ls[target_idx]
            line = det_lines[:, target_idx]
            line_min = det_lines_min[:, target_idx]
            line_max = det_lines_max[:, target_idx]
            mask = np.isfinite(line)
            if det_lines_min.size:
                mask &= np.isfinite(line_min)
            if det_lines_max.size:
                mask &= np.isfinite(line_max)
            if mask.any():
                S_valid = S_vals[mask]
                mean_vals = line[mask]
                order_valid = np.argsort(S_valid)
                S_line = S_valid[order_valid]
                mean_line = mean_vals[order_valid]
                smooth_line = _smooth_series(mean_line, window=5)
                band_lower = band_upper = None
                if det_lines_min.size and det_lines_max.size:
                    min_vals = line_min[mask]
                    max_vals = line_max[mask]
                    if min_vals.size and max_vals.size:
                        min_line = min_vals[order_valid]
                        max_line = max_vals[order_valid]
                        smooth_min = _smooth_series(min_line, window=5)
                        smooth_max = _smooth_series(max_line, window=5)
                        band_lower = np.minimum(smooth_min, smooth_max)
                        band_upper = np.maximum(smooth_min, smooth_max)
                        span_factor = 0.25
                        band_lower = smooth_line - span_factor * (
                            smooth_line - band_lower
                        )
                        band_upper = smooth_line + span_factor * (
                            band_upper - smooth_line
                        )
                        band_lower = np.clip(band_lower, 0.0, 1.0)
                        band_upper = np.clip(band_upper, 0.0, 1.0)
            line_label = f"L={Lval}"
            color, marker = _series_style(target_idx)
            mark_step = max(1, S_line.size // 8)
            line_obj = ax_det.plot(
                S_line,
                smooth_line,
                linewidth=1.6,
                color=color,
                alpha=0.94,
                label=line_label,
                marker=marker,
                markersize=5.2,
                markeredgewidth=0.8,
                markerfacecolor=color,
                markeredgecolor=color,
                markevery=mark_step,
            )[0]
            line_handles.append(line_obj)
            # No shaded interval for single-L curves
            handled_detection = True

        if handled_detection:
            ax_det.set_xlabel("Verification size $\\mathcal{S}$")
            ax_det.set_ylabel("Noise detection rate")
            ax_det.set_ylim(0.0, 1.0)
            axis_vals = (
                np.array(verify_counts_list, dtype=float)
                if verify_counts_list
                else np.array([], dtype=float)
            )
            _configure_verification_axis(
                ax_det,
                axis_vals if axis_vals.size else [1.0],
                start_at_zero=False,
                use_data_ticks=True,
                max_data_ticks=6,
            )
            _style_axes(ax_det, grid="both")
            tidy_ticks(ax_det, y_major=6)
            if line_handles:
                dedup_legend(
                    ax_det,
                    loc="best",
                    handles=line_handles,
                    labels=[h.get_label() for h in line_handles],
                )
            else:
                dedup_legend(ax_det, loc="best")

    if not handled_detection:
        cmap_det = colormaps["viridis"].resampled(256)
        im_det = ax_det.imshow(
            det_plot,
            origin="lower",
            aspect="auto",
            extent=[min(Ps), max(Ps), min(Ls), max(Ls)],
            vmin=0.0,
            vmax=1.0,
            cmap=cmap_det,
        )
        cbar = fig_det.colorbar(im_det, ax=ax_det, pad=0.02)
        cbar.set_label("Detection rate (true positives)")
        if det_plot.size:
            try:
                cs_det = ax_det.contour(
                    X,
                    Y,
                    det_plot,
                    levels=[0.8, 0.9, 0.95],
                    linewidths=[1.0, 1.2, 1.4],
                    linestyles=["--", "--", "-."],
                    colors="k",
                )
                ax_det.clabel(cs_det, inline=True, fontsize=8, fmt="%.2f")
            except Exception:
                pass
        if highlight_L is not None and highlight_p is not None:
            ax_det.plot(
                [highlight_p],
                [highlight_L],
                marker="x",
                markersize=8,
                color=ACCENT_COLOR,
            )
        ax_det.set_xlabel("Noisy lieutenant loss $p_{\\mathrm{L}}$")
        ax_det.set_ylabel("Lieutenants $L$ (= $n-1$)")
        ax_det.set_yticks(Ls)
        ax_det.set_yticklabels([str(int(count)) for count in Ls])
        ax_det.set_xlim(left=0.0)
        _style_axes(ax_det, grid="both")
        y_bins = max(4, len(Ls)) if Ls else None
        tidy_ticks(ax_det, x_major=6, y_major=y_bins)
    ax_det.set_title("Per-Node Diagnosis: Lieutenant Noise Detection Rate")
    det_path = os.path.join(out_dir, filename_detection)
    _save_figure(fig_det, det_path)
    return det_path


def parse_list(s: str, target_len: int, default_val: float) -> List[float]:
    s = s.strip()
    if not s:
        return [default_val] * target_len
    vals = [float(x) for x in s.split(",") if x.strip()]
    if len(vals) == 1 and target_len > 1:
        return [vals[0]] * target_len
    if len(vals) != target_len:
        raise SystemExit(
            f"Expected {target_len} comma-separated values, got {len(vals)}"
        )
    return vals


def _as_list_of_floats(val: object, target_len: int, default_val: float) -> List[float]:
    if val is None:
        return [default_val] * target_len
    if isinstance(val, (int, float)):
        return [float(val)] * target_len
    if isinstance(val, str):
        return parse_list(val, target_len, default_val)
    if isinstance(val, (list, tuple)):
        vals = [float(x) for x in val]  # type: ignore[assignment]
        if len(vals) == 1 and target_len > 1:
            return [vals[0]] * target_len
        if len(vals) != target_len:
            raise SystemExit(f"Expected {target_len} values, got {len(vals)} in config")
        return vals

    return [default_val] * target_len


AVAILABLE_FIGURES: Dict[str, str] = {
    "policy-plane": "Baseline vs ours policy plane",
    "loss-bias": "Loss-bias (1$\\to$0) detector",
    "commander-detection": "Commander loss detection",
}

FIGURE_ALIASES: Dict[str, str] = {
    "amplitude-bias": "loss-bias",
}

# Embedded runtime defaults (formerly noise_est.toml).
DEFAULT_CONFIG: Dict[str, object] = {
    "n": 6,
    "shots": 500,
    "verify_frac": 0.25,
    "engine": "perceval",
    "epsilon": 0.05,
    "epsilon0": 0.025,
    "epsilon1": 0.03,
    "epsilon_delta": 0.005,
    "delta": 0.05,
    "flag_rule": "lower",
    "tau_leak": 0.135,
    "tau_fid": 0.89,
    "p_L": 0.03,
    "q_class": 0.0,
    "dcr_hz": 10,
    "gate_ns": 1.0,
    "noisy_p_L": 0.05,
    "p_phys_cmd": "0.01",
    "pcvl_indist": 0.92,
    "pcvl_phase_error": 0.01,
    "pcvl_phase_imprecision": 0.02,
    "pcvl_g2": 0.01,
    "qkd_key": "a1b2c3d4e5f6",
    "batch_id": "batch-0001",
    "seed": 20240501,
    "policy_plane_clean_p_L": 0.015,
    "policy_plane_clean_seed": 20240502,
    "policy_plane_clean_windows": 20,
    "windows": 100,
    "quarantine_w": 1,
    "readmit_eps_frac": 0.5,
    "sweep_enable": True,
    "sweep_n": "6",
    "sweep_p_noisy": "0.04,0.06,0.08",
    "sweep_p_clean": 0.02,
    "sweep_noisy_count": 1,
    "sweep_reps": 100,
    "sweep_shots": 4000,
    "sweep_verify_frac": 0.25,
    "sweep_verify_counts": "80,160,200,240,280,320,480,640,800,1000,1500,2000,3000",
    "highlight_L": 5,
    "highlight_p": 0.06,
    "commander_p_C": "0.04,0.06,0.08",
    "commander_shots": 4000,
    "commander_verify_counts": "80,160,200,240,280,320,480,640,800,1000,1500,2000,3000",
    "commander_highlight_p_cmd": 0.06,
    "compare_p_L": "0.04,0.06,0.08",
    "compare_p_C": "0.04,0.06,0.08",
    "compare_verify_counts": "200,320,480,640,800,1200,1600,2000,3000",
    "compare_lieutenant_shots": 4000,
    "compare_commander_shots": 4000,
    "compare_lieutenant_reps": 100,
    "compare_commander_reps": 100,
    "compare_lieutenant_verify_frac": 0.25,
    "compare_highlight_p_L": 0.04,
    "compare_highlight_p_C": 0.04,
}


def _parse_figures(fig_args: Optional[Sequence[str]]) -> List[str]:
    if not fig_args:
        return list(AVAILABLE_FIGURES.keys())

    requested: List[str] = []
    for entry in fig_args:
        for part in entry.split(","):
            name = part.strip().lower()
            if name:
                requested.append(FIGURE_ALIASES.get(name, name))

    if not requested or "all" in requested:
        return list(AVAILABLE_FIGURES.keys())

    invalid = [name for name in requested if name not in AVAILABLE_FIGURES]
    if invalid:
        bad = ", ".join(sorted(set(invalid)))
        raise SystemExit(f"Unknown figure name(s): {bad}")

    unique: List[str] = []
    for name in requested:
        if name not in unique:
            unique.append(name)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Noise estimation simulator (single-file defaults)", add_help=True
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for figures/JSON (overrides built-in default)",
    )
    parser.add_argument(
        "--figures",
        "-f",
        action="append",
        help=(
            "Comma-separated list of figures to generate ("
            + ", ".join(sorted(AVAILABLE_FIGURES))
            + " or 'all'). Repeatable; default matches figures used in noise_est.tex."
        ),
    )
    parser.add_argument(
        "--policy-plane-details",
        action="store_true",
        help="Print per-batch baseline vs. ours comparison table for the policy-plane",
    )
    args = parser.parse_args()

    cfg_data = dict(DEFAULT_CONFIG)
    selected_figures = _parse_figures(args.figures)

    n = int(cfg_data.get("n", 6))
    if n < 3:
        raise SystemExit("n must be >= 3")
    num_lieutenants = n - 1

    shots = int(cfg_data.get("shots", 5000))
    verify_frac = float(cfg_data.get("verify_frac", 0.25))
    epsilon = float(cfg_data.get("epsilon", 0.05))
    delta = float(cfg_data.get("delta", 0.05))

    epsilon0 = float(cfg_data.get("epsilon0", 0.5 * epsilon))
    epsilon1 = float(cfg_data.get("epsilon1", epsilon))
    epsilon_delta = float(cfg_data.get("epsilon_delta", 0.005))
    tau_leak_cfg = float(cfg_data.get("tau_leak", cfg_data.get("tau_q", 0.135)))
    tau_fid_cfg = float(cfg_data.get("tau_fid", cfg_data.get("F_min", 0.89)))

    seed_cfg = cfg_data.get("seed")
    if seed_cfg is not None:
        try:
            run_seed = int(str(seed_cfg), 0)
        except ValueError as exc:
            raise SystemExit(f"Invalid seed value '{seed_cfg}' in config") from exc
    else:
        run_seed = secrets.randbits(64)

    p_L_default = 0.03
    raw_p_L = cfg_data.get("p_L")
    if raw_p_L is None:
        raw_p_L = cfg_data.get("p_phys")
    if raw_p_L is None:
        raw_p_L = p_L_default
    p_L_list = _as_list_of_floats(raw_p_L, num_lieutenants, p_L_default)
    p_phys_list = p_L_list  # legacy name retained for downstream code
    q_class_list = _as_list_of_floats(
        cfg_data.get("q_class", 0.0), num_lieutenants, 0.0
    )

    p_phys_jitter = 0.0
    q_class_jitter = 0.0

    if any(q > 0.0 for q in q_class_list):
        print(
            "[INFO] Clamping q_class to 0 per error-free classical channel assumption."
        )
        q_class_list = [0.0] * num_lieutenants

    noisy_index = int(cfg_data.get("noisy_index", 1))
    raw_noisy_p = cfg_data.get("noisy_p_L")
    if raw_noisy_p is None:
        raw_noisy_p = cfg_data.get("noisy_p_phys", 0.07)
    noisy_p_phys = float(raw_noisy_p)

    pcvl_dark_scale = float(cfg_data.get("pcvl_dark_scale", 1.0))
    pcvl_g2 = float(cfg_data.get("pcvl_g2", 0.01))
    pcvl_indist = float(cfg_data.get("pcvl_indist", 0.92))
    pcvl_phase_error = float(cfg_data.get("pcvl_phase_error", 0.01))
    pcvl_phase_imprecision = float(cfg_data.get("pcvl_phase_imprecision", 0.02))

    engine_str = str(cfg_data.get("engine", "perceval")).lower()
    if engine_str != "perceval":
        raise SystemExit(
            "noise_est is configured for Perceval-only execution; set engine='perceval'."
        )
    pcvl_noise = pcvl.NoiseModel(
        g2=pcvl_g2,
        indistinguishability=pcvl_indist,
        phase_error=pcvl_phase_error,
        phase_imprecision=pcvl_phase_imprecision,
    )
    engine_mode: Literal["fast", "perceval"] = "perceval"

    dcr_hz_list = _as_list_of_floats(
        cfg_data.get("dcr_hz", 10.0), num_lieutenants, 10.0
    )
    gate_ns_list = _as_list_of_floats(
        cfg_data.get("gate_ns", 1.0), num_lieutenants, 1.0
    )
    commander_default_loss = 0.01
    raw_p_C = cfg_data.get("p_C")
    if raw_p_C is None:
        raw_p_C = cfg_data.get("p_phys_cmd")
    if raw_p_C is None:
        raw_p_C = commander_default_loss
    p_phys_cmd_vals = _as_list_of_floats(raw_p_C, 2, commander_default_loss)
    dcr_cmd_vals = _as_list_of_floats(cfg_data.get("dcr_hz_cmd", 0.0), 2, 0.0)
    gate_cmd_vals = _as_list_of_floats(cfg_data.get("gate_ns_cmd", 0.0), 2, 0.0)

    flag_rule = _parse_flag_rule(str(cfg_data.get("flag_rule", "lower")))
    out_dir_cfg = str(cfg_data.get("out_dir", os.path.dirname(__file__) or "."))
    out_dir = args.out or out_dir_cfg
    windows = int(cfg_data.get("windows", 1))
    quarantine_w = int(cfg_data.get("quarantine_w", 1))
    readmit_eps_frac = float(cfg_data.get("readmit_eps_frac", 0.5))

    if p_phys_jitter > 0.0:
        rng_j = random.Random(run_seed ^ 0xA5A5_1357)
        p_phys_list = [
            max(0.0, min(1.0, v + rng_j.uniform(-p_phys_jitter, p_phys_jitter)))
            for v in p_phys_list
        ]
    if q_class_jitter > 0.0:
        rng_j2 = random.Random((run_seed + 0x55AA_2468) & 0xFFFFFFFF)
        q_class_list = [
            max(0.0, min(1.0, v + rng_j2.uniform(-q_class_jitter, q_class_jitter)))
            for v in q_class_list
        ]

        if any(q > 0.0 for q in q_class_list):
            q_class_list = [0.0] * num_lieutenants

    if noisy_index > 0:
        idx0 = noisy_index - 1
        if not (0 <= idx0 < num_lieutenants):
            raise SystemExit(
                f"noisy_index must be in [1,{num_lieutenants}] or 0 to disable"
            )
        p_phys_list[idx0] = max(0.0, min(1.0, noisy_p_phys))

    delta_eff = delta

    cfg = SimulationConfig(
        num_generals=n,
        shots=shots,
        verify_fraction=verify_frac,
        epsilon_threshold=epsilon,
        delta=delta_eff,
        seed=run_seed,
        flag_rule=flag_rule,
        engine=engine_mode,
        windows=windows,
        quarantine_w=quarantine_w,
        readmit_eps_frac=readmit_eps_frac,
        epsilon0=epsilon0,
        epsilon1=epsilon1,
        epsilon_delta=epsilon_delta,
        qkd_key=str(cfg_data.get("qkd_key", "")),
        batch_id=str(cfg_data.get("batch_id", "batch-0")),
        pcvl_noise_model=pcvl_noise,
        pcvl_dark_scale=pcvl_dark_scale,
        dcr_hz_list=dcr_hz_list,
        gate_ns_list=gate_ns_list,
        p_phys_cmd=(p_phys_cmd_vals[0], p_phys_cmd_vals[1]),
        dcr_hz_cmd=(dcr_cmd_vals[0], dcr_cmd_vals[1]),
        gate_ns_cmd=(gate_cmd_vals[0], gate_cmd_vals[1]),
        tau_leak=tau_leak_cfg,
        tau_fid=tau_fid_cfg,
    )

    sweep_enable = bool(cfg_data.get("sweep_enable", True))
    sweep_n_spec = str(cfg_data.get("sweep_n", "6:8:2"))
    sweep_p_spec = str(cfg_data.get("sweep_p_noisy", "0.04,0.06,0.08"))
    sweep_reps = int(cfg_data.get("sweep_reps", 6))
    sweep_noisy_count = int(cfg_data.get("sweep_noisy_count", 1))
    sweep_p_clean = float(cfg_data.get("sweep_p_clean", 0.02))
    sweep_shots = int(cfg_data.get("sweep_shots", shots))
    sweep_verify_frac = float(cfg_data.get("sweep_verify_frac", verify_frac))
    sweep_verify_counts_spec = cfg_data.get("sweep_verify_counts")
    sweep_verify_counts_list: Optional[List[int]]
    if sweep_verify_counts_spec is None:
        sweep_verify_counts_list = None
    else:
        sweep_verify_counts_list = _parse_int_grid(str(sweep_verify_counts_spec))
    highlight_L = int(cfg_data.get("highlight_L", 5))
    highlight_p = float(cfg_data.get("highlight_p", noisy_p_phys))

    commander_grid_cfg = cfg_data.get("commander_p_C")
    if commander_grid_cfg is None:
        commander_grid_cfg = cfg_data.get("commander_p_cmd", "0.04,0.06,0.08")
    commander_p_cmd_values = _parse_float_grid(str(commander_grid_cfg))
    commander_reps = int(cfg_data.get("commander_reps", sweep_reps))
    commander_shots_raw = cfg_data.get("commander_shots")
    commander_shots_override = (
        int(str(commander_shots_raw), 0)
        if commander_shots_raw is not None
        else sweep_shots
    )
    commander_verify_counts_spec = cfg_data.get("commander_verify_counts")
    if commander_verify_counts_spec is None:
        commander_verify_counts_list = sweep_verify_counts_list
    else:
        commander_verify_counts_list = _parse_int_grid(
            str(commander_verify_counts_spec)
        )
    commander_highlight_val = cfg_data.get("commander_highlight_p_cmd")
    if commander_highlight_val is None:
        commander_highlight_val = cfg_data.get("highlight_p")
    commander_highlight_float: Optional[float]
    if commander_highlight_val is None:
        commander_highlight_float = None
    else:
        try:
            commander_highlight_float = float(commander_highlight_val)
        except (TypeError, ValueError):
            commander_highlight_float = None

    requires_single = "policy-plane" in selected_figures
    requires_policy_plane = "policy-plane" in selected_figures

    single_result: Dict[str, object] | None = None
    if requires_single:
        single_result = run_simulation(cfg, p_phys_list, q_class_list)

    timeline: Optional[Sequence[Dict[str, object]]] = None
    combined_timeline: List[Dict[str, object]] = []
    if requires_policy_plane:
        stress_batches = int(cfg_data.get("policy_plane_stress_batches", windows))
        stress_seed_cfg = cfg_data.get("policy_plane_stress_seed")
        if stress_seed_cfg is not None:
            try:
                stress_seed = int(str(stress_seed_cfg), 0)
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid policy_plane_stress_seed value '{stress_seed_cfg}'"
                ) from exc
        else:
            stress_seed = cfg.seed ^ 0xACE5_1234

        stress_timeline = _run_independent_policy_plane_batches(
            cfg,
            p_phys_list,
            q_class_list,
            batch_count=max(0, stress_batches),
            seed=stress_seed,
            tag="stress",
            batch_id_prefix=f"{cfg.batch_id}-stress",
        )
        for entry in stress_timeline:
            tagged = dict(entry)
            tagged["window"] = len(combined_timeline) + 1
            combined_timeline.append(tagged)
        timeline = combined_timeline if combined_timeline else None

    policy_plane_clean_p_cfg = cfg_data.get("policy_plane_clean_p_L")
    if policy_plane_clean_p_cfg is None:
        policy_plane_clean_p_cfg = cfg_data.get("policy_plane_clean_p_phys")
    policy_plane_clean_seed_cfg = cfg_data.get("policy_plane_clean_seed")
    policy_plane_clean_windows = int(
        cfg_data.get("policy_plane_clean_windows", windows)
    )

    if "policy-plane" in selected_figures and policy_plane_clean_p_cfg is not None:
        if single_result is None:
            single_result = run_simulation(cfg, p_phys_list, q_class_list)

        if policy_plane_clean_seed_cfg is not None:
            try:
                clean_seed = int(str(policy_plane_clean_seed_cfg), 0)
            except ValueError as exc:
                raise SystemExit(
                    f"Invalid policy_plane_clean_seed value '{policy_plane_clean_seed_cfg}'"
                ) from exc
        else:
            clean_seed = cfg.seed ^ 0xC1EA0ACE

        clean_p_list = _as_list_of_floats(
            policy_plane_clean_p_cfg,
            num_lieutenants,
            float(policy_plane_clean_p_cfg),
        )

        cfg_clean = replace(cfg, seed=clean_seed, windows=1)
        clean_timeline = _run_independent_policy_plane_batches(
            cfg_clean,
            clean_p_list,
            q_class_list,
            batch_count=max(0, policy_plane_clean_windows),
            seed=clean_seed,
            tag="clean",
            batch_id_prefix=f"{cfg.batch_id}-clean",
        )
        if clean_timeline:
            offset = len(combined_timeline)
            for entry in clean_timeline:
                tagged = dict(entry)
                tagged["window"] = offset + int(tagged.get("window", 0))
                tagged["policy_plane_tag"] = "clean"
                combined_timeline.append(tagged)
        timeline = combined_timeline if combined_timeline else None

    if "loss-bias" in selected_figures:
        if sweep_enable:
            n_grid = _parse_int_grid(sweep_n_spec)
            if not n_grid:
                n_grid = [n]
            p_grid = _parse_float_grid(sweep_p_spec)
            if not p_grid:
                p_grid = [noisy_p_phys]

            sweep = sweep_loss_bias_grid(
                cfg,
                n_list=n_grid,
                p_noisy_list=p_grid,
                p_clean=sweep_p_clean,
                noisy_count=sweep_noisy_count,
                reps=sweep_reps,
                shots_override=sweep_shots,
                verify_frac_override=sweep_verify_frac,
                verify_counts=sweep_verify_counts_list,
                seed=run_seed ^ 0xBADC0FFE,
            )
            _ensure_dir(out_dir)
            sweep_path = os.path.join(out_dir, "noise_est_loss_bias_sweep.json")
            with open(sweep_path, "w", encoding="utf-8") as f:
                json.dump(sweep, f, indent=2)

            plot_loss_bias_sweep(
                out_dir,
                sweep,
                highlight_L=highlight_L,
                highlight_p=highlight_p,
                filename_detection="noise_est_loss_bias_detection.png",
            )
        else:
            print(
                "[loss-bias] sweep_enable=false; skipping non-TeX loss-bias distribution figure."
            )

    if "commander-detection" in selected_figures:
        if not commander_p_cmd_values:
            commander_p_cmd_values = [float(cfg.p_phys_cmd[0])]
        sweep_cmd = sweep_commander_detection_grid(
            cfg,
            commander_p_cmd_values,
            p_phys_list,
            q_class_list,
            verify_counts=commander_verify_counts_list,
            shots_override=commander_shots_override,
            reps=max(1, commander_reps),
            seed=run_seed ^ 0xC0DEA55,
        )
        _ensure_dir(out_dir)
        sweep_cmd_path = os.path.join(out_dir, "noise_est_commander_sweep.json")
        with open(sweep_cmd_path, "w", encoding="utf-8") as f:
            json.dump(sweep_cmd, f, indent=2)
        plot_commander_detection(
            out_dir,
            sweep_cmd,
            highlight_p=commander_highlight_float,
            filename="noise_est_commander_detection.png",
        )

    if "policy-plane" in selected_figures:
        if single_result is None:
            single_result = run_simulation(cfg, p_phys_list, q_class_list)
        plot_policy_plane(
            out_dir,
            single_result,
            timeline,
            filename="noise_est_policy_plane.png",
            baseline_mode="support",
            epsilon_threshold=float(cfg.epsilon_threshold),
            detailed_log=args.policy_plane_details,
        )

    if requires_single:
        z_fp = inv_std_normal_cdf(1.0 - delta_eff)
        total_openings = int(round(verify_frac * shots))
        det_openings = int(round(verify_frac * shots * 2 / 3))
        summary = (
            f"Run seed={run_seed} z={z_fp:.4f} "
            f"\\mathcal{{S}}≈{total_openings} "
            f"\\mathcal{{S}}_{{\\mathrm{{det}}}}≈{det_openings}"
        )
        print(summary)

    print("Plots generated successfully.")


if __name__ == "__main__":
    main()
