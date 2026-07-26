#!/usr/bin/env python3
"""
benchmark/casp_scorer.py
========================

CASP-inspired scoring metrics for protein structure prediction.

Provides implementations of the four core metrics used in the
Critical Assessment of Structure Prediction (CASP) community:

    - GDT_TS -- Global Distance Test, Total Score
    - RMSD -- Root-Mean-Square Deviation
    - lDDT -- Local Distance Difference Test
    - TM-score -- Template Modelling Score

All metrics operate on sets of atomic coordinates (typically C\\ :sub:`α`\\ atoms)
and follow the published definitions.  Unless otherwise noted, coordinate
arrays are expected as ``(n_residues, 3)`` ``numpy.float64`` arrays.

References
----------
    Zemla, A. (2003) 'LGA: a method for finding 3D similarities in
        protein structures', *Nucleic Acids Research*, 31(13),
        pp. 3370--3374.  doi:10.1093/nar/gkg571.
    Kabsch, W. (1978) 'A discussion of the solution for the best
        rotation to relate two sets of vectors', *Acta Crystallographica
        Section A*, 34(5), pp. 827--828.  doi:10.1107/S0567739478001680.
    Mariani, V., Biasini, M., Barbato, A. and Schwede, T. (2013) 'lDDT:
        a local superposition-free score for comparing protein structures
        and models using distance difference tests', *Bioinformatics*,
        29(21), pp. 2722--2728.  doi:10.1093/bioinformatics/btt473.
    Zhang, Y. and Skolnick, J. (2004) 'Scoring function for automated
        assessment of protein structure template quality',
        *Proteins: Structure, Function, and Bioinformatics*, 57(4),
        pp. 702--710.  doi:10.1002/prot.20264.
    Kabsch, W. (1976) 'A solution for the best rotation to relate two
        sets of vectors', *Acta Crystallographica Section A*, 32(5),
        pp. 922--923.  doi:10.1107/S0567739476001873.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# PDB parsing
# ═══════════════════════════════════════════════════════════════════════

_RESIDUE_NAMES_3TO1: dict[str, str] = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
}


def parse_pdb(
    path: Union[str, Path],
    *,
    select_atom: str = 'CA',
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Read atomic coordinates from a PDB file.

    Parameters
    ----------
    path : str or Path
        Path to the PDB file.
    select_atom : str, optional
        Atom name to extract (default: ``'CA'``).  Pass ``'all'`` to
        extract all ATOM record coordinates.

    Returns
    -------
    coords : ndarray, shape (N, 3)
        Extracted coordinates in :math:`\\text{\\AA}`.
    res_names : list of str
        Three-letter residue names, one per extracted atom.
    chain_ids : list of str
        Chain identifiers, one per extracted atom.

    Notes
    -----
    Only standard ATOM records (columns 13--16 == ``"ATOM "``) are
    considered; HETATM records are ignored.  Alternate location
    indicators (column 17) default to ``' '`` or ``'A'``.
    """
    coords, res_names, chain_ids = [], [], []
    path = Path(path)

    with open(path, 'r') as fh:
        for line in fh:
            if not line.startswith(('ATOM  ', 'HETATM')):
                continue
            if not line.startswith('ATOM  '):
                continue  # skip HETATM for strict CASP evaluation

            atom_name = line[12:16].strip()
            if select_atom != 'all' and atom_name != select_atom:
                continue

            alt_loc = line[16]
            if alt_loc not in (' ', 'A'):
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            res_name = line[17:20].strip()
            chain_id = line[21].strip() or ' '
            coords.append([x, y, z])
            res_names.append(res_name)
            chain_ids.append(chain_id)

    if not coords:
        raise ValueError(
            f"No ATOM records found in '{path}' "
            f"(select_atom='{select_atom}')."
        )

    return np.asarray(coords, dtype=np.float64), res_names, chain_ids


def _validate_coordinates(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
) -> None:
    """Check that both coordinate arrays have the same shape."""
    if coords_pred.shape != coords_ref.shape:
        raise ValueError(
            f"Predicted ({coords_pred.shape}) and reference "
            f"({coords_ref.shape}) coordinate arrays must have the "
            f"same shape."
        )
    if coords_pred.ndim != 2 or coords_pred.shape[1] != 3:
        raise ValueError(
            f"Coordinate arrays must be (N, 3); got {coords_pred.shape}."
        )


# ═══════════════════════════════════════════════════════════════════════
# Kabsch algorithm  --  optimal RMSD superposition
# ═══════════════════════════════════════════════════════════════════════


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Compute the optimal rotation matrix from P to Q (Kabsch, 1976, 1978).

    Parameters
    ----------
    P : ndarray, shape (N, 3)
        Moving coordinates (centred).
    Q : ndarray, shape (N, 3)
        Target coordinates (centred).

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix such that ``(P @ R)`` aligns with Q.
    """
    C = P.T @ Q                          # covariance matrix (3, 3)
    V, S, Wt = np.linalg.svd(C)          # C = V @ diag(S) @ Wt

    # Ensure a right-handed coordinate system (reflection correction).
    d = np.sign(np.linalg.det(V @ Wt))
    U = np.diag([1.0, 1.0, d])
    R = (V @ U) @ Wt

    return R


def kabsch_superimpose(
    coords_moving: np.ndarray,
    coords_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Superimpose *coords_moving* onto *coords_target* via the Kabsch
    algorithm (Kabsch, 1976, 1978).

    Parameters
    ----------
    coords_moving : ndarray, shape (N, 3)
        Coordinates to be rotated.
    coords_target : ndarray, shape (N, 3)
        Reference coordinates.

    Returns
    -------
    coords_superposed : ndarray, shape (N, 3)
        Superimposed coordinates.
    R : ndarray, shape (3, 3)
        Rotation matrix applied.
    rmsd_val : float
        RMSD after superposition.
    """
    _validate_coordinates(coords_moving, coords_target)
    N = coords_moving.shape[0]

    # Centre both sets.
    centroid_m = coords_moving.mean(axis=0)
    centroid_t = coords_target.mean(axis=0)
    Pm = coords_moving - centroid_m
    Pt = coords_target - centroid_t

    # Optimal rotation.
    R = _kabsch_rotation(Pm, Pt)

    # Apply rotation.
    coords_superposed = Pm @ R + centroid_t

    # RMSD.
    diff = coords_superposed - coords_target
    rmsd_val = math.sqrt((diff * diff).sum() / N)

    return coords_superposed, R, rmsd_val


# ═══════════════════════════════════════════════════════════════════════
# RMSD
# ═══════════════════════════════════════════════════════════════════════


def compute_rmsd(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
    *,
    superposition: bool = True,
) -> float:
    """
    Root-mean-square deviation between predicted and reference
    coordinates.

    Parameters
    ----------
    coords_pred : ndarray, shape (N, 3)
        Predicted (model) coordinates.
    coords_ref : ndarray, shape (N, 3)
        Reference (native) coordinates.
    superposition : bool, optional
        Whether to perform optimal Kabsch superposition before
        computing RMSD (default: ``True``).

    Returns
    -------
    rmsd : float
        RMSD in :math:`\\text{\\AA}`.

    References
    ----------
    Kabsch, W. (1978) *Acta Crystallographica Section A*, 34(5),
    pp. 827--828.  doi:10.1107/S0567739478001680.
    """
    _validate_coordinates(coords_pred, coords_ref)
    N = coords_pred.shape[0]

    if superposition:
        _, _, rmsd_val = kabsch_superimpose(coords_pred, coords_ref)
        return rmsd_val
    else:
        diff = coords_pred - coords_ref
        return math.sqrt((diff * diff).sum() / N)


# ═══════════════════════════════════════════════════════════════════════
# GDT_TS
# ═══════════════════════════════════════════════════════════════════════


def compute_gdt_ts(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
    *,
    thresholds: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    superpose_first: bool = True,
) -> float:
    """
    Global Distance Test, Total Score (GDT_TS).

    GDT_TS = :math:`\\frac{1}{4} (P_1 + P_2 + P_4 + P_8)`, where each
    :math:`P_t` is the percentage of residues---after optimal
    superposition---whose predicted--reference distance is less than
    *t* :math:`\\text{\\AA}`.

    Parameters
    ----------
    coords_pred : ndarray, shape (N, 3)
        Predicted (model) C\\ :sub:`α`\\ coordinates.
    coords_ref : ndarray, shape (N, 3)
        Reference (native) C\\ :sub:`α`\\ coordinates.
    thresholds : sequence of float, optional
        Distance thresholds in :math:`\\text{\\AA}` (default:
        ``(1.0, 2.0, 4.0, 8.0)``, giving GDT_TS).
    superpose_first : bool, optional
        Whether to optimally superimpose before computing distances
        (default: ``True``).  Set to ``False`` for GDT_TS without
        superposition (GDT_TS_NSA).

    Returns
    -------
    gdt : float
        GDT_TS score, ranging from 0 (worst) to 100 (best).

    References
    ----------
    Zemla, A. (2003) *Nucleic Acids Research*, 31(13), pp. 3370--3374.
    doi:10.1093/nar/gkg571.
    """
    _validate_coordinates(coords_pred, coords_ref)
    N = coords_pred.shape[0]

    if superpose_first:
        coords_pred, _, _ = kabsch_superimpose(coords_pred, coords_ref)

    # Euclidean distances (N,).
    distances = np.linalg.norm(coords_pred - coords_ref, axis=1)

    n_thresholds = len(thresholds)
    score_sum = 0.0

    for t in thresholds:
        count = float(np.sum(distances < t))
        score_sum += count / N * 100.0

    gdt = score_sum / n_thresholds
    return gdt


# ═══════════════════════════════════════════════════════════════════════
# lDDT
# ═══════════════════════════════════════════════════════════════════════


def compute_lddt(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
    *,
    inclusion_radius: float = 15.0,
    distance_thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """
    Local Distance Difference Test (lDDT), a superposition-free score
    for comparing protein structure models.

    lDDT evaluates the fraction of local inter-atomic distances in the
    model that are consistent with the reference.  A distance is
    considered ``preserved'' if the absolute difference between the
    model distance and the reference distance is below one of four
    thresholds.

    Only pairs of residues whose reference distance lies within
    *inclusion_radius* of each other are considered (``local'' subset).

    Parameters
    ----------
    coords_pred : ndarray, shape (N, 3)
        Predicted (model) coordinates.
    coords_ref : ndarray, shape (N, 3)
        Reference (native) coordinates.
    inclusion_radius : float, optional
        Maximum reference distance (:math:`\\text{\\AA}`) for a pair to
        be considered local (default: ``15.0``).
    distance_thresholds : sequence of float, optional
        Absolute distance-difference cutoffs (default: ``(0.5, 1.0,
        2.0, 4.0)``).

    Returns
    -------
    lddt : float
        lDDT score, ranging from 0 (poor) to 1 (perfect).

    References
    ----------
    Mariani, V. *et al.* (2013) *Bioinformatics*, 29(21),
    pp. 2722--2728.  doi:10.1093/bioinformatics/btt473.
    """
    _validate_coordinates(coords_pred, coords_ref)
    N = coords_pred.shape[0]

    if N < 2:
        return 1.0  # single residue is trivially preserved

    # Reference distance matrix  (upper triangle, excluding diagonal).
    D_ref = np.sqrt(
        np.maximum(
            np.sum((coords_ref[:, None, :] - coords_ref[None, :, :]) ** 2, axis=-1),
            0.0,
        )
    )
    D_pred = np.sqrt(
        np.maximum(
            np.sum((coords_pred[:, None, :] - coords_pred[None, :, :]) ** 2, axis=-1),
            0.0,
        )
    )

    # Mask for local pairs (i < j, within inclusion radius in reference).
    i_idx, j_idx = np.triu_indices(N, k=1)
    local_mask = D_ref[i_idx, j_idx] <= inclusion_radius
    i_local = i_idx[local_mask]
    j_local = j_idx[local_mask]

    if len(i_local) == 0:
        return 1.0  # no local pairs; trivially perfect

    # Absolute distance differences for local pairs.
    diff_local = np.abs(D_pred[i_local, j_local] - D_ref[i_local, j_local])

    # Fraction of pairs preserved under each threshold, averaged.
    total_pairs = len(i_local)
    n_thresholds = len(distance_thresholds)
    preserved_sum = 0.0

    for t in distance_thresholds:
        preserved_count = float(np.sum(diff_local <= t))
        preserved_sum += preserved_count / total_pairs

    lddt = preserved_sum / n_thresholds
    return lddt


# ═══════════════════════════════════════════════════════════════════════
# TM-score
# ═══════════════════════════════════════════════════════════════════════


def _tm_d0(L: int) -> float:
    """
    Length-dependent scale factor for TM-score.

    From Zhang & Skolnick (2004), equation (3):

        :math:`d_0(L) = 1.24 \\sqrt[3]{L - 15} - 1.8`

    for :math:`L > 19`; clamped to 0.5 for shorter lengths.
    """
    if L > 19:
        return 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8
    else:
        return 0.5


def _tm_scale_factor(L: int) -> float:
    """
    Normalisation factor for TM-score.

    For the standard TM-score (Zhang & Skolnick, 2004) the value is
    simply *L*, the length of the alignment.
    """
    return float(L)


def compute_tm_score(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
    *,
    n_iter: int = 5,
) -> float:
    """
    Template Modelling score (TM-score), a length-independent measure
    of global structural similarity.

    The TM-score is computed by iteratively superimposing the model
    onto the reference with residue weights that depend on the distance
    from the previous iteration:

        :math:`\\mathrm{TM} = \\frac{1}{L} \\sum_{i=1}^{L}
            \\frac{1}{1 + (d_i / d_0(L))^2}`

    where :math:`d_i` is the distance between residue *i* in the
    superposed model and the reference, and :math:`d_0(L)` is a
    length-dependent normalisation.

    Parameters
    ----------
    coords_pred : ndarray, shape (N, 3)
        Predicted (model) coordinates.
    coords_ref : ndarray, shape (N, 3)
        Reference (native) coordinates.
    n_iter : int, optional
        Number of iterative superposition cycles (default: ``5``).
        The original TM-score method (Zhang & Skolnick, 2004) uses
        a single iteration; subsequent iterations improve convergence.

    Returns
    -------
    tm : float
        TM-score, ranging from 0 (poor) to 1 (perfect).  Values above
        0.5 generally indicate the same fold.

    References
    ----------
    Zhang, Y. and Skolnick, J. (2004) *Proteins: Structure, Function,
        and Bioinformatics*, 57(4), pp. 702--710.
        doi:10.1002/prot.20264.
    """
    _validate_coordinates(coords_pred, coords_ref)
    L = coords_pred.shape[0]
    d0 = _tm_d0(L)

    # Start with coords_pred; iterate weighted superpositions.
    coords_current = coords_pred.copy()
    weights = np.ones(L, dtype=np.float64)

    for _ in range(n_iter):
        # Centre both sets.
        centroid_m = coords_current.mean(axis=0)
        centroid_t = coords_ref.mean(axis=0)
        Pm = coords_current - centroid_m
        Pt = coords_ref - centroid_t

        # Weighted Kabsch: apply weights to covariance.
        sqrt_w = np.sqrt(weights)[:, None]  # (N, 1)
        C = (sqrt_w * Pm).T @ (sqrt_w * Pt)

        V, S, Wt = np.linalg.svd(C)
        d = np.sign(np.linalg.det(V @ Wt))
        U = np.diag([1.0, 1.0, d])
        R = (V @ U) @ Wt

        coords_current = Pm @ R + centroid_t

        # Compute distances and update weights.
        dist = np.linalg.norm(coords_current - coords_ref, axis=1)
        weights = 1.0 / (1.0 + (dist / d0) ** 2)

    # Final TM-score (normalised by L).
    tm = float(weights.sum() / _tm_scale_factor(L))
    return tm


# ═══════════════════════════════════════════════════════════════════════
# Aggregate scorer
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class StructureScore:
    """
    Container for the complete set of structure prediction metrics.

    Attributes
    ----------
    gdt_ts : float
        GDT_TS score (0--100).
    rmsd : float
        RMSD after superposition (:math:`\\text{\\AA}`).
    lddt : float
        lDDT score (0--1).
    tm_score : float
        TM-score (0--1).
    n_residues : int
        Number of aligned residues.
    """

    gdt_ts: float
    rmsd: float
    lddt: float
    tm_score: float
    n_residues: int


def score_structure(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
    *,
    do_gdt: bool = True,
    do_lddt: bool = True,
    do_tm: bool = True,
) -> StructureScore:
    """
    Convenience function that computes all four CASP-style metrics in
    one call.

    Parameters
    ----------
    coords_pred : ndarray, shape (N, 3)
        Predicted (model) coordinates.
    coords_ref : ndarray, shape (N, 3)
        Reference (native) coordinates.
    do_gdt : bool, optional
        If ``True`` (default), compute GDT_TS.
    do_lddt : bool, optional
        If ``True`` (default), compute lDDT.
    do_tm : bool, optional
        If ``True`` (default), compute TM-score.

    Returns
    -------
    StructureScore
        Named tuple containing all requested metrics.

    Raises
    ------
    ValueError
        If the coordinate arrays have different shapes.
    """
    _validate_coordinates(coords_pred, coords_ref)
    N = coords_pred.shape[0]

    rmsd = compute_rmsd(coords_pred, coords_ref, superposition=True)
    gdt_ts = compute_gdt_ts(coords_pred, coords_ref) if do_gdt else float('nan')
    lddt_val = compute_lddt(coords_pred, coords_ref) if do_lddt else float('nan')
    tm_val = compute_tm_score(coords_pred, coords_ref) if do_tm else float('nan')

    return StructureScore(
        gdt_ts=gdt_ts,
        rmsd=rmsd,
        lddt=lddt_val,
        tm_score=tm_val,
        n_residues=N,
    )


# ═══════════════════════════════════════════════════════════════════════
# Self-test / validation
# ═══════════════════════════════════════════════════════════════════════

def _self_test() -> None:
    """Run a basic consistency check on all scoring functions."""
    np.random.seed(42)
    L = 200
    coords_ref = np.random.randn(L, 3).astype(np.float64) * 10.0

    # Perfect prediction.
    coords_perfect = coords_ref.copy()
    s_perfect = score_structure(coords_perfect, coords_ref)
    assert abs(s_perfect.gdt_ts - 100.0) < 1e-10, f"GDT_TS for perfect: {s_perfect.gdt_ts}"
    assert abs(s_perfect.rmsd) < 1e-10, f"RMSD for perfect: {s_perfect.rmsd}"
    assert abs(s_perfect.lddt - 1.0) < 1e-10, f"lDDT for perfect: {s_perfect.lddt}"
    assert abs(s_perfect.tm_score - 1.0) < 1e-10, f"TM-score for perfect: {s_perfect.tm_score}"

    # Slightly perturbed (sigma = 0.5 Å).
    coords_noisy = coords_ref + np.random.randn(L, 3).astype(np.float64) * 0.5
    s_noisy = score_structure(coords_noisy, coords_ref)
    assert s_noisy.gdt_ts > 90.0, f"GDT_TS too low: {s_noisy.gdt_ts}"
    assert 0.2 < s_noisy.rmsd < 1.0, f"RMSD unexpected: {s_noisy.rmsd}"
    assert s_noisy.lddt > 0.80, f"lDDT too low: {s_noisy.lddt}"
    assert s_noisy.tm_score > 0.95, f"TM-score too low: {s_noisy.tm_score}"

    # Very perturbed (sigma = 5.0 Å).
    coords_bad = coords_ref + np.random.randn(L, 3).astype(np.float64) * 5.0
    s_bad = score_structure(coords_bad, coords_ref)
    assert s_bad.gdt_ts < 60.0, f"GDT_TS too high: {s_bad.gdt_ts}"
    assert s_bad.rmsd > 3.0, f"RMSD too low: {s_bad.rmsd}"
    assert s_bad.tm_score < 0.8, f"TM-score too high: {s_bad.tm_score}"
    assert s_bad.lddt < 0.9, f"lDDT too high: {s_bad.lddt}"

    print(f"[benchmark/casp_scorer] Self-test passed")
    print(f"  Perfect:   GDT_TS={s_perfect.gdt_ts:.10f}  RMSD={s_perfect.rmsd:.10f}  "
          f"lDDT={s_perfect.lddt:.10f}  TM={s_perfect.tm_score:.10f}")
    print(f"  Noisy:     GDT_TS={s_noisy.gdt_ts:.4f}  RMSD={s_noisy.rmsd:.4f}  "
          f"lDDT={s_noisy.lddt:.4f}  TM={s_noisy.tm_score:.4f}")
    print(f"  Bad:       GDT_TS={s_bad.gdt_ts:.4f}  RMSD={s_bad.rmsd:.4f}  "
          f"lDDT={s_bad.lddt:.4f}  TM={s_bad.tm_score:.4f}")

    # Test GDT_TS numerical precision: self-consistent at high precision.
    gdt_1 = compute_gdt_ts(coords_noisy, coords_ref)
    gdt_2 = compute_gdt_ts(coords_noisy, coords_ref)
    assert abs(gdt_1 - gdt_2) < 1e-12, (
        f"GDT_TS not deterministic: {gdt_1} vs {gdt_2}"
    )
    print(f"  Determinism: GDT_TS diff = {abs(gdt_1 - gdt_2):.2e} (< 1e-12 passed)")


if __name__ == "__main__":
    _self_test()
