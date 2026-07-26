#!/usr/bin/env python3
"""
benchmark/decoy_cluster.py
==========================

SPICKER-based clustering of decoy structural ensembles.

Given a set of candidate protein conformations (decoys) produced by
a folding or sampling protocol, the SPICKER algorithm identifies the
most populated structural clusters.  The cluster centroids are
typically closer to the native structure than the average decoy,
and they provide high-confidence predictions of the native fold.

Algorithm summary
-----------------
    1. Compute the pairwise RMSD matrix for all decoys.
    2. For each decoy, count the number of neighbours within a
       specified RMSD cutoff, :math:`R_c`.
    3. Select the decoy with the largest neighbour count as a cluster
       centre, and remove its entire cluster (all decoys within
       :math:`R_c` of the centre) from the pool.
    4. Repeat step 3 until all decoys have been assigned or the
       remaining pool falls below a minimum cluster size.
    5. (Optional) Sub-cluster the largest cluster by splitting at
       a tighter cutoff to refine the representative structure.

References
----------
    Zhang, Y. and Skolnick, J. (2004) 'SPICKER: a clustering approach
        to identify near-native protein folds', *Journal of
        Computational Chemistry*, 25(6), pp. 865--871.
        doi:10.1002/jcc.20011.
    Shortle, D. (2002) 'Comets of the protein folding universe',
        *Structure*, 10(1), pp. 2--4.  doi:10.1016/S0969-2126(01)00702-5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from benchmark.casp_scorer import kabsch_superimpose


# ═══════════════════════════════════════════════════════════════════════
# Pairwise RMSD matrix
# ═══════════════════════════════════════════════════════════════════════


def compute_pairwise_rmsd(
    decoy_coords: list[np.ndarray],
    *,
    n_jobs: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute the pairwise RMSD matrix for a set of decoy structures.

    Each element :math:`M_{ij}` is the RMSD (after optimal Kabsch
    superposition) between decoy *i* and decoy *j*.

    Parameters
    ----------
    decoy_coords : list of ndarray
        List of coordinate arrays, each ``(N, 3)`` and ``float64``.
        All decoys must have the same *N*.
    n_jobs : int, optional
        Ignored (reserved for future parallelisation).  Present for API
        compatibility.
    verbose : bool, optional
        If ``True`` (default), print progress information.

    Returns
    -------
    rmsd_matrix : ndarray, shape (M, M)
        Symmetric pairwise RMSD matrix, where *M* is the number of
        decoys.

    Raises
    ------
    ValueError
        If fewer than two decoys are provided.
    ValueError
        If the decoys do not all have the same number of residues.
    """
    M = len(decoy_coords)
    if M < 2:
        raise ValueError(
            f"At least two decoys are required for clustering; "
            f"got {M}."
        )

    N = decoy_coords[0].shape[0]
    for idx, coords in enumerate(decoy_coords[1:], start=1):
        if coords.shape != (N, 3):
            raise ValueError(
                f"Decoy {idx} has shape {coords.shape}; expected "
                f"({N}, 3) from the first decoy."
            )

    rmsd_matrix = np.zeros((M, M), dtype=np.float64)

    if verbose:
        total_pairs = M * (M - 1) // 2
        print(
            f"[SPICKER] Computing {total_pairs} pairwise RMSDs "
            f"for {M} decoys ({N} residues each)..."
        )

    for i in range(M):
        for j in range(i + 1, M):
            _, _, rmsd_val = kabsch_superimpose(
                decoy_coords[i], decoy_coords[j]
            )
            rmsd_matrix[i, j] = rmsd_val
            rmsd_matrix[j, i] = rmsd_val

    if verbose:
        print(f"[SPICKER] Pairwise RMSD matrix computed.")

    return rmsd_matrix


# ═══════════════════════════════════════════════════════════════════════
# SPICKER cluster data structures
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Cluster:
    """
    A single cluster produced by the SPICKER algorithm.

    Attributes
    ----------
    centre_idx : int
        Index (into the original decoy list) of the cluster centre.
    member_indices : list of int
        Indices of all decoys assigned to this cluster.
    size : int
        Number of decoys in this cluster (convenience field).
    radius : float
        Structural radius of the cluster: the mean RMSD from the
        centre to all members (:math:`\\text{\\AA}`).
    centre_coords : ndarray, shape (N, 3), optional
        Coordinates of the cluster-centre decoy.  Populated when
        the caller provides the original coordinates via
        ``store_centres``.
    """

    centre_idx: int
    member_indices: list[int] = field(default_factory=list)
    size: int = 0
    radius: float = 0.0
    centre_coords: Optional[np.ndarray] = None


@dataclass
class ClusteringResult:
    """
    The complete SPICKER clustering output.

    Attributes
    ----------
    clusters : list of Cluster
        Detected clusters, sorted by size descending (largest first).
    rmsd_matrix : ndarray, shape (M, M)
        The pairwise RMSD matrix used for clustering.
    parameters : dict
        The clustering parameters used.
    unassigned : list of int
        Indices of decoys that were not assigned to any cluster.
    """

    clusters: list[Cluster] = field(default_factory=list)
    rmsd_matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    parameters: dict = field(default_factory=dict)
    unassigned: list[int] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# SPICKER clustering algorithm
# ═══════════════════════════════════════════════════════════════════════


def spicker_cluster(
    decoy_coords: list[np.ndarray],
    *,
    rmsd_cutoff: float = 2.0,
    min_cluster_size: int = 10,
    max_clusters: int = 5,
    compute_matrix: bool = True,
    rmsd_matrix: Optional[np.ndarray] = None,
    store_centres: bool = True,
    verbose: bool = True,
) -> ClusteringResult:
    """
    Cluster decoy structures using the SPICKER algorithm.

    Parameters
    ----------
    decoy_coords : list of ndarray
        List of coordinate arrays, each ``(N, 3)``.  All decoys must
        share the same residue count *N*.
    rmsd_cutoff : float, optional
        RMSD cutoff in :math:`\\text{\\AA}` for neighbour counting
        (default: ``2.0``).  Two decoys are considered neighbours
        if their pairwise RMSD falls below this value.
    min_cluster_size : int, optional
        Minimum number of members required to form a valid cluster
        (default: ``10``).  Remaining decoys after cluster extraction
        that fall below this threshold are marked as unassigned.
    max_clusters : int, optional
        Maximum number of clusters to extract (default: ``5``).
    compute_matrix : bool, optional
        If ``True`` (default), compute the pairwise RMSD matrix from
        *decoy_coords*.  If ``False``, *rmsd_matrix* must be provided.
    rmsd_matrix : ndarray, shape (M, M), optional
        Pre-computed pairwise RMSD matrix.  Required when
        ``compute_matrix=False``.
    store_centres : bool, optional
        If ``True`` (default), store the coordinate arrays of cluster
        centres in the returned ``Cluster`` objects.
    verbose : bool, optional
        If ``True`` (default), print progress information.

    Returns
    -------
    ClusteringResult
        Detected clusters, the RMSD matrix, parameters, and any
        unassigned decoy indices.

    Notes
    -----
    The SPICKER algorithm, as originally described by Zhang and
    Skolnick (2004), uses a multi-step procedure:

        1. A ``greedy'' selection picks the decoy with the largest
           neighbour count as the first cluster centre.
        2. That cluster's members are removed from the pool.
        3. The process repeats until the pool is exhausted or the
           next cluster would fall below *min_cluster_size*.

    This implementation follows that protocol.

    References
    ----------
    Zhang, Y. and Skolnick, J. (2004) *Journal of Computational
    Chemistry*, 25(6), pp. 865--871.  doi:10.1002/jcc.20011.
    """
    M = len(decoy_coords)
    if M < 2:
        raise ValueError(
            f"At least two decoys are required for clustering; "
            f"got {M}."
        )

    # --- RMSD matrix ---
    if compute_matrix:
        rmsd_mat = compute_pairwise_rmsd(decoy_coords, verbose=verbose)
    elif rmsd_matrix is not None:
        rmsd_mat = rmsd_matrix
        if rmsd_mat.shape != (M, M):
            raise ValueError(
                f"Provided rmsd_matrix has shape {rmsd_mat.shape}; "
                f"expected ({M}, {M})."
            )
    else:
        raise ValueError(
            "Either compute_matrix=True or rmsd_matrix must be provided."
        )

    # --- Greedy SPICKER clustering ---
    available = set(range(M))
    clusters: list[Cluster] = []
    unassigned: list[int] = []

    if verbose:
        print(
            f"[SPICKER] Clustering {M} decoys at "
            f"R_c = {rmsd_cutoff:.1f} Å, "
            f"min_cluster = {min_cluster_size}, "
            f"max_clusters = {max_clusters}..."
        )

    while available and len(clusters) < max_clusters:
        available_list = list(available)
        best_centre = -1
        best_count = 0
        best_members: list[int] = []

        for idx in available_list:
            # Count neighbours among remaining decoys.
            neighbours = [
                j for j in available_list
                if j != idx and rmsd_mat[idx, j] < rmsd_cutoff
            ]
            count = len(neighbours)

            if count > best_count:
                best_count = count
                best_centre = idx
                best_members = neighbours

        if best_centre == -1:
            # No further clustering possible.
            break

        cluster_members = [best_centre] + best_members
        cluster_size = len(cluster_members)

        # Skip if the cluster is below the minimum size.
        if cluster_size < min_cluster_size:
            break

        # Compute cluster radius: mean RMSD from centre to its members.
        radius_values = [rmsd_mat[best_centre, m] for m in best_members]
        radius = float(np.mean(radius_values)) if radius_values else 0.0

        centre_coords = (
            decoy_coords[best_centre].copy() if store_centres else None
        )

        cluster = Cluster(
            centre_idx=best_centre,
            member_indices=cluster_members,
            size=cluster_size,
            radius=radius,
            centre_coords=centre_coords,
        )
        clusters.append(cluster)

        # Remove cluster members from the pool.
        available -= set(cluster_members)

        if verbose:
            print(
                f"  Cluster {len(clusters)}: centre decoy "
                f"#{best_centre}, {cluster_size} members, "
                f"radius = {radius:.3f} Å"
            )

    # Remaining decoys are unassigned.
    unassigned = sorted(available)

    if verbose:
        print(
            f"[SPICKER] Done: {len(clusters)} clusters, "
            f"{len(unassigned)} decoys unassigned."
        )

    # Sort clusters by size descending.
    clusters.sort(key=lambda c: c.size, reverse=True)

    return ClusteringResult(
        clusters=clusters,
        rmsd_matrix=rmsd_mat,
        parameters={
            'rmsd_cutoff': rmsd_cutoff,
            'min_cluster_size': min_cluster_size,
            'max_clusters': max_clusters,
            'n_decoys': M,
        },
        unassigned=unassigned,
    )


# ═══════════════════════════════════════════════════════════════════════
# Utility: extract cluster representatives
# ═══════════════════════════════════════════════════════════════════════


def get_representatives(
    result: ClusteringResult,
    decoy_coords: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Extract the coordinate arrays of the cluster-centre structures.

    Parameters
    ----------
    result : ClusteringResult
        Output from :func:`spicker_cluster`.
    decoy_coords : list of ndarray
        The original decoy coordinate list passed to
        :func:`spicker_cluster`.

    Returns
    -------
    centres : list of ndarray
        Coordinates of each cluster centre.
    """
    return [
        decoy_coords[cl.centre_idx].copy() for cl in result.clusters
    ]


# ═══════════════════════════════════════════════════════════════════════
# Self-test / validation
# ═══════════════════════════════════════════════════════════════════════


def _self_test() -> None:
    """Run a basic consistency check on SPICKER clustering."""
    np.random.seed(42)
    L = 50  # residues
    M = 60  # decoys

    # Generate a native-like cluster (30 decoys near a reference).
    ref_coords = np.random.randn(L, 3).astype(np.float64) * 10.0
    cluster1 = [
        ref_coords + np.random.randn(L, 3).astype(np.float64) * 0.5
        for _ in range(30)
    ]

    # Generate a second cluster (20 decoys near a different reference).
    ref_coords2 = np.random.randn(L, 3).astype(np.float64) * 10.0 + 20.0
    cluster2 = [
        ref_coords2 + np.random.randn(L, 3).astype(np.float64) * 0.5
        for _ in range(20)
    ]

    # Remaining 10 decoys as random noise.
    noise = [
        np.random.randn(L, 3).astype(np.float64) * 15.0
        for _ in range(10)
    ]

    all_decoys = cluster1 + cluster2 + noise

    result = spicker_cluster(
        all_decoys,
        rmsd_cutoff=2.0,
        min_cluster_size=10,
        max_clusters=5,
        verbose=True,
    )

    assert len(result.clusters) >= 2, (
        f"Expected at least 2 clusters, got {len(result.clusters)}."
    )
    assert result.clusters[0].size >= 25, (
        f"Largest cluster too small: {result.clusters[0].size}."
    )
    assert result.clusters[1].size >= 15, (
        f"Second cluster too small: {result.clusters[1].size}."
    )

    print(f"[benchmark/decoy_cluster] Self-test passed")
    print(f"  Largest cluster:  {result.clusters[0].size} members, "
          f"radius = {result.clusters[0].radius:.3f} Å")
    print(f"  Second cluster:  {result.clusters[1].size} members, "
          f"radius = {result.clusters[1].radius:.3f} Å")
    print(f"  Unassigned: {len(result.unassigned)} decoys")


if __name__ == "__main__":
    _self_test()
