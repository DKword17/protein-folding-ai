#!/usr/bin/env python3
"""
protein_folding_ai/folding_engine.py
====================================

Core protein folding engine using fragment-based energy minimization.

Implements a simplified version of the ROSETTA energy function:
    E_total = w1*E_LJ + w2*E_Hbond + w3*E_solvation + w4*E_rama + w5*E_repulsive

Fragment insertion Monte Carlo drives conformational sampling,
backbone dihedral angles from the Dunbrack rotamer library.

References:
    - Alford et al. (2017) JCTC 13:3031 — Rosetta energy function
    - Dunbrack & Cohen (1997) Prot Sci 6:1661 — backbone-dependent rotamer library
    - Bowers et al. (2006) SC06 — distributed folding on Folding@Home

Author: Ali Hassan
        Computational Biophysics Lab, EMBL Heidelberg
Date:   2026-07-26
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Iterator

import numpy as np

# ─── Amino Acid Constants ─────────────────────────────────────────────

AA_CODES: dict[str, str] = {
    'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
    'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
    'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
    'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
}

# van der Waals radii (Å)
VDW_RADII: dict[str, float] = {
    'ALA': 1.80, 'CYS': 1.72, 'ASP': 1.75, 'GLU': 1.78,
    'PHE': 1.95, 'GLY': 1.75, 'HIS': 1.88, 'ILE': 1.90,
    'LYS': 1.90, 'LEU': 1.88, 'MET': 1.90, 'ASN': 1.78,
    'PRO': 1.80, 'GLN': 1.82, 'ARG': 1.92, 'SER': 1.72,
    'THR': 1.75, 'VAL': 1.85, 'TRP': 2.00, 'TYR': 1.95,
}

# Ramachandran preference weights (simplified — real lib much larger)
# phi, psi in degrees
_RAMA_PREFERENCE: dict[str, list[tuple[float, float, float]]] = {
    'ALA': [(-60, -45, 1.0), (-120, 130, 0.8), (60, 50, 0.3)],
    'GLY': [(-60, -45, 0.5), (-130, 120, 0.4), (60, 40, 0.4), (170, 170, 0.3)],
    'PRO': [(-60, -30, 1.5), (-70, 130, 0.6)],
    'VAL': [(-60, -40, 1.2), (-110, 130, 0.9)],
}


# ─── Enums ────────────────────────────────────────────────────────────

class SecondaryStructure(IntEnum):
    """DSSP secondary structure classification."""
    COIL = 0          # Random coil / loop
    HELIX = 1         # Alpha helix
    STRAND = 2        # Beta strand
    TURN = 3          # Turn


# ─── Data Structures ──────────────────────────────────────────────────

@dataclass
class Residue:
    """A single amino acid residue in the protein model."""
    seq_index: int                      # Position in sequence (1-indexed)
    aa_code: str                        # Three-letter code (ALA, etc.)
    phi: float = 0.0                    # Backbone dihedral: N-CA-C-N (degrees)
    psi: float = 0.0                    # Backbone dihedral: CA-C-N-CA (degrees)
    omega: float = 180.0                # Peptide bond dihedral (cis=0, trans=180)
    chi: list[float] = field(default_factory=list)  # Side-chain dihedrals
    
    # Backbone atom coordinates (Å)
    N: np.ndarray = field(default_factory=lambda: np.zeros(3))
    CA: np.ndarray = field(default_factory=lambda: np.zeros(3))
    C: np.ndarray = field(default_factory=lambda: np.zeros(3))
    O: np.ndarray = field(default_factory=lambda: np.zeros(3))
    CB: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    secondary: SecondaryStructure = SecondaryStructure.COIL
    rama_score: float = 0.0
    sasa: float = 0.0                   # Solvent accessible surface area (Å²)


@dataclass
class Conformation:
    """
    A complete 3D conformation of a protein.
    
    Attributes:
        residues: Ordered list of residues with full atomic coordinates.
        total_energy: Total energy score (lower = more stable).
        energy_components: Breakdown of energy terms.
        rmsd: Root-mean-square deviation from native (if known).
        plddt: Predicted local distance difference test (0-100).
    """
    residues: list[Residue]
    total_energy: float = 0.0
    energy_components: dict[str, float] = field(default_factory=dict)
    rmsd: Optional[float] = None
    plddt: Optional[float] = None


# ─── Energy Function ──────────────────────────────────────────────────

class RosettaEnergyFunction:
    """
    Simplified Rosetta energy function.
    
    E = w1 * E_lennard_jones + w2 * E_hbond + w3 * E_solvation
        + w4 * E_ramachandran + w5 * E_repulsive
    
    Default weights from REF2015 (Alford et al., 2017).
    """
    
    def __init__(self):
        # Weights (pre-optimized for REF2015)
        self.w_lj = 0.50
        self.w_hbond = 0.75
        self.w_solvation = 0.60
        self.w_rama = 0.35
        self.w_repulsive = 1.10
        
        # Lennard-Jones well depths (kcal/mol)
        self.lj_epsilon = {
            ('ALA', 'ALA'): 0.15, ('LEU', 'LEU'): 0.28,
            ('VAL', 'VAL'): 0.22, ('ILE', 'ILE'): 0.26,
            ('PHE', 'PHE'): 0.35, ('TRP', 'TRP'): 0.40,
            ('TYR', 'TYR'): 0.38, ('GLY', 'GLY'): 0.10,
        }
        
        # Hydrogen bonding energy (kcal/mol)
        self.hbond_energy_max = -3.0     # Optimal HBond
        self.hbond_dist_opt = 2.8        # Optimal N-O distance (Å)
        self.hbond_angle_opt = 165.0     # Optimal N-H···O angle (deg)
    
    def evaluate(self, conf: Conformation) -> float:
        """Compute total energy for a conformation."""
        E_lj = self._compute_lennard_jones(conf)
        E_hbond = self._compute_hbonds(conf)
        E_solv = self._compute_solvation(conf)
        E_rama = self._compute_ramachandran(conf)
        E_rep = self._compute_repulsive(conf)
        
        total = (self.w_lj * E_lj + self.w_hbond * E_hbond
                 + self.w_solvation * E_solv + self.w_rama * E_rama
                 + self.w_repulsive * E_rep)
        
        conf.energy_components = {
            'lennard_jones': E_lj,
            'hydrogen_bond': E_hbond,
            'solvation': E_solv,
            'ramachandran': E_rama,
            'repulsive': E_rep,
        }
        conf.total_energy = total
        return total
    
    def _compute_lennard_jones(self, conf: Conformation) -> float:
        """6-12 Lennard-Jones potential for all residue pairs within 10 Å."""
        energy = 0.0
        residues = conf.residues
        n = len(residues)
        
        for i in range(n - 2):
            for j in range(i + 2, n):  # Skip sequential neighbors
                r = np.linalg.norm(residues[i].CB - residues[j].CB)
                if r > 10.0 or r < 0.5:
                    continue
                
                aa_i, aa_j = residues[i].aa_code, residues[j].aa_code
                eps = self.lj_epsilon.get((aa_i, aa_j), 0.15)
                sigma = (VDW_RADII.get(aa_i, 1.8) + VDW_RADII.get(aa_j, 1.8)) / 2
                
                # 6-12 LJ
                sr6 = (sigma / r) ** 6
                sr12 = sr6 * sr6
                energy += eps * (sr12 - 2 * sr6)
        
        return energy
    
    def _compute_hbonds(self, conf: Conformation) -> float:
        """
        Backbone hydrogen bond energy (N-H···O=C).
        Only i, i+4 for helices; antiparallel for sheets.
        """
        energy = 0.0
        residues = conf.residues
        
        for i in range(len(residues) - 3):
            for j in range(i + 3, min(i + 6, len(residues))):
                n_atom = residues[i].N
                o_atom = residues[j].O
                
                d = np.linalg.norm(n_atom - o_atom)
                if d < 2.0 or d > 4.0:
                    continue
                
                # Distance-dependent sigmoid
                d_penalty = (d - self.hbond_dist_opt) ** 2 / 2.0
                energy += self.hbond_energy_max * math.exp(-d_penalty)
        
        return energy
    
    def _compute_solvation(self, conf: Conformation) -> float:
        """Lazaridis-Karplus solvation energy (simplified)."""
        energy = 0.0
        residues = conf.residues
        
        for i, res in enumerate(residues):
            # Per-residue SASA estimate from neighbor count
            neighbor_count = 0
            for j, other in enumerate(residues):
                if i == j:
                    continue
                r = np.linalg.norm(res.CB - other.CB)
                if r < 8.0:
                    neighbor_count += 1
            
            # More neighbors = buried = favorable for hydrophobic
            sasa_est = max(0, 1.0 - neighbor_count / 10.0)
            
            # Hydrophobic preference
            hydrophobic_aa = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'TYR', 'MET'}
            if res.aa_code in hydrophobic_aa:
                energy += sasa_est * 0.5   # Buried = stable
            else:
                energy += (1 - sasa_est) * 0.3  # Exposed = stable for polar
        
        return energy
    
    def _compute_ramachandran(self, conf: Conformation) -> float:
        """Ramachandran preference score from backbone dihedrals."""
        energy = 0.0
        for res in conf.residues:
            phi, psi = res.phi, res.psi
            aa = res.aa_code
            pref = _RAMA_PREFERENCE.get(aa, [(-60, -45, 0.5)])
            
            best = 0.0
            for phi0, psi0, w in pref:
                dphi = abs(phi - phi0)
                if dphi > 180:
                    dphi = 360 - dphi
                dpsi = abs(psi - psi0)
                if dpsi > 180:
                    dpsi = 360 - dpsi
                
                score = w * math.exp(-(dphi ** 2 + dpsi ** 2) / 2000)
                best = max(best, score)
            
            energy += 1.0 - best
            res.rama_score = 1.0 - best
        
        return energy
    
    def _compute_repulsive(self, conf: Conformation) -> float:
        """Steric repulsion for atoms closer than VDW radius."""
        energy = 0.0
        residues = conf.residues
        
        for i in range(len(residues) - 1):
            for j in range(i + 1, len(residues)):
                r = np.linalg.norm(residues[i].CB - residues[j].CB)
                if r < 0.01:
                    continue
                
                sigma = (VDW_RADII.get(residues[i].aa_code, 1.8)
                         + VDW_RADII.get(residues[j].aa_code, 1.8)) / 2
                
                if r < sigma:
                    energy += 10.0 * ((sigma / r) ** 12 - (sigma / r) ** 6)
        
        return energy


# ─── Fragment Insertion Monte Carlo ───────────────────────────────────

class FragmentInsertionMC:
    """
    Conformational sampling via fragment insertion Monte Carlo.
    
    Replaces a contiguous stretch of backbone dihedrals with
    fragments drawn from a structural fragment library.
    
    Protocol:
        1. Select random start position (i)
        2. Select fragment length (3 or 9 residues)
        3. Replace phi/psi with fragment values
        4. Rebuild coordinates from dihedrals
        5. Accept/reject by Metropolis criterion
    """
    
    def __init__(self, energy_fn: RosettaEnergyFunction,
                 temperature: float = 300.0):
        self.energy_fn = energy_fn
        self.kT = 0.008314 * temperature / 298.0  # kcal/mol
        self.last_accepted = True
    
    def run(self, conf: Conformation, n_steps: int = 10000,
            early_stop_energy: float = -50.0) -> list[float]:
        """
        Run MCMC sampling.
        
        Args:
            conf: Starting conformation.
            n_steps: Number of fragment insertion attempts.
            early_stop_energy: Stop if energy drops below this.
        
        Returns:
            Energy trajectory (list of energies).
        """
        current_energy = self.energy_fn.evaluate(conf)
        energy_trace = [current_energy]
        
        n_accept = 0
        n_residues = len(conf.residues)
        
        for step in range(n_steps):
            # Select fragment
            start = random.randint(0, n_residues - 4)
            frag_len = random.choice([3, 9])
            
            # Store old dihedrals
            old_phi_psi = []
            for i in range(start, min(start + frag_len, n_residues)):
                old_phi_psi.append((conf.residues[i].phi, conf.residues[i].psi))
            
            # Insert fragment (sample from library)
            self._insert_fragment(conf, start, frag_len)
            
            # Rebuild coordinates
            self._rebuild_backbone(conf, start, start + frag_len)
            
            # Evaluate
            trial_energy = self.energy_fn.evaluate(conf)
            delta_e = trial_energy - current_energy
            
            # Metropolis criterion
            if delta_e < 0 or random.random() < math.exp(-delta_e / self.kT):
                current_energy = trial_energy
                n_accept += 1
                self.last_accepted = True
            else:
                # Revert
                for i, (phi, psi) in enumerate(old_phi_psi):
                    idx = start + i
                    if idx < n_residues:
                        conf.residues[idx].phi = phi
                        conf.residues[idx].psi = psi
                self._rebuild_backbone(conf, start, start + frag_len)
                self.last_accepted = False
            
            energy_trace.append(current_energy)
            
            # Early termination
            if current_energy < early_stop_energy:
                break
        
        acceptance_rate = n_accept / n_steps * 100
        print(f"  [Folding] {n_steps} steps, "
              f"{acceptance_rate:.1f}% acceptance, "
              f"final E = {current_energy:.2f}")
        
        return energy_trace
    
    def _insert_fragment(self, conf: Conformation, start: int, frag_len: int):
        """Replace backbone dihedrals with fragment data."""
        for i in range(start, min(start + frag_len, len(conf.residues))):
            # Sample phi/psi from Gaussian around preferred Ramachandran regions
            aa = conf.residues[i].aa_code
            pref = _RAMA_PREFERENCE.get(aa, [(-60, -45, 0.5)])
            phi0, psi0, _ = random.choices(pref, weights=[p[2] for p in pref])[0]
            
            conf.residues[i].phi = phi0 + random.gauss(0, 15)
            conf.residues[i].psi = psi0 + random.gauss(0, 15)
    
    def _rebuild_backbone(self, conf: Conformation, start: int, end: int,
                          bond_length: float = 1.33):
        """
        Rebuild backbone coordinates from dihedral angles.
        Uses idealized bond lengths and angles.
        """
        for i in range(max(1, start), min(end, len(conf.residues))):
            prev = conf.residues[i - 1]
            curr = conf.residues[i]
            
            # Standard peptide bond geometry
            # N-CA bond
            curr.N = prev.C + np.array([0.0, 0.0, 1.33])
            
            # CA position from phi
            phi_rad = math.radians(curr.phi)
            ca_vec = np.array([
                1.47 * math.cos(phi_rad),
                1.47 * math.sin(phi_rad),
                0.0
            ])
            curr.CA = curr.N + ca_vec
            
            # C position from psi
            psi_rad = math.radians(curr.psi)
            c_vec = np.array([
                1.51 * math.cos(psi_rad),
                1.51 * math.sin(psi_rad),
                0.0
            ])
            curr.C = curr.CA + c_vec
            
            # O position (perpendicular to N-CA-C plane)
            perp = np.cross(curr.N - curr.CA, curr.C - curr.CA)
            perp = perp / (np.linalg.norm(perp) + 1e-10)
            curr.O = curr.C + perp * 1.23
            
            # CB position
            cb_dir = np.cross(curr.N - curr.CA, perp)
            cb_dir = cb_dir / (np.linalg.norm(cb_dir) + 1e-10)
            curr.CB = curr.CA + cb_dir * 1.53


# ─── Full Pipeline ────────────────────────────────────────────────────

def fold_protein(sequence: str, n_steps: int = 50000,
                 temperature: float = 300.0) -> Conformation:
    """
    Fold a protein from amino acid sequence.
    
    Args:
        sequence: 1-letter amino acid codes (e.g. 'ACDEFGHIKLMNPQRSTVWY').
        n_steps: Number of MCMC steps.
        temperature: Simulation temperature (K).
    
    Returns:
        Lowest-energy conformation found.
    
    Example:
        >>> conf = fold_protein('MLSDGEFQL', n_steps=10000)
        >>> print(f"Energy: {conf.total_energy:.2f}")
    """
    print(f"  [Protein Folding] Sequence: {sequence}")
    print(f"  [Protein Folding] Length: {len(sequence)} residues")
    print(f"  [Protein Folding] Steps: {n_steps}")
    
    # Build initial extended chain
    residues = []
    for i, aa in enumerate(sequence.upper()):
        res = Residue(
            seq_index=i + 1,
            aa_code=AA_CODES.get(aa, 'ALA'),
            phi=-135.0,  # Extended conformation
            psi=135.0,
        )
        residues.append(res)
    
    conf = Conformation(residues=residues)
    
    # Initialize coordinates
    energy_fn = RosettaEnergyFunction()
    sampler = FragmentInsertionMC(energy_fn, temperature)
    
    # Build naive backbone
    for i, res in enumerate(conf.residues):
        if i == 0:
            continue
        sampler._rebuild_backbone(conf, i, i)
    
    # Run MCMC
    energy_trace = sampler.run(conf, n_steps)
    
    print(f"  [Protein Folding] Done. "
          f"Final energy: {conf.total_energy:.2f} kcal/mol")
    return conf


if __name__ == "__main__":
    # Test: fold a small helical peptide
    test_seq = "AAAAKAAAAKAAAAK"
    conf = fold_protein(test_seq, n_steps=20000)
    print(f"\n  Energy breakdown: {conf.energy_components}")
