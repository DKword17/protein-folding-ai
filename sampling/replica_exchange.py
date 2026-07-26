#!/usr/bin/env python3
"""
sampling/replica_exchange.py
============================

Échange de répliques (Replica Exchange Molecular Dynamics — REMD).

Algorithme de Sugita & Okamoto (1999) : on simule M copies (répliques)
d'un même système à des températures différentes, et on tente
périodiquement d'échanger leurs conformations.

    P_acceptation(i ↔ j) = min{1, exp[(βᵢ − βⱼ)(Eᵢ − Eⱼ)]}

où β = 1/k_B T.

Deux variantes sont implémentées :

    1. **t-REMD** (classique) : chaque réplique diffère par sa température.
    2. **H-REMD** (Hamiltonien) : chaque réplique diffère par un facteur
       d'échelle λ agissant sur certains termes de l'énergie. Le critère
       d'échange devient :

    P_acc(i ↔ j) = min{1, exp[β(λᵢ − λⱼ)(E_repulsive_i − E_repulsive_j)]}

L'échange permet aux répliques à haute température (ou λ plus faible)
de franchir des barrières, tandis que les conditions natives explorent
les minima locaux. La chaîne globale satisfait le bilan détaillé.

Variables :
    Température          → T_i (K)
    β                    → beta = 1/k_B T
    Réplique             → copie du système (conformation + énergie)
    Probabilité          → probabilité d'acceptation d'un échange
    Poids (H-REMD)       → λ_i — facteur d'échelle Hamiltonien

Références :
    - Sugita & Okamoto (1999) Chem. Phys. Lett. 314:141
    - Frenkel & Smit, Understanding Molecular Simulation, 2e éd., ch. 12
    - Fukunishi et al. (2002) J. Chem. Phys. 116:9058 — H-REMD
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, "..")
from folding_engine import (
    AA_CODES,
    Conformation,
    FragmentInsertionMC,
    Residue,
    RosettaEnergyFunction,
)


# ─── Constante physique ───────────────────────────────────────────────

CONSTANTE_BOLTZMANN: float = 0.001987  # kcal · mol⁻¹ · K⁻¹


# ─── Utilitaires ──────────────────────────────────────────────────────

def températures_géométriques(
    t_min: float,
    t_max: float,
    n: int,
) -> list[float]:
    """
    Génère *n* températures en progression géométrique entre t_min et t_max.

    Le ratio constant garantit des taux d'acceptation uniformes entre
    paires adjacentes (Patriksson & van der Spoel, 2008).

        T_i = T_min · (T_max / T_min)^{i / (n - 1)}
    """
    if n < 2:
        raise ValueError("Il faut au moins 2 températures.")
    ratio = (t_max / t_min) ** (1.0 / (n - 1))
    return [t_min * (ratio ** i) for i in range(n)]


# ─── Structure de données pour une réplique ───────────────────────────

@dataclass
class Réplique:
    """
    Une réplique du système dans l'ensemble (T, λ).

    Attributs :
        température    : température de simulation (K)
        conformation   : conformation courante
        énergie        : E_total courante (kcal/mol)
        énergie_repulsive : terme répulsif seul (utile pour H-REMD)
        beta           : 1 / (k_B · T)
        poids          : λ — facteur d'échelle Hamiltonien (défaut 1.0)
        acceptations   : nombre d'échanges acceptés (cette réplique)
        tentatives     : nombre de tentatives d'échange
    """
    température: float
    conformation: Conformation
    énergie: float = 0.0
    énergie_repulsive: float = 0.0
    beta: float = field(init=False)
    poids: float = 1.0
    acceptations: int = 0
    tentatives: int = 0

    def __post_init__(self):
        self.beta = 1.0 / (CONSTANTE_BOLTZMANN * self.température)

    def probabilité_échange(self, autre: Réplique) -> float:
        """
        Probabilité d'acceptation pour un échange entre self et autre.

        t-REMD : P = exp[(β_self − β_autre)(E_self − E_autre)]
        H-REMD : P = exp[β · (λ_self − λ_autre)(E_rep_self − E_rep_autre)]

        Si les deux répliques sont à la même température, on utilise
        la formule H-REMD ; sinon, la formule t-REMD classique.
        """
        if abs(self.température - autre.température) > 1e-6:
            # t-REMD : échange de température
            return math.exp((self.beta - autre.beta) * (self.énergie - autre.énergie))
        else:
            # H-REMD : échange de facteur d'échelle
            Δλ = self.poids - autre.poids
            ΔE_rep = self.énergie_repulsive - autre.énergie_repulsive
            return math.exp(self.beta * Δλ * ΔE_rep)

    def vers_dictionnaire(self) -> dict:
        """Sérialise la réplique (sauf la conformation, trop lourde)."""
        return {
            "température": self.température,
            "énergie": self.énergie,
            "énergie_repulsive": self.énergie_repulsive,
            "beta": self.beta,
            "poids": self.poids,
            "acceptations": self.acceptations,
            "tentatives": self.tentatives,
        }


# ─── Classe principale d'échange de répliques ─────────────────────────

class ÉchangeDeRépliques:
    """
    Échantillonnage par échange de répliques (REMD).

    Paramètres :
        températures          : liste des températures (K) — une par réplique
        fonction_énergie      : instance de RosettaEnergyFunction
        pas_mc_entre_échanges : nombre de pas MC entre deux tentatives
        longueur_fragment     : longueur des fragments pour l'insertion MC
        poids_hamiltoniens    : liste des λ_i pour H-REMD (défaut : tous 1.0)
    """

    def __init__(
        self,
        températures: list[float],
        fonction_énergie: Optional[RosettaEnergyFunction] = None,
        pas_mc_entre_échanges: int = 100,
        longueur_fragment: int = 3,
        poids_hamiltoniens: Optional[list[float]] = None,
    ):
        if len(températures) < 2:
            raise ValueError("Il faut au moins deux températures pour REMD.")

        self.températures = sorted(températures)
        self.fonction_énergie = fonction_énergie or RosettaEnergyFunction()
        self.pas_mc_entre_échanges = pas_mc_entre_échanges
        self.longueur_fragment = longueur_fragment

        if poids_hamiltoniens is not None:
            if len(poids_hamiltoniens) != len(températures):
                raise ValueError(
                    "poids_hamiltoniens doit avoir la même longueur que températures."
                )
            self.poids_hamiltoniens = poids_hamiltoniens
        else:
            self.poids_hamiltoniens = [1.0] * len(températures)

        # Stockage
        self.historique_énergies: list[list[float]] = []
        self.historique_températures: list[list[float]] = []
        self.taux_acceptation: list[float] = []
        self.nombre_échanges_par_paire: list[list[int]] = []
        self.répliques: list[Réplique] = []
        self._séquence: str = ""

    # ── Initialisation ──────────────────────────────────────────────

    def initialiser(self, séquence: str) -> list[Réplique]:
        """Crée une conformation initiale pour chaque réplique."""
        self._séquence = séquence
        self.répliques.clear()
        self.historique_énergies.clear()
        self.historique_températures.clear()

        for T, λ in zip(self.températures, self.poids_hamiltoniens):
            conf = self._construire_chaîne_étendue(séquence)
            énergie = self._évaluer_avec_poids(conf, λ)
            énergie_rep = self.fonction_énergie._compute_repulsive(conf)
            repl = Réplique(
                température=T,
                conformation=conf,
                énergie=énergie,
                énergie_repulsive=énergie_rep,
                poids=λ,
            )
            self.répliques.append(repl)
            self.historique_énergies.append([énergie])
            self.historique_températures.append([T])

        n = len(self.températures)
        self.nombre_échanges_par_paire = [[0] * n for _ in range(n)]
        self.taux_acceptation = [0.0] * (n - 1)
        return self.répliques

    # ── Construction d'une chaîne étendue ──────────────────────────

    @staticmethod
    def _construire_chaîne_étendue(séquence: str) -> Conformation:
        """Construit une conformation en chaîne étendue (phi=−135, psi=135)."""
        résidus = []
        for i, aa in enumerate(séquence.upper()):
            résidus.append(Residue(
                seq_index=i + 1,
                aa_code=AA_CODES.get(aa, "ALA"),
                phi=-135.0,
                psi=135.0,
            ))
        conf = Conformation(residues=résidus)
        for i in range(1, len(résidus)):
            ÉchangeDeRépliques._reconstruire_coordonnées(conf, i, i + 1)
        return conf

    # ── Évaluation avec poids Hamiltonien ──────────────────────────

    def _évaluer_avec_poids(self, conf: Conformation, λ: float) -> float:
        """
        Évalue l'énergie totale avec un facteur d'échelle λ sur le
        terme répulsif (H-REMD).

        E_tot(λ) = E_LJ + E_HB + E_solv + E_rama + λ · E_rep
        """
        self.fonction_énergie.evaluate(conf)
        composantes = conf.energy_components
        return (
            composantes["lennard_jones"]
            + composantes["hydrogen_bond"]
            + composantes["solvation"]
            + composantes["ramachandran"]
            + λ * composantes["repulsive"]
        )

    # ── Pas MC local sur une réplique ─────────────────────────────

    def _pas_mc_local(self, réplique: Réplique, pas: int = 1):
        """Exécute *pas* tentatives d'insertion de fragment."""
        for _ in range(pas):
            n_res = len(réplique.conformation.residues)
            start = random.randint(0, n_res - 4)
            frag_len = self.longueur_fragment

            anciens_angles = []
            for i in range(start, min(start + frag_len, n_res)):
                r = réplique.conformation.residues[i]
                anciens_angles.append((r.phi, r.psi))
                r.phi += random.gauss(0, 15)
                r.psi += random.gauss(0, 15)

            self._reconstruire_coordonnées(
                réplique.conformation, start, start + frag_len
            )

            nouvelle_énergie = self._évaluer_avec_poids(
                réplique.conformation, réplique.poids
            )
            ΔE = nouvelle_énergie - réplique.énergie

            if ΔE < 0 or random.random() < math.exp(-ΔE * réplique.beta):
                réplique.énergie = nouvelle_énergie
                if abs(réplique.poids - 1.0) > 1e-6:
                    composantes = réplique.conformation.energy_components
                    réplique.énergie_repulsive = composantes.get("repulsive", 0.0)
                else:
                    réplique.énergie_repulsive = (
                        réplique.conformation.energy_components.get("repulsive", 0.0)
                    )
            else:
                for i, (phi, psi) in enumerate(anciens_angles):
                    idx = start + i
                    if idx < n_res:
                        réplique.conformation.residues[idx].phi = phi
                        réplique.conformation.residues[idx].psi = psi
                self._reconstruire_coordonnées(
                    réplique.conformation, start, start + frag_len
                )

    @staticmethod
    def _reconstruire_coordonnées(conf: Conformation, début: int, fin: int):
        """Reconstruit les coordonnées CA/C/O/CB à partir des angles (phi, psi)."""
        for i in range(max(1, début), min(fin, len(conf.residues))):
            prev = conf.residues[i - 1]
            curr = conf.residues[i]

            curr.N = prev.C + np.array([0.0, 0.0, 1.33])

            phi_rad = math.radians(curr.phi)
            curr.CA = curr.N + np.array([
                1.47 * math.cos(phi_rad),
                1.47 * math.sin(phi_rad),
                0.0,
            ])

            psi_rad = math.radians(curr.psi)
            curr.C = curr.CA + np.array([
                1.51 * math.cos(psi_rad),
                1.51 * math.sin(psi_rad),
                0.0,
            ])

            perp = np.cross(curr.N - curr.CA, curr.C - curr.CA)
            norme = np.linalg.norm(perp) + 1e-10
            curr.O = curr.C + (perp / norme) * 1.23

            cb_dir = np.cross(curr.N - curr.CA, perp / norme)
            cb_dir = cb_dir / (np.linalg.norm(cb_dir) + 1e-10)
            curr.CB = curr.CA + cb_dir * 1.53

    # ── Tentative d'échange entre deux répliques adjacentes ────────

    def _tenter_échange(self, i: int, j: int) -> bool:
        """
        Tente l'échange entre les répliques i et j (adjacentes).

        Utilise probabilité_échange() qui sélectionne automatiquement
        t-REMD ou H-REMD selon que les températures diffèrent ou non.
        """
        ri = self.répliques[i]
        rj = self.répliques[j]

        probabilité = min(1.0, ri.probabilité_échange(rj))

        ri.tentatives += 1
        rj.tentatives += 1

        if random.random() < probabilité:
            ri.conformation, rj.conformation = rj.conformation, ri.conformation
            ri.énergie, rj.énergie = rj.énergie, ri.énergie
            ri.énergie_repulsive, rj.énergie_repulsive = (
                rj.énergie_repulsive,
                ri.énergie_repulsive,
            )
            ri.acceptations += 1
            rj.acceptations += 1
            self.nombre_échanges_par_paire[i][j] += 1
            self.nombre_échanges_par_paire[j][i] += 1
            return True

        return False

    # ── Boucle principale REMD ──────────────────────────────────────

    def exécuter(
        self,
        séquence: str,
        itérations: int = 500,
        rapport_intervalle: int = 50,
        fichier_checkpoint: Optional[str] = None,
        intervalle_checkpoint: int = 100,
    ) -> list[Réplique]:
        """
        Boucle principale REMD.

        Pour chaque itération :
            1. Pas MC local sur chaque réplique
            2. Tentative d'échange entre paires adjacentes (alterné)
            3. Enregistrement des énergies

        Paramètres :
            séquence             : séquence d'acides aminés
            itérations           : nombre de cycles
            rapport_intervalle   : pas d'affichage
            fichier_checkpoint   : chemin .json pour sauvegarde/restauration
            intervalle_checkpoint: pas de sauvegarde

        Retourne :
            La liste des répliques après la dernière itération.
        """
        n = len(self.températures)

        # Tentative de restauration
        it_départ = 0
        if fichier_checkpoint and os.path.exists(fichier_checkpoint):
            it_départ = self._restaurer(fichier_checkpoint, séquence)
            if it_départ > 0:
                print(f"  [REMD] Reprise depuis checkpoint (itération {it_départ})")

        if it_départ == 0:
            self.initialiser(séquence)
            print(f"  [REMD] {n} répliques, "
                  f"{self.températures[0]:.1f}–{self.températures[-1]:.1f} K")
            if any(abs(λ - 1.0) > 1e-6 for λ in self.poids_hamiltoniens):
                plages = ", ".join(
                    f"Rép.{i} λ={λ:.2f}"
                    for i, λ in enumerate(self.poids_hamiltoniens)
                )
                print(f"  [REMD] H-REMD actif : {plages}")
            print(f"  [REMD] {itérations} itérations, "
                  f"échange tous les {self.pas_mc_entre_échanges} pas MC")
            print(f"  [REMD] Séquence : {séquence} ({len(séquence)} résidus)")

        t_début = time.time()

        for it in range(it_départ, itérations):
            # 1. Pas MC local
            for réplique in self.répliques:
                self._pas_mc_local(réplique, pas=self.pas_mc_entre_échanges)

            # 2. Échanges adjacents alternés (pair / impair)
            for décalage in (0, 1):
                for i in range(décalage, n - 1, 2):
                    self._tenter_échange(i, i + 1)

            # 3. Enregistrement
            for idx, réplique in enumerate(self.répliques):
                self.historique_énergies[idx].append(réplique.énergie)
                self.historique_températures[idx].append(
                    réplique.température
                )

            # 4. Rapport périodique
            if (it + 1) % rapport_intervalle == 0:
                elapsed = time.time() - t_début
                self._afficher_rapport(it + 1, elapsed)

            # 5. Checkpoint
            if fichier_checkpoint and (it + 1) % intervalle_checkpoint == 0:
                self._sauvegarder(fichier_checkpoint, it + 1)

        elapsed = time.time() - t_début
        self._afficher_résumé(elapsed)
        return self.répliques

    # ── Rapport et diagnostic ───────────────────────────────────────

    def _afficher_rapport(self, itération: int, elapsed: float = 0.0):
        """Affiche l'état courant avec les taux d'échange."""
        n = len(self.répliques)

        # Ligne d'énergies
        chaine_é = " | ".join(
            f"Rép.{i} T={r.température:.0f}K E={r.énergie:.2f}"
            for i, r in enumerate(self.répliques)
        )
        print(f"  [REMD it{itération}] {chaine_é}")

        # Taux d'échange
        tx = self.taux_acceptation
        for i in range(n - 1):
            t = self.nombre_échanges_par_paire[i][i + 1]
            taux = t / (itération + 1) * 100
            tx[i] = taux

        chaine_tx = ", ".join(
            f"T{i}↔T{i+1}={tx[i]:.0f}%" for i in range(n - 1)
        )
        print(f"  [REMD it{itération}] Échanges : {chaine_tx}")

        if elapsed > 0:
            reste = (elapsed / itération) * (self.itérations_restantes(itération))
            print(f"  [REMD it{itération}] Temps écoulé : {elapsed:.1f}s, "
                  f"estimation restante : {reste:.1f}s")

    def itérations_restantes(self, itération_courante: int) -> int:
        """Retourne le nombre d'itérations restantes."""
        # Approximation : on utilise le premier historique comme référence
        if self.historique_énergies:
            total = len(self.historique_énergies[0]) - 1
            return max(0, total - itération_courante)
        return 0

    def _afficher_résumé(self, elapsed: float = 0.0):
        """Affiche le résumé final de la simulation."""
        n = len(self.répliques)
        print(f"\n  [REMD] ===== RÉSUMÉ FINAL =====")
        for i, réplique in enumerate(self.répliques):
            accept = réplique.acceptations
            tent = réplique.tentatives
            taux = accept / max(1, tent) * 100
            print(
                f"  [REMD] Rép.{i} T={réplique.température:.0f}K "
                f"E_final={réplique.énergie:.2f} "
                f"échanges={taux:.1f}%"
            )
        if elapsed > 0:
            print(f"  [REMD] Temps total : {elapsed:.1f}s "
                  f"({elapsed / 60:.1f} min)")
        print(f"  [REMD] =========================\n")

    # ── Sérialisation / Checkpoint ─────────────────────────────────

    def _sauvegarder(self, chemin: str, itération: int):
        """Sauvegarde l'état courant dans un fichier JSON."""
        état = {
            "itération": itération,
            "séquence": self._séquence,
            "températures": self.températures,
            "poids_hamiltoniens": self.poids_hamiltoniens,
            "pas_mc_entre_échanges": self.pas_mc_entre_échanges,
            "longueur_fragment": self.longueur_fragment,
            "répliques": [r.vers_dictionnaire() for r in self.répliques],
            "historique_énergies": self.historique_énergies,
            "nombre_échanges_par_paire": self.nombre_échanges_par_paire,
            "timestamp": time.time(),
        }
        with open(chemin, "w") as f:
            json.dump(état, f, indent=2)
        print(f"  [REMD] Checkpoint → {chemin} (itération {itération})")

    def _restaurer(self, chemin: str, séquence: str) -> int:
        """
        Restaure l'état depuis un checkpoint.

        Retourne l'itération de départ (0 si échec).
        """
        try:
            with open(chemin) as f:
                état = json.load(f)

            if état.get("séquence") != séquence:
                print(f"  [REMD] Séquence différente — checkpoint ignoré.")
                return 0

            self.températures = état["températures"]
            self.poids_hamiltoniens = état["poids_hamiltoniens"]
            self.pas_mc_entre_échanges = état["pas_mc_entre_échanges"]
            self.longueur_fragment = état["longueur_fragment"]
            self.historique_énergies = état["historique_énergies"]
            self.nombre_échanges_par_paire = état["nombre_échanges_par_paire"]

            # Reconstruction des répliques
            self.répliques = []
            for d_rép, T, λ in zip(
                état["répliques"], self.températures, self.poids_hamiltoniens
            ):
                conf = self._construire_chaîne_étendue(séquence)
                # Note : on perd la conformation exacte — approximation
                repl = Réplique(
                    température=d_rép["température"],
                    conformation=conf,
                    énergie=d_rép["énergie"],
                    énergie_repulsive=d_rép.get("énergie_repulsive", 0.0),
                    poids=λ,
                    acceptations=d_rép["acceptations"],
                    tentatives=d_rép["tentatives"],
                )
                self.répliques.append(repl)

            n = len(self.températures)
            self.taux_acceptation = [0.0] * (n - 1)

            return état["itération"]

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"  [REMD] Échec de restauration : {e}")
            return 0

    # ── Diagnostics avancés ─────────────────────────────────────────

    def énergie_minimale(self) -> float:
        """Retourne l'énergie la plus basse trouvée par toutes les répliques."""
        if self.répliques:
            return min(r.énergie for r in self.répliques)
        return float("inf")

    def distribution_énergétique(
        self, indice_réplique: int, nb_bins: int = 30
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Histogramme des énergies pour une réplique donnée.

        Retourne (centres_bins, comptes).
        """
        if not self.historique_énergies or indice_réplique >= len(self.historique_énergies):
            return np.array([]), np.array([])
        énergies = np.array(self.historique_énergies[indice_réplique])
        comptes, bords = np.histogram(énergies, bins=nb_bins)
        centres = bords[:-1] + (bords[1] - bords[0]) / 2
        return centres, comptes

    def taux_échange_global(self) -> list[float]:
        """
        Retourne le taux d'acceptation entre chaque paire adjacente.

        Pour REMD, un taux idéal se situe entre 20% et 40%.
        """
        n = len(self.répliques)
        taux = []
        for i in range(n - 1):
            t = self.nombre_échanges_par_paire[i][i + 1]
            total = self.répliques[i].tentatives + self.répliques[i + 1].tentatives
            total //= 2
            taux.append(t / max(1, total) * 100)
        return taux

    def convergence_estimée(self, fenêtre: int = 100) -> bool:
        """
        Vérifie la convergence par stabilité de la variance sur une fenêtre.

        Méthode : on calcule la variance glissante de l'énergie pour
        chaque réplique. Si le coefficient de variation entre répliques
        est < 15%, on considère le système convergé.
        """
        variances = []
        for histo in self.historique_énergies:
            if len(histo) > fenêtre * 2:
                v1 = np.var(histo[-fenêtre * 2:-fenêtre])
                v2 = np.var(histo[-fenêtre:])
                variances.append(abs(v2 - v1) / (abs(v1) + 1e-10))
        if not variances:
            return False
        return np.mean(variances) < 0.15

    def temperature_effectif_moyenne(self) -> float:
        """
        Calcule la température effective moyenne ressentie par chaque
        réplique, basée sur le temps passé à chaque température.
        Utile pour vérifier que les répliques explorent bien tout
        l'espace des températures.
        """
        if not self.historique_températures:
            return 0.0
        temp_moy = [
            np.mean(hist) for hist in self.historique_températures
        ]
        return np.mean(temp_moy)

    def taux_rotation(self) -> float:
        """
        Taux de 'rotation' : nombre de fois par itération où une
        réplique donnée passe de T_min à T_max et revient.

        Un bon REMD a un taux de rotation élevé (≥ 1).
        """
        n = len(self.répliques)
        if n < 2 or not self.historique_températures:
            return 0.0

        rotations = 0
        # On suit la réplique index 0 dans l'espace des températures
        for hist_t in self.historique_températures:
            if len(hist_t) < 2:
                continue
            i_min = self.températures[0]
            i_max = self.températures[-1]
            état = 0  # 0 = bas, 1 = haut
            passages = 0
            for t in hist_t:
                if t <= i_min + 1:
                    if état == 1:
                        passages += 1
                    état = 0
                elif t >= i_max - 1:
                    if état == 0:
                        passages += 1
                    état = 1
            rotations += passages

        n_it = len(self.historique_températures[0])
        return rotations / max(1, n)


# ─── Point d'entrée pour les tests ────────────────────────────────────

if __name__ == "__main__":
    # Test : REMD à 3 températures sur un peptide de 15 résidus
    SÉQUENCE_TEST = "AAAAKAAAAKAAAAK"  # 15 résidus — peptide alanine-lysine

    # Températures en progression géométrique pour taux d'échange uniformes
    températures = températures_géométriques(300.0, 500.0, 3)

    print("  [REMD] ===== Test de convergence sur 15 résidus =====")
    print(f"  [REMD] Températures : {[f'{t:.1f}' for t in températures]} K")

    remd = ÉchangeDeRépliques(
        températures=températures,
        pas_mc_entre_échanges=100,
        longueur_fragment=3,
    )

    # Exécution
    répliques = remd.exécuter(
        SÉQUENCE_TEST,
        itérations=300,
        rapport_intervalle=100,
    )

    # Diagnostics
    print(f"\n  [Test] === Diagnostics ===")
    print(f"  [Test] Énergie minimale : {remd.énergie_minimale():.2f}")
    print(f"  [Test] Taux d'échange : {remd.taux_échange_global()}")
    print(f"  [Test] Taux de rotation : {remd.taux_rotation():.2f}")

    convergence = remd.convergence_estimée(fenêtre=100)
    print(f"  [Test] Convergence estimée : {'✓' if convergence else '✗'}")

    # Vérification de Boltzmann : l'énergie moyenne à haute température
    # doit être plus élevée qu'à basse température.
    énergies_finales = [r.énergie for r in répliques]
    print(f"  [Test] Énergies finales : {[f'{e:.2f}' for e in énergies_finales]}")
    ordre_correct = all(
        énergies_finales[i] <= énergies_finales[i + 1] + 5.0
        for i in range(len(énergies_finales) - 1)
    )
    print(f"  [Test] Ordre Boltzmann (E₀ ≤ E₁ + 5 ≤ E₂ + 5) : "
          f"{'✓' if ordre_correct else '✗'}")

    # Matrice d'échanges (symétrique)
    print(f"  [Test] Matrice d'échanges :")
    for i in range(len(températures)):
        ligne = " ".join(
            f"{remd.nombre_échanges_par_paire[i][j]:4d}"
            for j in range(len(températures))
        )
        print(f"         {ligne}")

    # Distribution d'énergie pour la réplique T=300K
    centres, comptes = remd.distribution_énergétique(0, nb_bins=20)
    if len(centres) > 0:
        idx_max = np.argmax(comptes)
        print(f"  [Test] Pic distribution T=300K : E ≈ {centres[idx_max]:.2f}")

    # ── Test H-REMD ────────────────────────────────────────────────
    print("\n  [REMD] === Test H-REMD (λ sur terme répulsif) ===")
    poids_h = [0.5, 0.75, 1.0]
    remd_h = ÉchangeDeRépliques(
        températures=[300.0, 300.0, 300.0],
        pas_mc_entre_échanges=100,
        poids_hamiltoniens=poids_h,
    )
    répliques_h = remd_h.exécuter(SÉQUENCE_TEST, itérations=200, rapport_intervalle=100)
    é_finales_h = [r.énergie for r in répliques_h]
    print(f"  [Test H-REMD] Énergies finales : {[f'{e:.2f}' for e in é_finales_h]}")
    ordre_h = all(
        é_finales_h[i] <= é_finales_h[i + 1] + 5.0
        for i in range(len(é_finales_h) - 1)
    )
    print(f"  [Test H-REMD] Ordre (λ croissant → E croissante) : "
          f"{'✓' if ordre_h else '✗'}")
    print(f"  [Test H-REMD] Taux d'échange : {remd_h.taux_échange_global()}")
