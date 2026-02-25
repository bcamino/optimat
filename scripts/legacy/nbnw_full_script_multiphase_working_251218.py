import numpy as np
# import pandas as pd

import subprocess
import multiprocessing as mp


from ortools.sat.python import cp_model
from tqdm import tqdm
from pymatgen.core.structure import Structure, PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pathlib import Path
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import *
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core.lattice import Lattice

try:
    import yaml
except ImportError as exc:
    raise ImportError("PyYAML is required to read params.yaml. Install it via `pip install PyYAML`.") from exc

try:
    from full_script_functions import write_gulp_input as _default_write_gulp_input
except ImportError:  # fallback when helper module is unavailable
    _default_write_gulp_input = None

from scipy.spatial.distance import pdist, squareform

from ase.visualize import view
from ase.io import write, read


from pymatgen.io.ase import AseAtomsAdaptor

import os, json, gzip, hashlib, time, textwrap


import copy


#import dataframe_image as dfi

from scipy import constants
from scipy.spatial import cKDTree, Voronoi
from scipy.spatial.qhull import QhullError

# import matplotlib.pyplot as plt


k_b = constants.physical_constants['Boltzmann constant in eV/K'][0]
# print(k_b)


class RunLogger:
    def __init__(self, directory, filename=None):
        os.makedirs(directory, exist_ok=True)
        base = os.path.basename(directory.rstrip(os.sep)) or "run"
        name = filename or f"{base}.log"
        self.path = os.path.join(directory, name)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, message):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        self._fh.write(f"[{ts}] {message}\n")
        self._fh.flush()

    def close(self):
        if not self._fh.closed:
            self._fh.close()


RUN_LOGGER = None


def log(message):
    """Mirror messages to stdout and the run log file."""
    text = str(message)
    print(text)
    if RUN_LOGGER is not None:
        RUN_LOGGER.log(text)


def load_params(path="params.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Parameter file '{path}' not found.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


PARAMS = load_params()
general_params = PARAMS.get("general", {})
optimization_params = PARAMS.get("optimization", {})
boltz_params = PARAMS.get("boltzmann", {})
perturb_params = PARAMS.get("perturb", {})
grid_params = PARAMS.get("grid", {})
io_params = PARAMS.get("io", {})

N_positions_final = 100

N_li = general_params.get("N_li", 2)
_N_initial_grid_raw = general_params.get("N_initial_grid", 100)
# Allow disabling truncation by setting N_initial_grid: false in params.
# Coerce numeric strings to int to avoid type issues.
if _N_initial_grid_raw is False:
    N_initial_grid = None
else:
    try:
        N_initial_grid = int(_N_initial_grid_raw)
    except (TypeError, ValueError):
        N_initial_grid = _N_initial_grid_raw if isinstance(_N_initial_grid_raw, (int, float)) else None
min_dist_grid = general_params.get("min_dist_grid", 1.0)
threshold_li = general_params.get("threshold_li", 1.5)
prox_penalty = general_params.get("prox_penalty", 1000)
LI_LI_EXCLUSION_ANG = general_params.get("li_li_exclusion_ang", 1.8)
KEEP_FINAL_INCUMBENTS_ONLY = general_params.get("keep_final_only", False)
li_conc_range = general_params.get("li_conc_range", [N_li, N_li + 1, 1])
RUN_NAME = general_params.get("run_name", None)
LI_SWEEP = np.arange(*li_conc_range)
# one_hot_value = general_params.get("one_hot_value", 200)
weight = general_params.get("weight", 500)
N_structures_opt = general_params.get("N_structures_opt", 2)
number_iterations = general_params.get("number_iterations", 1000)
number_runs = general_params.get("number_runs", 100)
initial_structure_path = general_params.get("initial_structure", "delithiated_tmp.cif")

threshold = optimization_params.get("threshold", 0.1)
num_iterations = optimization_params.get("num_iterations", 2)
max_time = optimization_params.get("max_time", 60)
DEDUP_INCUMBENTS = optimization_params.get("dedup_incumbents", False)
SOLVER_SEED = optimization_params.get("solver_seed", None)
SOLVER_SCALE = optimization_params.get("solver_scale", 1000.0)
CP_SAT_WORKERS = optimization_params.get("n_workers", 8)

RUN_BOLTZMANN_LOOP = boltz_params.get("run", False)
BOLTZMANN_NUM_ITERATIONS = boltz_params.get("num_iterations", 3)
BOLTZMANN_MAX_TIME = boltz_params.get("max_time", 60)
BOLTZMANN_USE_ENERGY_WINDOW = boltz_params.get("use_energy_window", True)
BOLTZMANN_ENERGY_WINDOW_EV = boltz_params.get("energy_window_ev", 0.02)
BOLTZMANN_USE_DIVERSITY_OBJECTIVE = boltz_params.get("use_diversity_objective", True)
BOLTZMANN_MIN_HAMMING_DISTANCE = boltz_params.get("min_hamming_distance", 10)
BOLTZMANN_GULP_LIMIT = boltz_params.get("gulp_limit", 50)

RUN_PERTURB_SAMPLING = perturb_params.get("run", False)
PERTURB_NUM_RUNS = perturb_params.get("num_runs", 10)
PERTURB_MAX_TIME = perturb_params.get("max_time", 20)
PERTURB_DIAG_NOISE_EV = perturb_params.get("diag_noise_ev", 0.1)
PERTURB_PAIR_NOISE_EV = perturb_params.get("pair_noise_ev", 0.0)
PERTURB_RANDOM_SEED = perturb_params.get("random_seed", 2025)
PERTURB_GULP_LIMIT = perturb_params.get("gulp_limit", 20)

M = grid_params.get("grid_resolution", 20)
N_positions_final = grid_params.get("n_positions_final", 100)
GRID_CONVERGENCE_MODE = grid_params.get("convergence_mode", "none")
GRID_OVERLAP_THRESHOLD = grid_params.get("overlap_threshold", 0.2)
DAMP_GRID_UPDATES = grid_params.get("damp_updates", False)
DAMP_GRID_THRESHOLD = grid_params.get("damp_threshold", 0.25)
GRID_USE_DECAY = grid_params.get("use_decay", False)
GRID_DECAY_THRESHOLD = grid_params.get("decay_threshold", 0.25)
GRID_DECAY_TTL = grid_params.get("decay_ttl", 2)
GRID_INIT_MODE = grid_params.get("init_mode", "uniform")
GRID_INIT_FPS_MULTIPLIER = grid_params.get("init_fps_multiplier", 2.0)
GRID_UPDATE_MODE = grid_params.get("update_mode", "probability")
GRID_USE_DUAL_GRID = grid_params.get("use_dual_grid", False)
# Default explore size: use N_initial_grid if set and numeric, otherwise N_positions_final
_explore_seed = N_initial_grid if isinstance(N_initial_grid, (int, float)) else N_positions_final
_explore_default = max(N_positions_final * 2, int(_explore_seed))
GRID_EXPLORE_SIZE = grid_params.get("explore_grid_size", _explore_default)
GRID_TRACKER_DECAY = grid_params.get("tracker_decay", 0.85)
GRID_TRACKER_MIN_VISITS = grid_params.get("tracker_min_visits", 1)
GRID_TRACKER_DECIMALS = grid_params.get("tracker_round_decimals", 4)
GRID_USE_TRACKER = grid_params.get("use_tracker", GRID_UPDATE_MODE != "probability")
GRID_RANDOM_SEED = grid_params.get("grid_random_seed", 0)
GRID_VORONOI_SHELLS = grid_params.get("voronoi_shells", 1)
try:
    GRID_VORONOI_MERGE_THRESHOLD = float(grid_params.get("voronoi_merge_threshold", 0.0) or 0.0)
except (TypeError, ValueError):
    GRID_VORONOI_MERGE_THRESHOLD = 0.0
GRID_VORONOI_SYMM_PRUNE = grid_params.get("voronoi_symmetrize_merge", False)
GRID_FIXED_GRID_SHAPE = grid_params.get("fixed_grid_shape", None)  # e.g., [9,9,9] to force NxNxN uniform grid
PHASE1_TAIL_APPEND = [
    "accuracy   10.00000\n",
    "xtol opt   8.000000\n",
    "gtol opt   8.000000\n",
    "ftol opt   8.000000\n",
    "maxcyc opt  1000000\n",
    "dump every 10 gulp.res\n",
    "output cif cryst.cif\n",
]
PHASE2_HEAD = [
    "opti conv property full nosymm comp\n",
]
PHASE2_TAIL = [
    "species   6\n",
    "Tc     core    1.971000\n",
    "Tc     shel    1.029000\n",
    "Mn     core    4.000000\n",
    "O      core    0.513000\n",
    "O      shel   -2.513000\n",
    "Li     core    1.000000\n",
    "buck\n",
    "Mn    core O     shel  3087.82600     0.264200  0.000000      0.00  2.00\n",
    "buck\n",
    "Li    core O     shel  426.48     0.300000  0.000000      0.00 25.00\n",
    "buck\n",
    "O     shel Tc    shel  1686.12500     0.296200  0.000000      0.00 2.\n",
    "buck\n",
    "O     shel O     shel  22.4100000     0.693700  32.32000      0.00 25.00\n",
    "lennard 16  6\n",
    "O     shel O     core  100.00000      0.0000000      0.800 25.000\n",
    "lennard 12  6\n",
    "O     shel Tc    shel    0.00000      6.0000000      0.000 25.000\n",
    "polynomial\n",
    " 1\n",
    "Mn    core O     shel    -1.000000     0.000000     0.000 &\n",
    " 0.000  0.000 &\n",
    " 2.000\n",
    "polynomial\n",
    " 5\n",
    "Mn    core O     shel   -91.561008   313.864964  -337.087774   161.755949 &\n",
    "  -36.321307     3.120870  0.000  2.000  3.000\n",
    "\n",
    "polynomial\n",
    "1\n",
    "Tc  shel O  shel -1.0 0.00 &\n",
    "0.00 &\n",
    "0.00 2.00\n",
    "polynomial\n",
    "5\n",
    "Tc  shel O  shel -85.63039810664417 298.45249998619494 -319.8220476295288 &\n",
    "152.56867656711893  -34.034459974422994 2.905795544978889 &\n",
    "0.0 2.00 3.00\n",
    "\n",
    "spring\n",
    "O      20.530000     #1000.0000\n",
    "spring\n",
    "Tc     148.00000     #1000.0000\n",
    "accuracy 10.000  4 50  8.000\n",
    "cutd   4.4000\n",
    "xtol opt   8.000000\n",
    "gtol opt   8.000000\n",
    "ftol opt   8.000000\n",
    "maxcyc opt  800\n",
    "dump every     10 gulp.res\n",
    "output cif cryst.cif\n",
]
PHASE3_HEAD = [
    "opti conp property full nosymm comp\n",
]
PHASE3_TAIL = [
    "species   6\n",
    "Tc     core    1.971000\n",
    "Tc     shel    1.029000\n",
    "Mn     core    4.000000\n",
    "O      core    0.513000\n",
    "O      shel   -2.513000\n",
    "Li     core    1.000000\n",
    "buck\n",
    "Mn    core O     shel  3087.82600     0.264200  0.000000      0.00  2.00\n",
    "buck\n",
    "Li    core O     shel  426.48     0.300000  0.000000      0.00 25.00\n",
    "buck\n",
    "O     shel Tc    shel  1686.12500     0.296200  0.000000      0.00 2.\n",
    "buck\n",
    "O     shel O     shel  22.4100000     0.693700  32.32000      0.00 25.00\n",
    "lennard 16  6\n",
    "O     shel O     core  100.00000      0.0000000      0.800 25.000\n",
    "lennard 12  6\n",
    "O     shel Tc    shel    0.00000      6.0000000      0.000 25.000\n",
    "polynomial\n",
    " 1\n",
    "Mn    core O     shel    -1.000000     0.000000     0.000 &\n",
    " 0.000  2.000\n",
    "polynomial\n",
    " 5\n",
    "Mn    core O     shel   -91.561008   313.864964  -337.087774   161.755949 &\n",
    "  -36.321307     3.120870  0.000  2.000  3.000\n",
    "\n",
    "polynomial\n",
    "1\n",
    "Tc  shel O  shel -1.0 0.00 0.00 &\n",
    "0.00 2.00\n",
    "polynomial\n",
    "5\n",
    "Tc  shel O  shel -85.63039810664417 298.45249998619494 -319.8220476295288 &\n",
    "152.56867656711893  -34.034459974422994 2.905795544978889 &\n",
    "0.0 2.00 3.00\n",
    "\n",
    "spring\n",
    "O      20.530000     #1000.0000\n",
    "spring\n",
    "Tc     148.00000     #1000.0000\n",
    "accuracy 10.000  4 50  8.000\n",
    "cutd   4.4000\n",
    "xtol opt   8.000000\n",
    "gtol opt   8.000000\n",
    "ftol opt   8.000000\n",
    "maxcyc opt   500\n",
    "dump every     10 gulp.res\n",
    "output cif cryst.cif\n",
]
# hard-disable dual grid for simplified debugging
GRID_USE_DUAL_GRID = False
# Radius for grid-quality check (Li→nearest-grid distance); None to disable file output
GRID_QUALITY_RADIUS = grid_params.get("quality_radius", None)
# Maximum allowed fractional drift for host atoms (Mn/Tc/O) before a relaxed structure is discarded
HOST_FRAC_DRIFT_TOL = general_params.get("max_host_frac_change", 0.05)

input_name = io_params.get("input_name", "gulp_klmc.gin")
gulp_io_path = io_params.get("gulp_io_path", "klmc/")
mace_io_path = io_params.get("mace_io_path", "mace_io_files")

global_best_energy_scaled = None

def vview(structure):
    view(AseAtomsAdaptor().get_atoms(structure))

np.seterr(divide='ignore')
# plt.style.use('tableau-colorblind10')

# import seaborn as sns
import time


class StreamingIncumbentSaver(cp_model.CpSolverSolutionCallback):
    def __init__(self, x, site_options, li_sites, mn_sites, scale, out_dir, limit=None):
        super().__init__()
        self.x = x
        self.site_options = site_options
        self.li_sites = li_sites
        self.mn_sites = mn_sites
        self.scale = scale
        self.out_dir = out_dir
        self.limit = limit
        self.count = 0
        os.makedirs(out_dir, exist_ok=True)
        self.inc_path = os.path.join(out_dir, "incumbents.jsonl.gz")

    def on_solution_callback(self):
        if self.limit is not None and self.count >= self.limit:
            return
        # Decode current incumbent
        assignment = {s: next(a for a in opts if self.Value(self.x[(s,a)]) == 1)
                      for s, opts in self.site_options.items()}
        E = None if self.ObjectiveValue() is None else self.ObjectiveValue() / self.scale

        # Minimal record (same shape as append_incumbent)
        li_on  = sorted(int(s) for s in self.li_sites if assignment[s] == "Li")
        mn3_on = sorted(int(s) for s in self.mn_sites if assignment[s] == "Mn3")
        cfg_bytes = json.dumps({"li_on": li_on, "mn3_on": mn3_on}, separators=(",", ":")).encode()
        cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:16]

        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "E": E, "li_on": li_on, "mn3_on": mn3_on,
            "n_li": len(li_on), "n_mn3": len(mn3_on),
            "cfg": cfg_hash, "tags": {"status": "INCUMBENT"}
        }
        with gzip.open(self.inc_path, "ab") as gz:
            gz.write((json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8"))

        self.count += 1


def add_li_mn_charge_balance_constraints(model: cp_model.CpModel, x, li_sites, mn_sites, N_li: int):
    """
    Enforces:
      - total number of Li atoms == N_li
      - total number of Mn3+ ions == N_li
    """
    # Li count constraint
    model.Add(sum(x[(s, "Li")] for s in li_sites) == N_li)

    # Mn3+ count constraint
    model.Add(sum(x[(s, "Mn3")] for s in mn_sites) == N_li)


def add_li_proximity_exclusions(model: cp_model.CpModel, x, proximity_groups):
    """
    Enforce Li–Li exclusion constraints.

    Parameters
    ----------
    model : cp_model.CpModel
        The model to which constraints are added.
    x : dict
        Variable dictionary {(site_id, option): BoolVar}.
    proximity_groups : list
        Each element is either:
            - a tuple/list of two site IDs (s, t)  → pairwise exclusion
            - a list/tuple of ≥2 site IDs forming a clique (all mutually too close)

    Effect
    ------
    For every group g of sites, ensures that at most one can be occupied by Li:
        sum_{s∈g} x[s, "Li"] ≤ 1
    Uses CP-SAT’s AddAtMostOne for stronger propagation.
    """
    num_groups = 0
    for g in proximity_groups:
        # Normalise input
        if isinstance(g, tuple) or isinstance(g, list):
            sites = list(g)
        else:
            raise ValueError("Each proximity group must be a list/tuple of site IDs.")
        if len(sites) < 2:
            continue

        # Collect the BoolVars for these sites
        li_vars = [x[(s, "Li")] for s in sites]
        model.AddAtMostOne(li_vars)
        num_groups += 1


def add_ut_qubo_objective(
    model: cp_model.CpModel,
    x: dict,                  # {(site_id, option_name): BoolVar}
    var2siteopt: dict,        # {qubo_col_index: (site_id, option_name)}
    Q_ut: np.ndarray,         # upper-triangular QUBO matrix (shape n x n)
    *,
    scale: float = 1000.0,    # integer scaling for CP-SAT
    tiny: float = 1e-12,
    name_prefix: str = "y"
):
    """
    Add a minimization objective equivalent to an *upper-triangular* QUBO.

    Energy = sum_i     Q[i,i] * x_(s,a)
           + sum_{i<j} Q[i,j] * (x_(s,a) AND x_(t,b))

    where Q's columns/rows index your original binary vars (Li, Mn4, Mn3),
    and var2siteopt maps each original var index -> (site, option).

    Notes:
    - We only iterate i<=j because Q is upper-triangular (no double counting).
    - Same-site off-diagonals (i<j but s==t) are skipped (redundant under one-hot).
    - Coefficients are scaled to integers for CP-SAT.
    """
    n = Q_ut.shape[0]

    # 1) Integerize (keep upper triangle semantics)
    Q = np.array(Q_ut, dtype=float, copy=True)
    Qi = np.rint(Q * scale).astype(int)
    # prune tiny
    Qi[np.abs(Qi) < tiny] = 0
    SCALE = int(scale)

    obj_terms = []
    num_diag_added = 0
    num_pairs_added = 0
    num_pairs_skipped_same_site = 0
    num_pairs_skipped_zero = 0
    num_pairs_total_seen = 0

    # 2) Diagonal: Q[i,i] * x_(s,a)
    for i in range(n):
        if i not in var2siteopt:
            continue
        s, a = var2siteopt[i]
        c = Qi[i, i]
        if c != 0:
            obj_terms.append(c * x[(s, a)])
            num_diag_added += 1

    # 3) Off-diagonals (upper triangle): Q[i,j] * y, with y = AND(x_(s,a), x_(t,b))
    for i in range(n):
        if i not in var2siteopt:
            continue
        s, a = var2siteopt[i]
        row = Qi[i]
        for j in range(i + 1, n):      # i<j, upper-triangular entries only
            num_pairs_total_seen += 1
            c = row[j]
            if c == 0:
                num_pairs_skipped_zero += 1
                continue
            if j not in var2siteopt:
                continue
            t, b = var2siteopt[j]
            if s == t:
                # cross-terms within the same physical site are redundant with one-hot
                num_pairs_skipped_same_site += 1
                continue

            y = model.NewBoolVar(f"{name_prefix}_{s}_{a}_{t}_{b}")
            model.Add(y <= x[(s, a)])
            model.Add(y <= x[(t, b)])
            model.Add(y >= x[(s, a)] + x[(t, b)] - 1)
            obj_terms.append(c * y)
            num_pairs_added += 1

    # 4) Set objective
    model.Minimize(sum(obj_terms))

    # 5) Diagnostics
    summary = {
        "num_diag_added": num_diag_added,
        "num_pairs_total_seen": num_pairs_total_seen,
        "num_pairs_added": num_pairs_added,
        "num_pairs_skipped_zero": num_pairs_skipped_zero,
        "num_pairs_skipped_same_site": num_pairs_skipped_same_site,
        "scale": SCALE,
    }
    return SCALE, summary


def perturb_qubo(Q_ut, *, diag_noise_ev=0.0, pair_noise_ev=0.0, rng=None):
    """
    Return a slightly perturbed copy of the upper-triangular QUBO matrix.
    Noise is applied to the diagonal and/or upper triangle (i<j).
    """
    if (diag_noise_ev <= 0.0) and (pair_noise_ev <= 0.0):
        return np.array(Q_ut, copy=True)

    rng = rng or np.random.default_rng()
    Q = np.array(Q_ut, copy=True)
    n = Q.shape[0]

    if diag_noise_ev > 0.0:
        Q[np.diag_indices(n)] += rng.normal(scale=diag_noise_ev, size=n)

    if pair_noise_ev > 0.0:
        iu = np.triu_indices(n, k=1)
        Q[iu] += rng.normal(scale=pair_noise_ev, size=iu[0].shape[0])

    # Ensure symmetry / upper-triangular semantics
    Q = np.triu(Q)
    return Q

def _build_qubo_energy_terms(
    model: cp_model.CpModel,
    x: dict,
    var2siteopt: dict,
    Q_ut: np.ndarray,
    *,
    scale: float = 1000.0,
    tiny: float = 1e-12,
    name_prefix: str = "y",
):
    """
    Construct the linearised QUBO energy terms without attaching an objective.
    Returns (scale_int, terms_list, diagnostics).
    Each entry in terms_list is (coefficient:int, BoolVar).
    """
    n = Q_ut.shape[0]
    Q = np.array(Q_ut, dtype=float, copy=True)
    Qi = np.rint(Q * scale).astype(int)
    Qi[np.abs(Qi) < tiny] = 0
    SCALE = int(scale)
    terms = []
    num_diag_added = 0
    num_pairs_added = 0
    num_pairs_skipped_same_site = 0
    num_pairs_skipped_zero = 0
    num_pairs_total_seen = 0
    for i in range(n):
        if i not in var2siteopt:
            continue
        s, a = var2siteopt[i]
        c = Qi[i, i]
        if c != 0:
            terms.append((c, x[(s, a)]))
            num_diag_added += 1
    for i in range(n):
        if i not in var2siteopt:
            continue
        s, a = var2siteopt[i]
        row = Qi[i]
        for j in range(i + 1, n):
            num_pairs_total_seen += 1
            c = row[j]
            if c == 0:
                num_pairs_skipped_zero += 1
                continue
            if j not in var2siteopt:
                continue
            t, b = var2siteopt[j]
            if s == t:
                num_pairs_skipped_same_site += 1
                continue
            y = model.NewBoolVar(f"{name_prefix}_{s}_{a}_{t}_{b}")
            model.Add(y <= x[(s, a)])
            model.Add(y <= x[(t, b)])
            model.Add(y >= x[(s, a)] + x[(t, b)] - 1)
            terms.append((c, y))
            num_pairs_added += 1
    summary = {
        "num_diag_added": num_diag_added,
        "num_pairs_total_seen": num_pairs_total_seen,
        "num_pairs_added": num_pairs_added,
        "num_pairs_skipped_zero": num_pairs_skipped_zero,
        "num_pairs_skipped_same_site": num_pairs_skipped_same_site,
    }
    return SCALE, terms, summary
def _decode_assignment_from_solver(site_options, solver, x_vars):
    assignment = {}
    for s, opts in site_options.items():
        for a in opts:
            if solver.Value(x_vars[(s, a)]) == 1:
                assignment[s] = a
                break
    return assignment
def _add_hamming_distance_constraint(model, x_vars, site_options, assignment, min_distance):
    delta_terms = []
    for s, opts in site_options.items():
        chosen = assignment.get(s)
        for a in opts:
            var = x_vars[(s, a)]
            if a == chosen:
                delta_terms.append(1 - var)
            else:
                delta_terms.append(var)
    if delta_terms:
        model.Add(sum(delta_terms) >= min_distance)


def append_incumbent(
    output_dir: str,
    assignment: dict,            # {site_id: option_name}
    energy_ev: float | None,
    *,
    li_sites: list,
    mn_sites: list,
    tags: dict = None           # optional metadata, e.g. {"status":"FINAL"}
):
    """Append one incumbent configuration to incumbents.jsonl.gz."""
    li_on = sorted(int(s) for s in li_sites if assignment[s] == "Li")
    mn3_on = sorted(int(s) for s in mn_sites if assignment[s] == "Mn3")

    # Stable short hash for deduplication
    cfg_bytes = json.dumps({"li_on": li_on, "mn3_on": mn3_on},
                           separators=(",", ":")).encode("utf-8")
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:16]

    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "E": None if energy_ev is None else float(energy_ev),
        "li_on": li_on,
        "mn3_on": mn3_on,
        "n_li": len(li_on),
        "n_mn3": len(mn3_on),
        "cfg": cfg_hash,
    }
    if tags:
        rec["tags"] = tags

    inc_path = os.path.join(output_dir, "incumbents.jsonl.gz")
    with gzip.open(inc_path, "ab") as gz:
        gz.write((json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8"))

    return cfg_hash


def append_unique_extxyz(extxyz_path, structures, energies, tol_frac=1e-4, tol_lat=1e-3):
    """
    Append unique pymatgen.Structure entries to an extxyz file.
    - Creates the file if it doesn't exist.
    - Deduplicates vs existing frames AND within this batch.
    - Stores per-structure energy in Atoms.info['energy'].

    Returns: (n_added, n_skipped_existing, n_skipped_within_batch)
    """
    adaptor = AseAtomsAdaptor()

    # 1) Build existing hash set (if file exists)
    existing_hashes = set()
    if os.path.exists(extxyz_path) and os.path.getsize(extxyz_path) > 0:
        try:
            for atoms in read(extxyz_path, index=":"):
                # Rebuild a pmg Structure to reuse same hashing
                pmg = adaptor.get_structure(atoms)
                existing_hashes.add(structure_hash_pbc(pmg, tol_frac, tol_lat))
        except Exception:
            # If the extxyz is huge or partially corrupted, you can decide to ignore
            pass

    # 2) Filter incoming by hash (against existing and within-batch)
    batch_hashes = set()
    unique_atoms = []
    skipped_existing = 0
    skipped_batch = 0

    for s, e in zip(structures, energies):
        h = structure_hash_pbc(s, tol_frac, tol_lat)
        if h in existing_hashes:
            skipped_existing += 1
            continue
        if h in batch_hashes:
            skipped_batch += 1
            continue

        a = adaptor.get_atoms(s)
        a.info["energy"] = float(e) if e is not None else None
        unique_atoms.append(a)
        batch_hashes.add(h)

    # 3) Append to file (or create)
    if unique_atoms:
        write(extxyz_path, unique_atoms, append=os.path.exists(extxyz_path))

    return len(unique_atoms), skipped_existing, skipped_batch

def append_extxyz(extxyz_path, structures, energies=None):
    """
    Append structures to an extxyz file without deduplication.

    Parameters
    ----------
    extxyz_path : str
        Target extxyz file path.
    structures : list[pymatgen.Structure]
        Structures to append.
    energies : list[float] or None
        Optional per-structure energies. Stored in Atoms.info["energy"].
    """
    adaptor = AseAtomsAdaptor()

    atoms_list = []
    if energies is None:
        energies = [None] * len(structures)

    for s, e in zip(structures, energies):
        a = adaptor.get_atoms(s)
        a.info["energy"] = float(e) if e is not None else None
        atoms_list.append(a)

    if not atoms_list:
        return

    # Append if file exists, otherwise create
    write(extxyz_path, atoms_list, format="extxyz",
              append=os.path.exists(extxyz_path))


def compute_grid_change_metrics(prev_grid_cart, new_grid_cart, overlap_threshold):
    prev = np.asarray(prev_grid_cart)
    new = np.asarray(new_grid_cart)
    if prev.size == 0 or new.size == 0:
        return None
    tree = cKDTree(prev)
    dists, _ = tree.query(new, k=1)
    if dists.size == 0:
        return None
    return {
        "avg_disp": float(np.mean(dists)),
        "max_disp": float(np.max(dists)),
        "overlap_frac": float(np.mean(dists <= overlap_threshold)),
    }


def report_grid_convergence(prev_grid, new_grid, mode, overlap_threshold):
    if mode.lower() == "none":
        return
    metrics = compute_grid_change_metrics(prev_grid, new_grid, overlap_threshold)
    if not metrics:
        return
    mode = mode.lower()
    if mode in ("displacement", "both"):
        log(f"Grid displacement → avg: {metrics['avg_disp']:.4f} Å, max: {metrics['max_disp']:.4f} Å")
    if mode in ("overlap", "both"):
        log(f"Grid overlap fraction (≤ {overlap_threshold:.3f} Å): {metrics['overlap_frac']*100:.2f}%")


def append_grid_extxyz(extxyz_path, lattice, grid_cart, tag=None):
    """Append the current Li-grid (cartesian coords) as an extxyz frame."""
    grid_cart = np.asarray(grid_cart, dtype=float)
    if grid_cart.size == 0:
        return 0

    frac = lattice.get_fractional_coords(grid_cart)
    structure = Structure(lattice, ["Li"] * len(frac), frac, coords_are_cartesian=False)
    atoms = AseAtomsAdaptor().get_atoms(structure)
    if tag is not None:
        atoms.info["tag"] = tag
    write(extxyz_path, atoms, append=os.path.exists(extxyz_path))
    return 1


def _frac_delta(a, b):
    """Minimum-image fractional delta."""
    return ((a - b + 0.5) % 1.0) - 0.5


def filter_structures_by_host_drift(structures, energies, reference, max_frac_change=0.05, host_species=("Mn", "Tc", "O")):
    """
    Drop optimized structures whose host (Mn/Tc/O) fractional coordinates drift
    more than max_frac_change (per-component fractional units) from the reference.
    Returns (kept_structures, kept_energies, rejected_count).
    """
    if not structures:
        return [], [], 0

    ref_indices = [i for i, site in enumerate(reference) if site.specie.symbol in host_species]
    ref_coords = reference.frac_coords[ref_indices]

    kept_structs = []
    kept_energies = []
    rejected = 0

    for s, e in zip(structures, energies if energies else [None] * len(structures)):
        if len(s) <= max(ref_indices):
            rejected += 1
            continue
        coords = s.frac_coords[ref_indices]
        delta = np.abs(_frac_delta(coords, ref_coords))
        if np.any(delta > max_frac_change):
            rejected += 1
            continue
        kept_structs.append(s)
        if energies:
            kept_energies.append(e)

    return kept_structs, kept_energies, rejected


def compute_grid_quality(li_structures, grid_cart, lattice, radius=None):
    """
    Compute nearest-grid distances for all Li in li_structures relative to grid_cart (cartesian).
    Returns metrics dict.
    """
    li_structures = li_structures or []
    grid_cart = np.asarray(grid_cart, dtype=float)
    if grid_cart.size == 0 or len(li_structures) == 0:
        return {"n_li": 0, "n_structures": len(li_structures)}

    li_coords = []
    for s in li_structures:
        li_idx = [i for i, site in enumerate(s) if site.specie.symbol == "Li"]
        if li_idx:
            li_coords.append(s.cart_coords[li_idx])
    if not li_coords:
        return {"n_li": 0, "n_structures": len(li_structures)}

    li_coords = np.concatenate(li_coords, axis=0)
    tree = cKDTree(grid_cart)
    dists, _ = tree.query(li_coords, k=1)
    metrics = {
        "n_li": int(len(li_coords)),
        "n_structures": int(len(li_structures)),
        "dist_min": float(np.min(dists)),
        "dist_max": float(np.max(dists)),
        "dist_mean": float(np.mean(dists)),
        "dist_std": float(np.std(dists)),
    }
    if radius is not None:
        within = np.count_nonzero(dists <= radius)
        metrics["within_radius"] = int(within)
        metrics["within_radius_frac"] = float(within / len(dists))
        metrics["radius"] = float(radius)
    return metrics


def write_grid_quality(path, metrics, tag):
    """Append grid quality metrics to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = {"tag": tag, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    rec.update(metrics or {})
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def append_li_positions_extxyz(path, structures, tag_prefix="iter"):
    """
    Dump Li positions from optimized structures into a cumulative extxyz.
    One frame per structure.
    """
    if not structures:
        return
    lattice = structures[0].lattice
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for idx, s in enumerate(structures):
        li_idx = [i for i, site in enumerate(s) if site.specie.symbol == "Li"]
        if not li_idx:
            continue
        coords = s.cart_coords[li_idx]
        atoms = Atoms(positions=coords, symbols=["Li"] * len(li_idx), cell=lattice.matrix, pbc=True)
        atoms.info["tag"] = f"{tag_prefix}_{idx}"
        write(path, atoms, append=True)


def _frac_delta(a, b):
    """Minimum-image fractional delta."""
    return ((a - b + 0.5) % 1.0) - 0.5


def filter_structures_by_host_drift(structures, energies, reference, max_frac_change=0.05, host_species=("Mn", "Tc", "O")):
    """
    Drop optimized structures whose host (Mn/Tc/O) fractional coordinates drift
    more than max_frac_change (per-component fractional units) from the reference.
    Returns (kept_structures, kept_energies, rejected_count).
    """
    if not structures:
        return [], [], 0

    ref_indices = [i for i, site in enumerate(reference) if site.specie.symbol in host_species]
    ref_coords = reference.frac_coords[ref_indices]

    kept_structs = []
    kept_energies = []
    rejected = 0

    for s, e in zip(structures, energies if energies else [None] * len(structures)):
        if len(s) < max(ref_indices) + 1:
            rejected += 1
            continue
        coords = s.frac_coords[ref_indices]
        delta = np.abs(_frac_delta(coords, ref_coords))
        if np.any(delta > max_frac_change):
            rejected += 1
            continue
        kept_structs.append(s)
        if energies:
            kept_energies.append(e)

    return kept_structs, kept_energies, rejected


def farthest_point_sampling(points, k, seed=0):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0 or k <= 0:
        return np.empty((0, 3))
    if k >= n:
        return pts.copy()
    rng = np.random.default_rng(seed)
    start = rng.integers(n)
    selected = [start]
    dists = np.linalg.norm(pts - pts[start], axis=1)
    for _ in range(1, k):
        next_idx = int(np.argmax(dists))
        selected.append(next_idx)
        new_d = np.linalg.norm(pts - pts[next_idx], axis=1)
        dists = np.minimum(dists, new_d)
    return pts[selected]


def generate_voronoi_grid(structure, min_dist_grid=1.0, max_points=None, neighbor_shell=1, decimals=6):
    """
    Estimate interstitial sites from the Voronoi network of the host structure.
    Returns cartesian coordinates sorted by distance from existing atoms (largest voids first).
    """
    coords = structure.cart_coords
    if len(coords) == 0:
        return np.empty((0, 3)), np.empty((0,))

    lattice = structure.lattice
    translations = []
    rng = range(-neighbor_shell, neighbor_shell + 1)
    for i in rng:
        for j in rng:
            for k in rng:
                translations.append(np.array([i, j, k], dtype=float))
    translations = np.array(translations, dtype=float)
    trans_cart = translations @ lattice.matrix

    pts = []
    central_indices = []
    idx = 0
    for t_frac, t_cart in zip(translations, trans_cart):
        for base in coords:
            pts.append(base + t_cart)
            if np.allclose(t_frac, 0.0):
                central_indices.append(idx)
            idx += 1
    pts = np.array(pts, dtype=float)
    if len(pts) < 4:
        return np.empty((0, 3)), np.empty((0,))

    try:
        vor = Voronoi(pts)
    except QhullError:
        return np.empty((0, 3)), np.empty((0,))

    tree = cKDTree(coords)
    unique = {}
    for point_index in central_indices:
        region_index = vor.point_region[point_index]
        if region_index == -1:
            continue
        region = vor.regions[region_index]
        if not region or any(v_idx == -1 for v_idx in region):
            continue
        vertices = vor.vertices[region]
        for vertex in vertices:
            frac = lattice.get_fractional_coords(vertex)
            frac = frac % 1.0
            cart = lattice.get_cartesian_coords(frac)
            dist = tree.query(cart, k=1)[0]
            if dist <= min_dist_grid:
                continue
            key = tuple(np.round(frac, decimals))
            prev = unique.get(key)
            if prev is None or dist > prev[1]:
                unique[key] = (cart, dist)

    if not unique:
        return np.empty((0, 3)), np.empty((0,))

    coords_list = np.array([val[0] for val in unique.values()])
    dists = np.array([val[1] for val in unique.values()])
    order = np.argsort(dists)[::-1]
    coords_list = coords_list[order]
    dists = dists[order]
    if max_points is not None and len(coords_list) > max_points:
        coords_list = coords_list[:max_points]
        dists = dists[:max_points]
    return coords_list, dists


def initialize_li_grids(structure, *, n_model, n_dense, min_dist, mode="uniform", seed=0, use_dual=False, voronoi_shells=1):
    mode = (mode or "uniform").lower()
    if GRID_FIXED_GRID_SHAPE:
        base_model = generate_filtered_grid(
            structure,
            N_initial_grid=np.prod(GRID_FIXED_GRID_SHAPE),
            min_dist_grid=min_dist,
            num_points=GRID_FIXED_GRID_SHAPE,
        )
        model = base_model
        explore = base_model
        return model, explore

    seed_count = n_model if n_model else 1000
    base_model = generate_filtered_grid(structure, N_initial_grid=seed_count, min_dist_grid=min_dist)
    if mode == "farthest_point":
        dense = generate_filtered_grid(structure, N_initial_grid=n_dense, min_dist_grid=min_dist)
        count = n_model if n_model else len(dense)
        sampled = farthest_point_sampling(dense, count if count > 0 else len(dense), seed=seed)
        model = sampled
        explore = dense if use_dual else model
    elif mode == "voronoi":
        vor_cart, _ = generate_voronoi_grid(
            structure,
            min_dist_grid=min_dist,
            max_points=max(n_dense, n_model or n_dense),
            neighbor_shell=max(1, int(voronoi_shells))
        )
        # Optional symmetrize + merge for Voronoi points
        if GRID_VORONOI_SYMM_PRUNE and GRID_VORONOI_MERGE_THRESHOLD > 0 and vor_cart.size > 0:
            symmops = SpacegroupAnalyzer(structure).get_symmetry_operations()
            frac = structure.lattice.get_fractional_coords(vor_cart) % 1.0
            sym_coords = []
            for op in symmops:
                sym_coords.append(op.operate_multi(frac) % 1.0)
            sym_coords = np.vstack(sym_coords)
            merged_frac = average_close_points(sym_coords, GRID_VORONOI_MERGE_THRESHOLD)
            merged_cart = structure.lattice.get_cartesian_coords(merged_frac % 1.0)
            log(f"Voronoi sym-prune applied (threshold {GRID_VORONOI_MERGE_THRESHOLD} Å): {len(vor_cart)} -> {len(merged_cart)} points")
            vor_cart = merged_cart
        # Optional clustering of close Voronoi points
        if GRID_VORONOI_MERGE_THRESHOLD > 0 and vor_cart.size > 0:
            before = len(vor_cart)
            vor_cart = average_close_points(vor_cart, GRID_VORONOI_MERGE_THRESHOLD)
            log(f"Voronoi merge applied (threshold {GRID_VORONOI_MERGE_THRESHOLD} Å): {before} -> {len(vor_cart)} points")
        if vor_cart.size == 0:
            model = base_model
            explore = base_model if use_dual else model
        else:
            if n_model is None:
                model = vor_cart
            elif len(vor_cart) < n_model:
                supplement = base_model[:max(0, n_model - len(vor_cart))]
                model = np.vstack([vor_cart, supplement]) if supplement.size else vor_cart
            else:
                model = vor_cart[:n_model]
            if use_dual:
                explore = vor_cart
            else:
                explore = model
    else:
        model = base_model
        explore = base_model if use_dual else model
    return model, explore


def init_grid_tracker():
    return {}


def _decay_tracker(tracker, decay):
    to_delete = []
    for key, rec in tracker.items():
        rec["score"] *= decay
        if rec["score"] < 1e-6:
            to_delete.append(key)
    for key in to_delete:
        tracker.pop(key, None)
    return tracker


def update_grid_tracker(tracker, coords, weights, iteration, *, decay=0.85, min_visits=1, decimals=4):
    if tracker is None:
        return None
    tracker = _decay_tracker(tracker, decay)
    for coord, weight in zip(coords, weights):
        if weight <= 0:
            continue
        frac = np.asarray(coord, dtype=float) % 1.0
        key = tuple(np.round(frac, decimals))
        rec = tracker.get(key)
        if rec is None:
            tracker[key] = {
                "coord": frac,
                "score": float(weight),
                "visits": float(weight),
                "last_seen": iteration,
            }
        else:
            rec["coord"] = (rec["coord"] + frac) / 2.0
            rec["score"] += float(weight)
            rec["visits"] += float(weight)
            rec["last_seen"] = iteration
    tracker = {
        k: v for k, v in tracker.items()
        if v["visits"] >= min_visits or (iteration - v.get("last_seen", iteration)) <= 2
    }
    return tracker


def select_tracker_coords(tracker, limit, min_visits=1):
    if tracker is None or len(tracker) == 0:
        return np.empty((0, 3))
    sorted_records = sorted(
        tracker.values(),
        key=lambda rec: (rec["score"], rec["visits"]),
        reverse=True,
    )
    coords = []
    for rec in sorted_records:
        if rec["visits"] >= min_visits:
            coords.append(rec["coord"] % 1.0)
        if len(coords) >= limit:
            break
    if not coords:
        coords = [rec["coord"] % 1.0 for rec in sorted_records[:limit]]
    return np.array(coords)


def extract_li_cartesian_coords(structure: Structure):
    """Return Cartesian coordinates of Li sites in the current structure."""
    if structure is None:
        return np.empty((0, 3))
    mask = np.array(structure.atomic_numbers) == 3
    if not mask.any():
        return np.empty((0, 3))
    return structure.cart_coords[mask]


def damp_grid_update(prev_grid_cart, new_grid_cart, threshold_ang):
    if prev_grid_cart is None or new_grid_cart is None:
        return new_grid_cart
    prev = np.asarray(prev_grid_cart, dtype=float)
    new = np.asarray(new_grid_cart, dtype=float)
    if prev.size == 0:
        return new
    tree = cKDTree(prev)
    dists, idxs = tree.query(new, k=1)
    result = prev.copy()
    matched = np.zeros(len(prev), dtype=bool)
    for dist, idx, coord in zip(dists, idxs, new):
        if dist <= threshold_ang:
            result[idx] = 0.5 * (result[idx] + coord)
            matched[idx] = True
        else:
            result = np.vstack([result, coord]) if result.size else np.array([coord])
    return result


def update_grid_with_decay(new_grid_cart, decay_records, threshold_ang, ttl):
    new = np.asarray(new_grid_cart, dtype=float)
    if new.size == 0:
        # Still decay existing records
        updated = []
        for rec in decay_records:
            rec["ttl"] -= 1
            if rec["ttl"] > 0:
                updated.append(rec)
        return np.array([rec["coord"] for rec in updated]), updated

    records = list(decay_records)
    coords_prev = np.array([rec["coord"] for rec in records]) if records else np.empty((0, 3))
    tree = cKDTree(coords_prev) if len(records) > 0 else None

    refreshed = [False] * len(records)

    for coord in new:
        if tree is not None and coords_prev.size > 0:
            dist, idx = tree.query(coord, k=1)
            if dist <= threshold_ang:
                records[idx]["coord"] = 0.5 * (records[idx]["coord"] + coord)
                records[idx]["ttl"] = ttl
                refreshed[idx] = True
                continue
        records.append({"coord": coord, "ttl": ttl})
        refreshed.append(True)

    updated = []
    for rec, was_refreshed in zip(records, refreshed):
        if not was_refreshed:
            rec["ttl"] -= 1
        if rec["ttl"] > 0:
            updated.append(rec)
    coords = np.array([rec["coord"] for rec in updated])
    return coords, updated


def average_close_points(symmetrised_coords, threshold, return_counts=False):
    """
    Averages the coordinates of points that are closer than the given threshold.

    Parameters:
    symmetrised_coords (np.array): (N,3) array of 3D coordinates.
    threshold (float): Distance threshold for grouping points.

    Returns:
    np.array: New array with averaged coordinates.
    """
    # Compute pairwise distances
    distances = squareform(pdist(symmetrised_coords))
    
    # Track processed points
    visited = np.zeros(len(symmetrised_coords), dtype=bool)
    averaged_coords = []
    counts = []

    for i in range(len(symmetrised_coords)):
        if visited[i]:  # Skip if already processed
            continue

        # Find all close points (including itself)
        close_points = np.where(distances[i] < threshold)[0]
        visited[close_points] = True  # Mark as processed

        # Compute the average of these points
        avg_coord = np.mean(symmetrised_coords[close_points], axis=0)
        averaged_coords.append(avg_coord)
        counts.append(len(close_points))

    averaged_coords = np.array(averaged_coords)
    if return_counts:
        return averaged_coords, np.array(counts, dtype=int)
    return averaged_coords


def build_li_proximity_groups(
    li_grid_coords,
    threshold_ang,
    *,
    lattice=None,
    coords_are_cartesian=True,
    site_ids=None,
    return_pairs_also=True,
):
    """
    Build Li–Li proximity groups for CP-SAT 'AtMostOne' constraints.

    Parameters
    ----------
    li_grid_coords : (M,3) array
        Li candidate site coordinates. If 'coords_are_cartesian' is False,
        they are treated as fractional.
    threshold_ang : float
        Distance threshold (Å) for "too close".
    lattice : (3,3) array-like or pymatgen Lattice, optional
        Required if periodic MIC distances matter. If None and
        coords_are_cartesian=True, uses plain Euclidean distances (no PBC).
    coords_are_cartesian : bool
        Whether li_grid_coords are in Cartesian. If True and lattice is given,
        MIC distances are used. If False, coords are treated as fractional.
    site_ids : list[int], optional
        CP site IDs aligned with li_grid_coords. If None, uses range(M).
    return_pairs_also : bool
        If True, also return the raw list of close pairs.

    Returns
    -------
    groups : list[list[int]]
        Each group is a clique (size ≥ 2) of site IDs that are mutually
        within the threshold (use one AddAtMostOne per group).
    pairs  : list[tuple[int,int]]  (only if return_pairs_also=True)
        All offending pairs (site_i, site_j) by ID.
    """
    coords = np.asarray(li_grid_coords, dtype=float)
    M = coords.shape[0]
    if site_ids is None:
        site_ids = list(range(M))

    # Build fractional coords (needed for MIC)
    if coords_are_cartesian:
        if lattice is None:
            # no PBC: simple Euclidean threshold
            # build edges directly
            edges = []
            for i in range(M):
                for j in range(i+1, M):
                    if np.linalg.norm(coords[j] - coords[i]) < threshold_ang:
                        edges.append((i, j))
        else:
            L = lattice.matrix if hasattr(lattice, "matrix") else np.asarray(lattice, dtype=float)
            f = _frac_coords(coords, L)
            edges = _pair_edges_with_threshold(f % 1.0, L, threshold_ang)
    else:
        # coords already fractional
        if lattice is None:
            raise ValueError("lattice is required when using fractional coords to compute distances.")
        L = lattice.matrix if hasattr(lattice, "matrix") else np.asarray(lattice, dtype=float)
        edges = _pair_edges_with_threshold(coords % 1.0, L, threshold_ang)

    # Maximal cliques from the proximity graph
    cliques = _maximal_cliques_from_edges(M, edges)

    # Map internal indices -> site_ids
    if len(site_ids) != M:
        log(f"[warn] build_li_proximity_groups: site_ids ({len(site_ids)}) != coords ({M}); truncating to available entries.")
    groups = []
    for clique in cliques:
        mapped = []
        for i in clique:
            if i < len(site_ids):
                mapped.append(site_ids[i])
        if len(mapped) >= 2:
            groups.append(mapped)
    if return_pairs_also:
        pairs = []
        for (i, j) in edges:
            if i < len(site_ids) and j < len(site_ids):
                pairs.append((site_ids[i], site_ids[j]))
        return groups, pairs
    return groups


def build_new_structural_model(
    opt_structures,
    M,
    N_positions_final,
    initial_structure,
    threshold,
    return_stats=False,
    fix_angles=True,
    grid_update_mode="probability",
    explore_target=None,
    use_dual=False,
    tracker=None,
    tracker_decay=0.85,
    tracker_min_visits=1,
    tracker_decimals=4,
    iteration=0,
):


    if fix_angles == True:
        # --- Average only the lattice lengths, keep angles fixed from the reference structure ---
        alpha = 90
        beta = 90
        gamma = 90

        a_vals, b_vals, c_vals = [], [], []
        for s in opt_structures:
            a, b, c = s.lattice.lengths
            a_vals.append(a)
            b_vals.append(b)
            c_vals.append(c)

        a_mean = np.mean(a_vals)
        b_mean = np.mean(b_vals)
        c_mean = np.mean(c_vals)

        lattice_new = Lattice.from_parameters(
            a_mean,
            b_mean,
            c_mean,
            alpha,
            beta,
            gamma
        )

    else:
        lattice_all = []
        for structure in opt_structures:
            lattice_all.append(structure.lattice.matrix)
        lattice_new = np.mean(lattice_all,axis=0)

    mn_coord_new = []
    o_coord_new = []

    for structure in opt_structures:
        work = structure.copy()
        work.replace_species({'Tc':'Mn'})
        mn_indices_new = np.where(np.array(work.atomic_numbers)==25)[0]
        o_indices_new = np.where(np.array(work.atomic_numbers)==8)[0]

        mn_coord_new.append(work.frac_coords[mn_indices_new]%1)
        o_coord_new.append(work.frac_coords[o_indices_new]%1)

    mn_coord_new = unwrap_frac_coords(mn_coord_new)
    o_coord_new = unwrap_frac_coords(o_coord_new)

    mn_coord_new = np.array(mn_coord_new)
    o_coord_new = np.array(o_coord_new)

    mn_coord_average = np.mean(mn_coord_new,axis=0)
    o_coord_average = np.mean(o_coord_new,axis=0)
    if return_stats == True:

        mn_coord_average_std = np.average(np.std(mn_coord_new,axis=0))
        o_coord_average_std = np.average(np.std(o_coord_new,axis=0))

        mn_coord_max_std = np.average(np.max(mn_coord_new,axis=0))
        o_coord_max_std = np.average(np.max(o_coord_new,axis=0))

    
    #Lithium

    li_coords_all = []

    for structure in opt_structures:
        work = structure.copy()
        work.replace_species({'Tc':'Mn'})
        
        li_index = np.where(np.array(work.atomic_numbers) == 3)[0]

        li_coords = work.frac_coords[li_index]
        li_coords_all.extend(li_coords)
            
    li_coords_all = unwrap_frac_coords(li_coords_all)

    li_coords_all = np.array(li_coords_all)
    extras = {}
    explore_target = explore_target or N_positions_final
    explore_target = max(explore_target, N_positions_final)
    grid_mode = (grid_update_mode or "probability").lower()

    centers = find_fractional_centers(M)
    model_coords_raw = np.empty((0, 3))
    explore_coords_raw = np.empty((0, 3))

    if grid_mode == "probability":
        grid = compute_probability_grid(li_coords_all, M)
        extras["probability_grid_sum"] = float(np.sum(grid))
        top_model = find_top_x_points(grid, centers, N_positions_final)
        top_explore = find_top_x_points(grid, centers, explore_target)
        model_coords_raw = np.array([pt[0] for pt in top_model]) if top_model else np.empty((0, 3))
        explore_coords_raw = np.array([pt[0] for pt in top_explore]) if top_explore else model_coords_raw
    else:
        model_coords_raw = li_coords_all
        explore_coords_raw = li_coords_all

    #Symmetrise
    symmops = SpacegroupAnalyzer(initial_structure).get_symmetry_operations()
    num_symmops = len(symmops)

    def symmetrise_and_average(coord_array, return_counts=False):
        coord_array = np.asarray(coord_array, dtype=float)
        if coord_array.size == 0:
            if return_counts:
                return np.empty((0, 3)), np.array([], dtype=int)
            return np.empty((0, 3))
        symmetrised = []
        for symmop in symmops:
            for coord in coord_array:
                symmetrised.append(symmop.operate(coord) % 1)
        symmetrised = np.array(symmetrised)
        if return_counts:
            return average_close_points(symmetrised, threshold, return_counts=True)
        return average_close_points(symmetrised, threshold)

    if grid_mode == "cluster":
        averaged_symmetrised_coords, cluster_counts = symmetrise_and_average(explore_coords_raw, return_counts=True)
        extras["cluster_counts"] = cluster_counts.tolist()
        if tracker is not None:
            tracker = update_grid_tracker(
                tracker,
                averaged_symmetrised_coords,
                cluster_counts if len(cluster_counts) else np.ones(len(averaged_symmetrised_coords)),
                iteration,
                decay=tracker_decay,
                min_visits=tracker_min_visits,
                decimals=tracker_decimals,
            )
            extras["tracker_size"] = len(tracker)
            ranked_coords = select_tracker_coords(tracker, explore_target, tracker_min_visits)
        else:
            order = np.argsort(cluster_counts)[::-1] if len(cluster_counts) else np.array([], dtype=int)
            ranked_coords = averaged_symmetrised_coords[order] if len(order) else averaged_symmetrised_coords
        if ranked_coords.size == 0:
            ranked_coords = averaged_symmetrised_coords
        model_coords = ranked_coords[:N_positions_final] if ranked_coords.size else ranked_coords
        explore_coords = ranked_coords[:explore_target] if use_dual else model_coords
    else:
        averaged_model = symmetrise_and_average(model_coords_raw)
        averaged_explore = symmetrise_and_average(explore_coords_raw)
        model_coords = averaged_model
        explore_coords = averaged_explore if use_dual else averaged_model

    model_coords = np.array(model_coords)
    if model_coords.size == 0 and explore_coords.size > 0:
        model_coords = np.array(explore_coords[:N_positions_final])
    if explore_coords.size == 0:
        explore_coords = model_coords

    extras["explore_grid_frac"] = np.array(explore_coords)
    extras["tracker"] = tracker

    lattice = lattice_new if isinstance(lattice_new, Lattice) else Lattice(lattice_new)

    # Then use it everywhere:
    li_sites = [PeriodicSite('Li', coord, lattice) for coord in model_coords]
    mn_sites = [PeriodicSite('Mn', coord, lattice) for coord in mn_coord_average]
    o_sites = [PeriodicSite('O', coord, lattice) for coord in o_coord_average]

    # Combine and build
    all_sites = li_sites + mn_sites + o_sites
    structure = Structure.from_sites(all_sites)

    return structure, model_coords, extras


def build_QUBO(structure, threshold_li=0, prox_penalty=0):
    
    structure_tmp = copy.deepcopy(structure)
    structure_tmp.add_site_property("charge", [+1.0] * structure_tmp.num_sites)
    # structure_tmp.to_file('data/bnw/tmp/test_structure.cif')
    # ewald_matrix = compute_ewald_matrix_fast(structure,triu=True)
    ewald = EwaldSummation(structure_tmp, eta=None, w=1)

    # ewald_matrix = ewald.total_energy_matrix
    ewald_matrix = ewald.real_space_energy_matrix + ewald.reciprocal_space_energy_matrix
    ewald_matrix = np.triu(ewald_matrix,1)

    print(ewald_matrix[0])

    charges = {
        25: [4,3],
        3: [1],
        8: [-2]
    }

    if threshold_li > 0 and prox_penalty > 0:
        # THE PROX PENALTY WILL BE MULTIPLIED BY THE CHARGES (1 IN THIS CASE)
        # Add contstraint on proximity of lithium atoms
        dm = structure.distance_matrix
        num_sites = structure.num_sites
        li_indices = np.where(np.array(structure.atomic_numbers) == 3)[0]
        num_o = np.sum(np.array(structure.atomic_numbers) == 8)
        
        # Create a mask for all (i,j) pairs where both i and j are Li
        li_mask = np.zeros((num_sites, num_sites), dtype=bool)
        li_mask[np.ix_(li_indices, li_indices)] = True

        # Apply the distance threshold
        below_thresh_mask = dm < threshold_li

        # Combine masks
        final_mask = li_mask & below_thresh_mask

        # Create constraint matrix
        prox_constraint = np.where(final_mask, prox_penalty, 0)
        np.fill_diagonal(prox_constraint,0)

        ewald_matrix += prox_constraint
  
    ewald_discrete, expanded_charges, expanded_matrix = compute_discrete_ewald_matrix(structure, charges, ewald_matrix)

    species_dict = {'Mn': ['Mn', 'Tc']}  # Mn sites can be either Mn4+ (Mn) or Mn3+ (Tc)
    buckingham_dict = {'Li-O':[426.480 ,    0.3000  ,   0.00],
                        'Mn-O':[3087.826    ,   0.2642 ,    0.00], # This is the Mn4+
                        'Tc-O':[1686.125  ,    0.2962 ,    0.00], # This is the Mn3+
                        'O-O' : [22.410  ,     0.6937,   32.32]
                        }
    buckingham_discrete, species_vector = compute_buckingham_matrix_discrete_fast(
        structure, species_dict, buckingham_dict, R_max=25.0
    )

    Q_discrete = build_qubo_discrete_from_Ewald_IP(ewald_discrete,buckingham_discrete)
    
    # === Create mask to remove 'O' sites ===
    mask = [el != 'O' for el in species_vector]

    # Apply mask to species vector
    reduced_species_vector = [el for el, keep in zip(species_vector, mask) if keep]

    # Apply mask to QUBO matrix
    QUBO, oo_energy = reduce_qubo_discrete_limno(Q_discrete, species_vector)

    # Compute correct indices based on reduced species vector
    li_indices = [i for i, el in enumerate(reduced_species_vector) if el == 'Li']
    mn_indices = [i for i, el in enumerate(reduced_species_vector) if el in ('Mn', 'Tc')]


    # # THIS IS A QUICK FIX THAT ONLY WORKS IF THE ATOMS ARE IN ORDER Mn-O-Li
    # li_indices = np.array([i for i, el in enumerate(species_vector) if el == 'Li']) - num_o
    # li_indices = li_indices.tolist()
 
    # mn_indices = [i for i, el in enumerate(species_vector) if el in ('Mn', 'Tc')]

    return QUBO, li_indices, mn_indices, ewald_discrete, buckingham_discrete


def build_qubo_discrete_from_Ewald_IP(ewald_discrete,buckingham_matrix):
    Q = ewald_discrete + buckingham_matrix

    return Q


def build_site_option_maps_from_indices(li_indices, mn_indices):
    """
    Input:
      li_indices: list[int]  -> QUBO columns that mean 'Li present'
      mn_indices: list[int]  -> [Mn4_0, Mn3_0, Mn4_1, Mn3_1, ...]
    Output:
      site_options: dict[site_id] -> list[str] of options
      var2siteopt: dict[qubo_col] -> (site_id, option_name)
      li_sites: list[int] of site_ids that are Li grid sites
      mn_sites: list[int] of site_ids that are Mn sites
    """
    assert len(mn_indices) % 2 == 0, "mn_indices must be even length (pairs)."

    site_options = {}
    var2siteopt  = {}
    li_sites, mn_sites = [], []

    # 1) Li grid sites: create a site with options ["Empty","Li"] for each li_index
    for k in li_indices:
        s = len(site_options)
        site_options[s] = ["Empty", "Li"]
        var2siteopt[k]  = (s, "Li")      # the QUBO var corresponds to the "Li" option
        li_sites.append(s)

    # 2) Mn sites: every consecutive pair -> one Mn site with ["Mn4","Mn3"]
    for p in range(0, len(mn_indices), 2):
        k4 = mn_indices[p]
        k3 = mn_indices[p+1]
        s = len(site_options)
        site_options[s] = ["Mn4", "Mn3"]
        var2siteopt[k4] = (s, "Mn4")
        var2siteopt[k3] = (s, "Mn3")
        mn_sites.append(s)

    return site_options, var2siteopt, li_sites, mn_sites


def build_x_vars_and_onehot(model: cp_model.CpModel, site_options):
    """
    Make BoolVars x[(s,a)] and add one-hot per site: sum_a x[s,a] == 1
    Returns: x dict
    """
    x = {}
    for s, opts in site_options.items():
        for a in opts:
            x[(s, a)] = model.NewBoolVar(f"x_{s}_{a}")
        # one-hot: exactly one option per site
        model.Add(sum(x[(s, a)] for a in opts) == 1)
    return x


def compute_buckingham_matrix_discrete(structure, species_dict, buckingham_dict, R_max, max_shift=None,
                                       distance_analysis=False, distance_threshold=0.1):
    """
    Compute an expanded Buckingham potential matrix for a system where certain chemical species
    can exist as multiple elements (e.g., 'Ca' → ['Mg', 'Ca']).

    Parameters
    ----------
    structure : pymatgen.Structure
        The atomic structure.
    species_dict : dict
        Dictionary mapping species labels (str) to possible alternative elements.
        Example: {'Ca': ['Mg', 'Ca']}.
    buckingham_dict : dict
        Dictionary of Buckingham parameters for each element pair (e.g., "Ca-F").
    R_max : float
        Maximum real space cutoff.
    max_shift : int, optional
        Maximum lattice vector translation in each direction.
    distance_analysis : bool
        If True, will flag very short distances.
    distance_threshold : float
        Threshold for flagging short distances.

    Returns
    -------
    buckingham_matrix_expanded : np.ndarray
        The expanded Buckingham interaction matrix.
    expanded_species : list of str
        Species labels corresponding to the rows/columns of the matrix.
    """
    import numpy as np
    from pymatgen.core.periodic_table import Element
    from tqdm import tqdm

    def buckingham_potential(params, r):
        A, rho, C = params
        return A * np.exp(-r / rho) - C / r**6 if r != 0 else 0

    frac_coords = structure.frac_coords
    lattice_vectors = structure.lattice.matrix
    cart_coords = frac_coords @ lattice_vectors
    distance_matrix = structure.distance_matrix
    sites = structure.sites

    N = len(structure)
    index_map = {}
    expanded_species = []
    new_idx = 0

    # Step 1: Build index mapping and expanded species list
    for i, site in enumerate(sites):
        sp = str(site.specie)
        if sp in species_dict:
            options = species_dict[sp]
            index_map[i] = list(range(new_idx, new_idx + len(options)))
            expanded_species.extend(options)
            new_idx += len(options)
        else:
            index_map[i] = [new_idx]
            expanded_species.append(sp)
            new_idx += 1

    expanded_N = len(expanded_species)
    buckingham_matrix_expanded = np.zeros((expanded_N, expanded_N))

    # Step 2: Determine max lattice shift
    if max_shift is None:
        max_real = np.ceil(R_max / np.linalg.norm(lattice_vectors, axis=1)).astype(int)
        nx, ny, nz = max_real
    else:
        nx = ny = nz = max_shift

    # Step 3: Fill the expanded matrix
    for i in tqdm(range(N), desc="Buckingham matrix"):
        for j in range(i + 1, N):
            sp_i = str(sites[i].specie)
            sp_j = str(sites[j].specie)
            options_i = species_dict.get(sp_i, [sp_i])
            options_j = species_dict.get(sp_j, [sp_j])
            dr_init = cart_coords[i] - cart_coords[j]
            dr_dm = distance_matrix[i][j]

            for ii, ei in zip(index_map[i], options_i):
                for jj, ej in zip(index_map[j], options_j):
                    pair_key1 = f"{ei}-{ej}"
                    pair_key2 = f"{ej}-{ei}"
                    key = pair_key1 if pair_key1 in buckingham_dict else pair_key2 if pair_key2 in buckingham_dict else None
                    if not key:
                        continue

                    if distance_analysis and dr_dm < distance_threshold:
                        buckingham_matrix_expanded[ii, jj] = 1e6
                    else:
                        for rnx in range(-nx, nx + 1):
                            for rny in range(-ny, ny + 1):
                                for rnz in range(-nz, nz + 1):
                                    shift = rnx * lattice_vectors[0] + rny * lattice_vectors[1] + rnz * lattice_vectors[2]
                                    dr = dr_init + shift
                                    dist = np.linalg.norm(dr)
                                    if dist < R_max:
                                        V = buckingham_potential(buckingham_dict[key], dist)
                                        buckingham_matrix_expanded[ii, jj] += V

    return buckingham_matrix_expanded, expanded_species


def compute_buckingham_matrix_discrete_fast(
    structure,
    species_dict,
    buckingham_dict,
    R_max,
    distance_analysis=False,
    distance_threshold=0.1,
):
    """
    Faster Buckingham matrix builder:
      - Uses get_points_in_sphere (spherical neighbor enumeration, no box scan)
      - Vectorizes over all periodic images for each (i,j)
      - Reuses sums per unique rho and a global sum of 1/r^6
    """

    sites = structure.sites
    N = len(sites)
    lat = structure.lattice
    fcoords = structure.frac_coords
    ccoords = structure.cart_coords  # only for initial neighbor seeds

    # ---- 0) Preprocess species options and index map (same as your code) ----
    index_map = {}
    expanded_species = []
    new_idx = 0
    for i, site in enumerate(sites):
        sp = str(site.specie)
        options = species_dict.get(sp, [sp])
        index_map[i] = list(range(new_idx, new_idx + len(options)))
        expanded_species.extend(options)
        new_idx += len(options)
    expanded_N = len(expanded_species)
    B = np.zeros((expanded_N, expanded_N), dtype=float)

    # ---- 1) Preprocess Buckingham parameters & unique rhos ----
    # Normalize keys to canonical "A-B" with A<=B
    def canon_pair(a, b):
        return (a, b) if a <= b else (b, a)

    params = {}       # (ei,ej) -> (A, rho, C)
    unique_rhos = set()
    for key, (A, rho, C) in buckingham_dict.items():
        Ael, Bel = key.replace(" ", "").split("-")
        k = canon_pair(Ael, Bel)
        params[k] = (float(A), float(rho), float(C))
        unique_rhos.add(float(rho))
    unique_rhos = sorted(unique_rhos)

    # ---- 2) Main loop: i<j, gather all periodic images within R_max once ----
    # For each (i,j), compute
    #   S_r6   = sum_k (1/r_k^6)
    #   S_exp[ρ] = sum_k exp(-r_k / ρ)  for each unique ρ
    # and then fill all option pairs via parameters lookup.
    for i in tqdm(range(N), desc="Buckingham (fast)"):
        ri = ccoords[i]
        spi = str(sites[i].specie)
        opts_i = species_dict.get(spi, [spi])
        inds_i = index_map[i]

        # neighbor search around ri using *fractional* seed list fcoords
        # get_points_in_sphere expects fractional list + cartesian center
        nf, dists, js, _imgs = lat.get_points_in_sphere(fcoords, ri, R_max, zip_results=False)

        # Strip self-image at r≈0
        mask = dists > 1e-12
        nf = nf[mask]; dists = dists[mask]; js = js[mask]
        if len(dists) == 0:
            continue

        # Precompute aggregates once per i for all j (we’ll mask per j below)
        # These are *per-(i,j)* actually, so we compute per j mask shortly.

        # Group contributions by j to avoid scanning full arrays for each rho repeatedly
        # Build an index list for each distinct j in neighbors
        js_unique, inv = np.unique(js, return_inverse=True)
        # For each block corresponding to a single j, precompute S_r6 and S_exp[ρ]
        for idx, j in enumerate(js_unique):
            if j <= i:
                continue  # keep upper triangle & avoid double counting
            block_mask = (inv == idx)
            dj = dists[block_mask]
            if dj.size == 0:
                continue

            # distance-based screening
            if distance_analysis and dj.min() < distance_threshold:
                # Apply huge penalty to all option pairs of (i,j)
                for ii in inds_i:
                    for jj in index_map[j]:
                        B[ii, jj] += 1e6
                        B[jj, ii] += 1e6
                continue

            # vectorized aggregates for this (i,j)
            inv_r6 = (dj**-6).sum()               # S_r6
            Sexp_by_rho = {rho: np.exp(-dj / rho).sum() for rho in unique_rhos}

            spj = str(sites[j].specie)
            opts_j = species_dict.get(spj, [spj])
            inds_j = index_map[j]

            # Fill contributions for all option pairs
            for ii, ei in zip(inds_i, opts_i):
                for jj, ej in zip(inds_j, opts_j):
                    A, rho, C = params.get(canon_pair(ei, ej), (None, None, None))
                    if A is None:
                        continue
                    Vij = A * Sexp_by_rho[rho] - C * inv_r6
                    B[ii, jj] += Vij
                    B[jj, ii] += Vij   # symmetric

    return B, expanded_species


def compute_buckingham_matrix_discrete_parallel(
    structure,
    species_dict,
    buckingham_dict,
    R_max,
    distance_analysis=False,
    distance_threshold=0.1,
):
    """
    Parallel fast Buckingham matrix builder.
    """

    sites = structure.sites
    N = len(sites)
    lat = structure.lattice
    fcoords = structure.frac_coords
    ccoords = structure.cart_coords

    # --- Species expansion
    index_map = {}
    expanded_species = []
    idx = 0
    for i, site in enumerate(sites):
        sp = str(site.specie)
        opts = species_dict.get(sp, [sp])
        index_map[i] = list(range(idx, idx + len(opts)))
        expanded_species.extend(opts)
        idx += len(opts)
    expanded_N = len(expanded_species)

    # --- Preprocess Buckingham params
    def canon_pair(a, b):
        return (a, b) if a <= b else (b, a)

    buckingham_params = {}
    unique_rhos = set()
    for key, (A, rho, C) in buckingham_dict.items():
        a, b = key.replace(" ", "").split("-")
        pair = canon_pair(a, b)
        buckingham_params[pair] = (float(A), float(rho), float(C))
        unique_rhos.add(float(rho))
    unique_rhos = sorted(unique_rhos)

    # --- Build work chunks
    ncpu = mp.cpu_count()
    chunk = (N + ncpu - 1) // ncpu
    tasks = []
    for k in range(ncpu):
        i_start = k * chunk
        i_end = min((k + 1) * chunk, N)
        if i_start >= i_end:
            continue

        args = (
            i_start, i_end,
            fcoords, ccoords, sites, lat,
            species_dict, buckingham_params, unique_rhos, R_max,
            distance_analysis, distance_threshold,
            index_map, expanded_N
        )
        tasks.append(args)

    # --- Run in parallel
    log(f"Launching {len(tasks)} parallel workers over {ncpu} CPUs...")
    with mp.Pool(len(tasks)) as pool:
        partial_results = list(
            tqdm(pool.imap(_buckingham_worker, tasks), total=len(tasks))
        )

    # --- Sum partial matrices
    B = np.sum(partial_results, axis=0)

    return B, expanded_species



def compute_discrete_ewald_matrix(structure, charge_options_by_Z, ewald_matrix):
    """
    Computes an expanded charge-weighted Ewald matrix for a pymatgen.Structure
    using a dictionary of charge options by atomic number.

    Parameters
    ----------
    structure : pymatgen.Structure
        The atomic structure.
    charge_options_by_Z : dict
        Dictionary mapping atomic numbers (Z) to lists of possible charges.
        Example: {26: [2, 3]} for Fe²⁺ and Fe³⁺.
    ewald_matrix : np.ndarray, optional
        Precomputed Ewald matrix. If None, will call compute_ewald_matrix(structure).

    Returns
    -------
    weighted_ewald : np.ndarray
        The Ewald matrix weighted by the outer product of the expanded charges.
    expanded_charges : np.ndarray
        1D array of charges including duplicated sites.
    expanded_ewald_matrix : np.ndarray
        Expanded Ewald matrix matching the length of `expanded_charges`.
    """

    num_sites = len(structure)
    atomic_numbers = structure.atomic_numbers

    total_new_sites = sum(
        (len(charge_options_by_Z[Z]) - 1) * np.sum(np.array(atomic_numbers) == Z)
        for Z in charge_options_by_Z
        if Z in atomic_numbers
    )

    expanded_N = num_sites + total_new_sites
    expanded_matrix = np.zeros((expanded_N, expanded_N))
    expanded_charges = []
    index_map = {}

    new_idx = 0
    for i, Z in enumerate(atomic_numbers):
        if Z in charge_options_by_Z:
            possible_charges = charge_options_by_Z[Z]
            index_map[i] = list(range(new_idx, new_idx + len(possible_charges)))
            expanded_charges.extend(possible_charges)
            new_idx += len(possible_charges)
        else:
            # Default to site charge if it exists, else zero
            try:
                default_charge = structure[i].properties.get("charge", 0)
            except AttributeError:
                default_charge = 0
            index_map[i] = [new_idx]
            expanded_charges.append(default_charge)
            new_idx += 1

    expanded_charges = np.array(expanded_charges)

    # Expand the Ewald matrix based on duplication map
    for i in range(num_sites):
        for j in range(num_sites):
            for ii in index_map[i]:
                for jj in index_map[j]:
                    expanded_matrix[ii, jj] = ewald_matrix[i, j]

    charge_matrix = np.outer(expanded_charges, expanded_charges)
    weighted_ewald = expanded_matrix * charge_matrix

    return weighted_ewald, expanded_charges, expanded_matrix


def compute_probability_grid(li_coords_all, M):
    """
    Compute a MxMxM grid of probabilities for the points in li_coords_all.
    
    Parameters:
    li_coords_all (ndarray): Nx3 array of fractional coordinates.
    M (int): Size of the grid along each dimension.
    
    Returns:
    ndarray: MxMxM array of probabilities.
    """
    # Initialize the grid
    grid = np.zeros((M, M, M))
    
    # Convert fractional coordinates to grid indices
    indices = (li_coords_all * M).astype(int)

    # Ensure indices are within bounds
    indices = np.clip(indices, 0, M-1)
    
    # Count the points in each grid cell
    for index in indices:
        grid[tuple(index)] += 1
    
    # Normalize the grid so that the sum of all values is 1
    total_points = len(li_coords_all)
    grid /= total_points
    
    return grid


def cpsat_core_from_indices(li_indices, mn_indices, N_li, proximity_groups=[]):
    model = cp_model.CpModel()

    # A) sites & options
    site_options, var2siteopt, li_sites, mn_sites = build_site_option_maps_from_indices(
        li_indices, mn_indices
    )

    # B) x vars + one-hot per site
    x = build_x_vars_and_onehot(model, site_options)

    # C) Li constraints
    add_li_mn_charge_balance_constraints(model, x, li_sites, mn_sites, N_li)
    add_li_proximity_exclusions(model, x, proximity_groups)

    # (No objective yet; we’ll add it when we map your pair energies W)
    return model, x, site_options, var2siteopt, li_sites, mn_sites


def find_fractional_centers(M):
    """
    Find the fractional coordinates of the centers of the grid cells.
    
    Parameters:
    M (int): Size of the grid along each dimension.
    
    Returns:
    ndarray: MxMxM array of fractional coordinates of the centers.
    """
    centers = np.zeros((M, M, M, 3))
    for i in range(M):
        for j in range(M):
            for k in range(M):
                centers[i, j, k] = [(i + 0.5) / M, (j + 0.5) / M, (k + 0.5) / M]
    return centers


def find_top_x_points(grid, centers, x):
    """
    Find the top x points in the grid in terms of probability.
    
    Parameters:
    grid (ndarray): MxMxM array of probabilities.
    centers (ndarray): MxMxM array of fractional coordinates of the centers.
    x (int): Number of top points to find.
    
    Returns:
    list: List of tuples (fractional_coordinate, probability) for the top x points.
    """
    # Flatten the grid and the coordinates
    flat_grid = grid.flatten()
    flat_centers = centers.reshape(-1, 3)
    
    # Get the indices of the top x values
    top_indices = np.argsort(flat_grid)[-x:]
    
    # Get the top x values and their corresponding fractional coordinates
    top_points = [(flat_centers[i], flat_grid[i]) for i in top_indices]
    
    # Sort the top points by probability in descending order
    top_points.sort(key=lambda x: x[1], reverse=True)
    
    
    return top_points


def generate_filtered_grid(structure, N_initial_grid=1000, min_dist_grid=1.5, num_points=None):
    lattice = structure.lattice.matrix         # 3x3 array
    cart_coords = structure.cart_coords        # (N_atoms, 3)

    if num_points is None:
        # Estimate the number of points per dimension
        volume = np.abs(np.linalg.det(lattice))
        spacing = (volume / N_initial_grid) ** (1/3)
        
        # Determine the number of grid points along each lattice vector
        lengths = np.linalg.norm(lattice, axis=1)
        num_points = np.maximum(np.round(lengths / spacing).astype(int), 1)
    else:
        num_points = np.array(num_points, dtype=int)
    
    # Create fractional grid
    x = np.linspace(0, 1, num_points[0], endpoint=False)
    y = np.linspace(0, 1, num_points[1], endpoint=False)
    z = np.linspace(0, 1, num_points[2], endpoint=False)
    grid_frac = np.array(np.meshgrid(x, y, z, indexing='ij')).reshape(3, -1).T  # (M, 3)

    # Convert to Cartesian coordinates
    grid_cart = grid_frac @ lattice  # (M, 3)

    # Remove grid points that are too close to any atom
    tree = cKDTree(cart_coords)
    distances, _ = tree.query(grid_cart, k=1)
    mask = distances > min_dist_grid
    filtered_grid = grid_cart[mask]

    return filtered_grid


def init_run_store(
    output_dir: str,
    initial_structure,                 # pymatgen.Structure (framework, no Li)
    li_sites: list,                    # CP site IDs for Li grid
    mn_sites: list,                    # CP site IDs for Mn
    initial_grid_cart: np.ndarray,     # (M,3) Li grid in CARTESIAN
    mn_atom_indices: list,             # len == len(mn_sites); atom indices in initial_structure
    QUBO_ut: np.ndarray,               # upper-triangular QUBO matrix (n x n)
    SCALE: int,                        # integer scaling used in objective
    solver_params: dict,               # e.g. {"time":180,"workers":8,"seed":42}
    extra_meta: dict = None,           # optional extra info
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "energy_model"), exist_ok=True)

    # --- Geometry data ---
    lat = initial_structure.lattice.matrix.astype(np.float32)
    species_Z = np.array([sp.Z for sp in initial_structure.species], dtype=np.int16)
    frac_coords = initial_structure.frac_coords.astype(np.float32)
    li_grid_frac = initial_structure.lattice.get_fractional_coords(initial_grid_cart).astype(np.float32)

    # Save geometry
    np.savez_compressed(
        os.path.join(output_dir, "geometry.npz"),
        lattice=lat,
        species_Z=species_Z,
        frac_coords=frac_coords,
        li_grid_frac=li_grid_frac,
        mn_atom_indices=np.array(mn_atom_indices, dtype=np.int32),
    )

    # Save mapping
    with open(os.path.join(output_dir, "mapping.json"), "w") as f:
        json.dump(
            {"li_sites": list(map(int, li_sites)), "mn_sites": list(map(int, mn_sites))},
            f,
            indent=2,
        )

    # Save QUBO model
    np.savez_compressed(
        os.path.join(output_dir, "energy_model", "qubo_ut.npz"),
        Q_ut=QUBO_ut.astype(np.float32),
        SCALE=int(SCALE),
    )

    # --- Hashes for provenance ---
    geom_hash = _sha256_of_arrays(lat, species_Z, frac_coords, li_grid_frac)
    qubo_hash = _sha256_of_arrays(QUBO_ut)

    # --- Meta info ---
    meta = {
        "run_id": os.path.basename(output_dir),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "solver_params": solver_params,
        "SCALE": int(SCALE),
        "geom_hash": geom_hash,
        "qubo_hash": qubo_hash,
        "files": {
            "geometry": "geometry.npz",
            "mapping": "mapping.json",
            "qubo": "energy_model/qubo_ut.npz",
            "incumbents": "incumbents.jsonl.gz",
        },
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {"geom_hash": geom_hash, "qubo_hash": qubo_hash}


def iter_incumbent_records(output_dir, *, keep_final_only=False, dedup_cfg=True):
    """
    Yields incumbent dicts from incumbents.jsonl.gz.
    - keep_final_only=True: only yield those with tags.status == 'FINAL'
    - dedup_cfg=True: skip duplicates by 'cfg' hash
    """
    inc_path = os.path.join(output_dir, "incumbents.jsonl.gz")
    seen = set()
    with gzip.open(inc_path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            if keep_final_only and not (rec.get("tags", {}).get("status") == "FINAL"):
                continue
            if dedup_cfg:
                cfg = rec.get("cfg")
                if cfg in seen:
                    continue
                seen.add(cfg)
            yield rec


def join_structure_grid(structure,initial_grid):

    initial_grid_pmg = Structure(structure.lattice,[3]*len(initial_grid),initial_grid,coords_are_cartesian=True)
    
    #### TMP
    atoms = AseAtomsAdaptor().get_atoms(initial_grid_pmg)
    view(atoms)
    # vview(structure)

    return Structure.from_sites(initial_grid_pmg.sites+structure.sites)


def load_run_assets(output_dir):
    """
    Returns:
      lattice: Lattice
      base_struct: Structure (framework only, no Li; all Mn as Mn element)
      li_sites: list[int]
      mn_sites: list[int]
      li_grid_frac: (M,3) np.ndarray fractional coords aligned with li_sites
      mn_atom_indices: list[int] atom indices in base_struct aligned with mn_sites
    """
    geom = np.load(os.path.join(output_dir, "geometry.npz"))
    lat_mat = geom["lattice"]
    species_Z = geom["species_Z"]
    frac_coords = geom["frac_coords"]
    li_grid_frac = geom["li_grid_frac"]
    mn_atom_indices = geom["mn_atom_indices"].tolist()

    with open(os.path.join(output_dir, "mapping.json"), "r") as f:
        mapping = json.load(f)
    li_sites = mapping["li_sites"]
    mn_sites = mapping["mn_sites"]

    lattice = Lattice(lat_mat)
    species = [Element.from_Z(int(z)) for z in species_Z]
    base_struct = Structure(lattice, species, frac_coords, coords_are_cartesian=False)
    return lattice, base_struct, li_sites, mn_sites, li_grid_frac, mn_atom_indices


def make_output_dir(base="runs", prefix="out", fixed_name=None):
    """
    Create an output directory:
      - if fixed_name is provided, use base/fixed_name (create if missing)
        (if it exists, append a timestamp suffix)
      - otherwise, create a unique timestamped dir under base with prefix.
    Example when auto: runs/out_20251112_153042_001
    """
    os.makedirs(base, exist_ok=True)
    if fixed_name:
        path = os.path.join(base, fixed_name)
        if os.path.exists(path):
            stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            suffix = str(int(time.time() * 1000) % 1000).zfill(3)
            path = os.path.join(base, f"{fixed_name}_{stamp}_{suffix}")
        os.makedirs(path, exist_ok=True)
        return path
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    suffix = str(int(time.time() * 1000) % 1000).zfill(3)
    name = f"{prefix}_{stamp}_{suffix}"
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_qubo_var_mapping(output_dir, var2siteopt, li_sites, mn_sites, filename="energy_model/var_mapping.json"):
    """
    Persist the QUBO variable mapping so downstream analysis can reconstruct
    which columns correspond to Li or Mn options without re-running helpers.
    """
    os.makedirs(os.path.join(output_dir, "energy_model"), exist_ok=True)
    li_set = set(int(s) for s in li_sites)
    mn_set = set(int(s) for s in mn_sites)
    records = []
    for idx in sorted(var2siteopt.keys()):
        site_id, option = var2siteopt[idx]
        site_id = int(site_id)
        entry = {
            "var_index": int(idx),
            "site_id": site_id,
            "option": option,
            "site_type": "Li" if site_id in li_set else "Mn" if site_id in mn_set else "Other",
        }
        records.append(entry)
    payload = {
        "records": records,
        "li_sites": sorted(li_set),
        "mn_sites": sorted(mn_set),
    }
    with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_energy_components(output_dir, ewald_matrix, buckingham_matrix, filename="energy_model/energy_terms.npz"):
    """
    Save the decomposed energy matrices (Ewald + Buckingham) for debugging.
    """
    os.makedirs(os.path.join(output_dir, "energy_model"), exist_ok=True)
    payload = {}
    if ewald_matrix is not None:
        payload["ewald_discrete"] = np.asarray(ewald_matrix, dtype=np.float32)
    if buckingham_matrix is not None:
        payload["buckingham_discrete"] = np.asarray(buckingham_matrix, dtype=np.float32)
    if not payload:
        return
    np.savez_compressed(os.path.join(output_dir, filename), **payload)


def parse_gulp_to_pymatgen(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    energy_final = None
    energy_initial = None

    # First occurrence (top) → treat as initial
    for line in lines:
        if "Total lattice energy " in line and " eV " in line:
            try:
                energy_initial = float(line.split()[-2])
                break
            except Exception:
                energy_initial = None
                break

    # Last occurrence (bottom) → final
    for line in lines[::-1]:
        if "Total lattice energy " in line and " eV " in line:
            try:
                energy_final = float(line.split()[-2])
            except Exception:
                energy_final = None
            break

    # Number of atoms (if available)
    n_atoms = None
    for line in lines:
        if "Total number atoms/shells" in line:
            try:
                n_atoms = int(line.strip().split()[-1])
            except Exception:
                n_atoms = None
            break

    # --- 1) Lattice parameters ---
    a = b = c = alpha = beta = gamma = None
    for line in lines:
        if "a =" in line and "alpha" in line:
            parts = line.strip().split()
            a = float(parts[2])
            alpha = float(parts[5])
        elif "b =" in line and "beta" in line:
            parts = line.strip().split()
            b = float(parts[2])
            beta = float(parts[5])
        elif "c =" in line and "gamma" in line:
            parts = line.strip().split()
            c = float(parts[2])
            gamma = float(parts[5])
            break

    if None in (a, b, c, alpha, beta, gamma):
        raise ValueError("Could not parse complete lattice parameters.")

    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    # --- 2) Final fractional coordinates ---
    species = []
    coords = []
    in_block = False
    for line in lines:
        if "Final fractional coordinates of atoms" in line:
            in_block = True
            continue
        if in_block:
            if not line.strip():
                if n_atoms is None:
                    break
                continue
            tokens = line.split()
            # lines are usually: index, label, ..., fracx fracy fracz ...
            if tokens and tokens[0].isdigit():
                try:
                    sym = tokens[1]
                    fx, fy, fz = map(float, tokens[-4:-1])
                    species.append(sym)
                    coords.append([fx, fy, fz])
                    if n_atoms is not None and len(species) >= n_atoms:
                        break
                except Exception:
                    # stop if format stops matching atoms
                    break
            else:
                # reached footer of the block
                if n_atoms is None:
                    break
                # otherwise keep scanning until we collected n_atoms

    if not species:
        raise ValueError("No atomic coordinates parsed from GULP output.")

    structure = Structure(lattice, species, coords, coords_are_cartesian=False)
    structure.translate_sites(np.arange(structure.num_sites), [1, 1, 1], to_unit_cell=True)

    return structure, energy_final, energy_initial


def read_opt_structures(folder_path, input_name='gulp_klmc.gin', max_structures=None):
    """
    Scan a GULP result directory and parse up to `max_structures` optimized outputs.
    Results are expected under subdirectories named A0, A1, ...
    """
    base = Path(folder_path)
    if not base.exists():
        log(f"GULP result folder missing: {base}")
        return [], []
    output_name = input_name[:-3] + 'gout'
    entries = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("A")]

    def sort_key(p):
        suf = p.name[1:]
        return int(suf) if suf.isdigit() else float("inf")

    entries.sort(key=sort_key)
    if max_structures is not None:
        entries = entries[:max_structures]

    all_opt_structures = []
    all_opt_energy = []
    all_opt_energy_init = []
    for subdir in entries:
        file_path = subdir / output_name
        log(str(file_path))
        if file_path.exists():
            try:
                structure, energy_final, energy_initial = parse_gulp_to_pymatgen(str(file_path))
                all_opt_structures.append(structure)
                all_opt_energy.append(energy_final)
                all_opt_energy_init.append(energy_initial)
            except Exception as exc:
                log(f"[warn] Failed to parse {file_path}: {exc}")
                continue

    return all_opt_structures, all_opt_energy, all_opt_energy_init


def reduce_qubo_discrete_limno(Q_discrete, species_vector):
    # Convert species to atomic numbers
    atomic_number_vector = np.array([Element(sym).Z for sym in species_vector])

    # Identify indices
    oxygen_positions = np.where(atomic_number_vector == 8)[0]
    other_element_positions = np.where(atomic_number_vector != 8)[0]

    oo_energy = np.sum(Q_discrete[np.ix_(oxygen_positions, oxygen_positions)])

    # Make a copy of the matrix
    Q = copy.deepcopy(Q_discrete)

    # Transfer O-X interaction into X-X diagonal
    for i in other_element_positions:
        for j in oxygen_positions:
            Q[i][i] += Q[i][j]

    # Remove rows and columns corresponding to O atoms
    Q_reduced = np.delete(Q, oxygen_positions, axis=0)
    Q_reduced = np.delete(Q_reduced, oxygen_positions, axis=1)

    return Q_reduced, oo_energy


def structure_from_incumbent_record(
    record: dict,
    base_struct: Structure,
    li_sites: list[int],
    mn_sites: list[int],
    li_grid_frac: np.ndarray,
    mn_atom_indices: list[int],
    *,
    encode_mn_as_tc: bool = True,   # match your GULP script (Mn3 -> 'Tc')
    set_oxidation: bool = False      # alternative: Mn3/4 as oxidation states on Mn
) -> Structure:
    """
    record is a dict from incumbents.jsonl.gz with keys: li_on, mn3_on, ...
    Returns a new Structure with Li added and Mn adjusted (either Tc element or oxidation).
    """
    struct = base_struct.copy()

    # 1) Adjust Mn sites: mn_sites[i] -> atom index mn_atom_indices[i]
    mn3_set = set(record.get("mn3_on", []))
    for i, s in enumerate(mn_sites):
        atom_idx = mn_atom_indices[i]
        if s in mn3_set:
            if encode_mn_as_tc:
                struct.replace(atom_idx, Element("Tc"))  # your pipeline expects Tc for Mn3+
            elif set_oxidation:
                struct.replace(atom_idx, Species("Mn", 3))
        else:
            if set_oxidation:
                struct.replace(atom_idx, Species("Mn", 4))
            # else: leave as elemental Mn

    # 2) Add Li atoms at selected grid sites
    li_pos = {s: i for i, s in enumerate(li_sites)}
    for s in record.get("li_on", []):
        gi = li_pos[s]
        frac = li_grid_frac[gi]
        struct.append("Li", frac, coords_are_cartesian=False)

    return struct


def structure_hash_pbc(pmg_structure, tol_frac=1e-4, tol_lat=1e-3):
    """
    Order- and translation-invariant hash:
      - lattice matrix (rounded)
      - sorted list of (symbol, rounded fractional coords)
    """
    lat = pmg_structure.lattice.matrix
    lat_key = tuple(_round_array(lat, tol_lat).flatten())

    fracs = pmg_structure.frac_coords % 1.0
    fracs_key = _round_array(fracs, tol_frac)
    syms = [str(sp) for sp in pmg_structure.species]
    rows = sorted((syms[i], fracs_key[i,0], fracs_key[i,1], fracs_key[i,2]) for i in range(len(syms)))

    return hash((lat_key, tuple(rows), len(syms)))


def unwrap_frac_coords(frac_coords_list):
    """
    Aligns fractional coordinates modulo lattice to a reference so that they 
    can be meaningfully averaged. Assumes all coords_list[i] are (N_atoms, 3).
    """
    # Use the first set of coordinates as reference
    ref = frac_coords_list[0]
    unwrapped_list = []

    for coords in frac_coords_list:
        delta = coords - ref
        # Apply minimum image convention
        delta -= np.round(delta)
        aligned = ref + delta
        unwrapped_list.append(aligned)

    return np.array(unwrapped_list)


def write_gulp_input(structure, filename="gulp_input.gin"):
    with open(filename, "w") as f:
        f.write("sp opti fbfgs conp property\n")
        f.write("vectors\n")
        for line in structure.lattice.matrix:
            f.write(" ".join([f"{x:.6f}" for x in line]) + "\n")
        
        f.write("0 0 0 0 0 0\n")
        f.write("cartesian\n")
        
        for an, line in zip(structure.atomic_numbers, structure.cart_coords):
            symbol = Element.from_Z(an).symbol
            f.write(f"{symbol} core {line[0]:.6f} {line[1]:.6f} {line[2]:.6f}\n")
        
        f.write("\nspecies\n")
        f.write("Mn     core    4.000000\n")
        f.write("Tc     core    3.000000\n")
        f.write("Li     core    1.000000\n")
        f.write("O      core   -2.000000\n")
        
        f.write("buck\n")
        f.write("Li core O core 426.480     0.3000     0.00 0.00 25.00\n")
        f.write("Mn core O core 3087.826    0.2642     0.00 0.00 25.00\n")
        f.write("Tc core O core 1686.125    0.2962     0.00 0.00 25.00\n")
        f.write("O  core O core 22.410      0.6937    32.32 0.000 25.00\n")
        #######   OPTIONS



def write_gulp_inputs_from_incumbents(
        output_dir: str,
        dest_dir: str,
        *,
        limit: int | None = None,
        keep_final_only: bool = False,
        filename_pattern: str = "A{idx}.gin",
        encode_mn_as_tc: bool = True,
        set_oxidation: bool = False,
        write_gulp_input_fn=None,   # pass your write_gulp_input if not global
        # --- new options for batch files ---
        write_taskfarm: bool = False,
        write_slurm: bool = False,
        job_name: str = "gulp_run",
        account: str = "e05-algor-smw",
        partition: str = "standard",
        qos: str = "short",
        exe_path: str = "/work/e05/e05/bcamino/klmc_exe/klmc3.062024.x",
        ntasks: int = 128,                 # total MPI ranks
        ntasks_per_node: int = 128,        # ranks per node
        cpus_per_task: int = 1,
        dedup_by_hash: bool = True,
    ):
        """
        Rebuild structures for incumbents and write GULP .gin files.
        Also writes:
        - taskfarm.config (task_start 0, task_end N-1)
        - SLURM_js.slurm (submission script)
        """
        from pathlib import Path

       

        os.makedirs(dest_dir, exist_ok=True)
        write_gulp_input = None
        if write_gulp_input_fn is not None:
            write_gulp_input = write_gulp_input_fn
        elif "write_gulp_input" in globals():
            write_gulp_input = globals()["write_gulp_input"]
        elif _default_write_gulp_input is not None:
            write_gulp_input = _default_write_gulp_input
        if write_gulp_input is None:
            raise RuntimeError("write_gulp_input function is not available; pass write_gulp_input_fn or ensure full_script_functions is importable.")
        count = 0
        # load run assets
        lattice, base_struct, li_sites, mn_sites, li_grid_frac, mn_atom_indices = load_run_assets(output_dir)

        # write all incumbents (or FINAL only) as gin files
        for idx, rec in enumerate(iter_incumbent_records(output_dir, keep_final_only=keep_final_only, dedup_cfg=dedup_by_hash)):
            struct = structure_from_incumbent_record(
                rec, base_struct, li_sites, mn_sites, li_grid_frac, mn_atom_indices,
                encode_mn_as_tc=encode_mn_as_tc, set_oxidation=set_oxidation
            )
            name = filename_pattern.format(idx=idx, cfg=rec.get("cfg", f"{idx:05d}"))
            path = Path(dest_dir) / name
            write_gulp_input(structure=struct, filename=str(path))
            count += 1
            if limit is not None and count >= limit:
                break

        # Nothing to schedule -> return early
        if count == 0:
            # still write empty taskfarm if requested (task_end -1 is odd, so skip)
            return 0
        if count<128:
            ntasks = count
            ntasks_per_node = count
        # ----- Write taskfarm.config -----
        if write_taskfarm:
            tf_path = Path(dest_dir) / "taskfarm.config"
            with open(tf_path, "w") as f:
                f.write("task_start 0\n")
                f.write(f"task_end {count-1}\n")
                f.write("cpus_per_worker 1\n")
                f.write("application gulp\n")

        # ----- Write SLURM_js.slurm -----
        if write_slurm:
            slurm_path = Path(dest_dir) / "SLURM_js.slurm"
            actual_ntasks = ntasks if count >= ntasks else count
            script = textwrap.dedent(f"""\
                #!/bin/bash

                #SBATCH --job-name={job_name}
                #SBATCH --time=00:20:00
                #SBATCH --nodes=1
                #SBATCH --account={account}
                #SBATCH --partition={partition}
                #SBATCH --qos={qos}

                export OMP_NUM_THREADS=1

                EXE="{exe_path}"
                srun -n {actual_ntasks} --ntasks-per-node={ntasks_per_node} --cpus-per-task={cpus_per_task} --distribution=block:block --hint=nomultithread --exact ${{EXE}} 1> stdout 2> stderr

                mkdir -p result
                mv A* ./result 2>/dev/null || true

                mkdir -p log
                mv master.log ./log 2>/dev/null || true
                mv workgroup*.log ./log 2>/dev/null || true
                """)
            with open(slurm_path, "w") as f:
                f.write(script)

        return count


def rebuild_incumbent_structures(
    output_dir: str,
    *,
    keep_final_only: bool = False,
    encode_mn_as_tc: bool = True,
    set_oxidation: bool = False,
):
    """Rebuild structures from incumbents.jsonl.gz. Returns (structures, energies)."""
    inc_path = os.path.join(output_dir, "incumbents.jsonl.gz")
    if not os.path.exists(inc_path):
        return [], []

    lattice, base_struct, li_sites, mn_sites, li_grid_frac, mn_atom_indices = load_run_assets(output_dir)
    structures = []
    energies = []
    for rec in iter_incumbent_records(output_dir, keep_final_only=keep_final_only, dedup_cfg=True):
        struct = structure_from_incumbent_record(
            rec, base_struct, li_sites, mn_sites, li_grid_frac, mn_atom_indices,
            encode_mn_as_tc=encode_mn_as_tc, set_oxidation=set_oxidation
        )
        structures.append(struct)
        energies.append(rec.get("E"))
    return structures, energies


def write_incumbent_structures_extxyz(
    output_dir: str,
    dest_extxyz: str,
    *,
    keep_final_only: bool = False,
    encode_mn_as_tc: bool = True,
    set_oxidation: bool = False,
    tol_frac: float = 1e-4,
    tol_lat: float = 1e-3,
):
    """Rebuild structures and append to extxyz. Returns added count."""
    structures, energies = rebuild_incumbent_structures(
        output_dir,
        keep_final_only=keep_final_only,
        encode_mn_as_tc=encode_mn_as_tc,
        set_oxidation=set_oxidation,
    )
    if not structures:
        return 0
    added, _, _ = append_unique_extxyz(dest_extxyz, structures, energies, tol_frac=tol_frac, tol_lat=tol_lat)
    return added


def split_template_head_tail(template_lines):
    """
    Split a GULP input template into (head, tail) around the geometry block.
    We look for a start marker ('cell' or 'cartesian') and an end marker ('species' or 'totalenergy').
    Head includes the start marker line; tail starts at the end marker line.
    """
    start_idx = None
    end_idx = None
    for i, line in enumerate(template_lines):
        l = line.strip().lower()
        if "cell" in l or "cartesian" in l:
            start_idx = i
            break
    if start_idx is None:
        start_idx = 0
    for j in range(start_idx + 1, len(template_lines)):
        l = template_lines[j].strip().lower()
        if "species" in l or "totalenergy" in l:
            end_idx = j
            break
    if end_idx is None:
        end_idx = len(template_lines)
    head = template_lines[: start_idx + 1]
    tail = template_lines[end_idx:]
    return head, tail


def get_head_tail(template_lines, head_override=None, tail_override=None):
    """
    Return (head, tail) using overrides if provided; otherwise split template_lines.
    """
    if head_override is not None or tail_override is not None:
        head = head_override if head_override is not None else []
        tail = tail_override if tail_override is not None else []
        return head, tail
    return split_template_head_tail(template_lines)


def extract_geom_from_res(res_path):
    """
    Extract geometry lines from gulp.res between 'cell' and 'totalenergy' or 'species'.
    Returns list of lines (with trailing newlines).
    """
    paths_to_try = []
    if os.path.exists(res_path):
        paths_to_try.append(res_path)
    # Also search recursively under the parent for any gulp.res
    base = Path(res_path).parent
    for p in base.rglob("gulp.res"):
        if p.as_posix() != res_path:
            paths_to_try.append(str(p))
    for p in paths_to_try:
        try:
            with open(p, "r") as fh:
                lines = fh.readlines()
        except Exception:
            continue
        if not lines:
            continue
        break
    else:
        return []
    start = None
    stop = None
    for i, line in enumerate(lines):
        if "cell" in line.lower():
            start = i
            break
    if start is None:
        return []
    for j in range(start + 1, len(lines)):
        l = lines[j].lower()
        if "totalenergy" in l or "species" in l:
            stop = j
            break
    if stop is None:
        stop = len(lines)
    return lines[start + 1 : stop]


def write_gulp_batch_scripts(
    dest_dir: str,
    count: int,
    *,
    job_name: str = "gulp_run",
    account: str = "e05-algor-smw",
    partition: str = "standard",
    qos: str = "short",
    exe_path: str = "/work/e05/e05/bcamino/klmc_exe/klmc3.062024.x",
    ntasks: int = 128,
    ntasks_per_node: int = 128,
    cpus_per_task: int = 1,
    work_subdir: str | None = None,
):
    """
    Write taskfarm.config and SLURM_js.slurm into dest_dir for a batch of size `count`.
    If `work_subdir` is provided, the SLURM script will `cd` into that subdirectory
    before running GULP (useful when gin files live in dest_dir/work_subdir).
    """
    from pathlib import Path

    if count <= 0:
        return 0
    os.makedirs(dest_dir, exist_ok=True)
    if count < ntasks:
        ntasks = count
        ntasks_per_node = count

    tf_path = Path(dest_dir) / "taskfarm.config"
    with open(tf_path, "w") as f:
        f.write("task_start 0\n")
        f.write(f"task_end {count-1}\n")
        f.write("cpus_per_worker 1\n")
        f.write("application gulp\n")

    slurm_path = Path(dest_dir) / "SLURM_js.slurm"
    script = textwrap.dedent(f"""\
        #!/bin/bash

        #SBATCH --job-name={job_name}
        #SBATCH --time=00:20:00
        #SBATCH --nodes=1
        #SBATCH --account={account}
        #SBATCH --partition={partition}
        #SBATCH --qos={qos}

        export OMP_NUM_THREADS=1

        EXE="{exe_path}"
        WORKDIR="{work_subdir or '.'}"

        cd "$WORKDIR"

        srun -n {ntasks} --ntasks-per-node={ntasks_per_node} --cpus-per-task={cpus_per_task} --distribution=block:block --hint=nomultithread --exact ${{EXE}} 1> stdout 2> stderr

        mkdir -p result
        mv A* ./result 2>/dev/null || true

        mkdir -p log
        mv master.log ./log 2>/dev/null || true
        mv workgroup*.log ./log 2>/dev/null || true
        """)
    with open(slurm_path, "w") as f:
        f.write(script)

    return count

def _bron_kerbosch(R, P, X, adj, cliques, max_nodes=1000):
    """Simple Bron–Kerbosch to enumerate maximal cliques (no pivot)."""
    # small safeguard
    if len(cliques) > max_nodes:
        return
    if not P and not X:
        if len(R) >= 2:
            cliques.append(tuple(sorted(R)))
        return
    # iterate over a copy since P will mutate
    for v in list(P):
        _bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, cliques, max_nodes)
        P.remove(v)
        X.add(v)


def _buckingham_worker(args):
    (i_start, i_end,
     fcoords, ccoords, sites, lat,
     species_dict, buckingham_params, unique_rhos, R_max,
     distance_analysis, distance_threshold,
     index_map, expanded_N) = args

    B_local = np.zeros((expanded_N, expanded_N), dtype=float)

    # Helper to canonicalize pair keys
    def canon_pair(a, b):
        return (a, b) if a <= b else (b, a)

    for i in range(i_start, i_end):
        ri = ccoords[i]
        spi = str(sites[i].specie)
        opts_i = species_dict.get(spi, [spi])
        inds_i = index_map[i]

        # Get periodic neighbors within cutoff
        nf, dists, js, _ = lat.get_points_in_sphere(
            fcoords, ri, R_max, zip_results=False
        )

        mask = dists > 1e-12
        dists = dists[mask]
        js = js[mask]

        if dists.size == 0:
            continue

        # Group neighbors by j
        js_unique, inv = np.unique(js, return_inverse=True)

        for group_idx, j in enumerate(js_unique):
            if j <= i:
                continue

            block_mask = (inv == group_idx)
            dj = dists[block_mask]
            if dj.size == 0:
                continue

            # Safety check
            if distance_analysis and dj.min() < distance_threshold:
                for ii in inds_i:
                    for jj in index_map[j]:
                        B_local[ii, jj] += 1e6
                        B_local[jj, ii] += 1e6
                continue

            # Per-(i,j) vectorized terms
            inv_r6 = (dj**-6).sum()
            Sexp = {rho: np.exp(-dj / rho).sum() for rho in unique_rhos}

            spj = str(sites[j].specie)
            opts_j = species_dict.get(spj, [spj])
            inds_j = index_map[j]

            for ii, ei in zip(inds_i, opts_i):
                for jj, ej in zip(inds_j, opts_j):
                    key = canon_pair(ei, ej)
                    if key not in buckingham_params:
                        continue
                    A, rho, C = buckingham_params[key]

                    Vij = A * Sexp[rho] - C * inv_r6
                    B_local[ii, jj] += Vij
                    B_local[jj, ii] += Vij

    return B_local


def _frac_coords(coords_cart, lattice):
    """Convert cartesian coords to fractional with given 3x3 lattice matrix."""
    return np.dot(coords_cart, np.linalg.inv(lattice).T)


def _maximal_cliques_from_edges(N, edges, cap_cliques=10000):
    """Return list of maximal cliques (each a tuple of node indices)."""
    # adjacency as sets
    adj = [set() for _ in range(N)]
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    cliques = []
    _bron_kerbosch(set(), set(range(N)), set(), adj, cliques, max_nodes=cap_cliques)
    return cliques


def _mic_delta_frac(df):
    """Minimum-image wrap for fractional deltas to (-0.5, 0.5]."""
    return df - np.round(df)


def _pair_edges_with_threshold(frac_coords, lattice, threshold):
    """
    Build edges (i,j) where PBC distance < threshold.
    frac_coords: (N,3) fractional coords in [0,1)
    lattice: 3x3 cartesian lattice matrix
    """
    N = len(frac_coords)
    edges = []
    L = np.asarray(lattice)
    for i in range(N):
        for j in range(i+1, N):
            df = _mic_delta_frac(frac_coords[j] - frac_coords[i])
            dcart = df @ L
            if np.linalg.norm(dcart) < threshold:
                edges.append((i, j))
    return edges


def _round_array(a, tol):
    return np.round(a / tol).astype(np.int64)


def _sha256_of_arrays(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]  # short hash for filenames/IDs







# -------------------------------------------------------------------
# MAIN PARALLEL FUNCTION
# -------------------------------------------------------------------




# ROOT RUN FOLDER
folder_path_root = os.path.abspath(make_output_dir(base="runs", prefix="out", fixed_name=RUN_NAME))

for N_li in LI_SWEEP:
    # ----------------------------------------------------------------
    # Setup per N_li: structure, grid, logging
    # ----------------------------------------------------------------
    initial_structure = Structure.from_file(initial_structure_path)

    # Allow N_initial_grid=False to keep all candidate points (no truncation)
    grid_seed = N_initial_grid if N_initial_grid else 1000  # used for density/spacing
    n_model_eff = N_initial_grid if N_initial_grid else None
    init_dense = max(grid_seed, int(grid_seed * GRID_INIT_FPS_MULTIPLIER))
    initial_grid, initial_grid_explore = initialize_li_grids(
        initial_structure,
        n_model=n_model_eff,
        n_dense=init_dense,
        min_dist=min_dist_grid,
        mode=GRID_INIT_MODE,
        seed=GRID_RANDOM_SEED,
        use_dual=GRID_USE_DUAL_GRID,
        voronoi_shells=GRID_VORONOI_SHELLS,
    )

    structure = join_structure_grid(initial_structure, initial_grid)
    grid = copy.deepcopy(initial_grid)
    grid_explore = copy.deepcopy(initial_grid_explore)

    grid_decay_records = []
    if GRID_USE_DECAY:
        grid_decay_records = [
            {"coord": np.array(coord, dtype=float), "ttl": GRID_DECAY_TTL}
            for coord in grid
        ]

    grid_tracker = (
        init_grid_tracker() if GRID_USE_TRACKER or GRID_UPDATE_MODE == "cluster" else None
    )

    # Per-N_li folder
    folder_path = os.path.join(folder_path_root, f"{N_li}_Li")
    os.makedirs(folder_path, exist_ok=True)

    RUN_LOGGER = RunLogger(folder_path)
    log(f"Run log file: {RUN_LOGGER.path}")

    # Grid snapshots (optional, as in your original)
    grid_extxyz_opt = os.path.join(folder_path, "all_optimized_grid.extxyz")

    # Where best pre-GULP structures will be accumulated (for this N_li)
    before_path = os.path.join(folder_path, "all_before_optimized.extxyz")

    # GULP folder where we will later write A0.gin, A1.gin, ...
    gulp_root_dir = os.path.join(folder_path, "all_gulp")
    os.makedirs(gulp_root_dir, exist_ok=True)

    # Ensure we start fresh per N_li if you want a clean all_before_optimized
    if os.path.exists(before_path):
        os.remove(before_path)

    # ----------------------------------------------------------------
    # Main loop over iterations for this N_li
    # ----------------------------------------------------------------
    for i in range(num_iterations):
        log(f"************ Begin Iteration {i} (N_li = {N_li}) ************")

        output_dir = os.path.join(folder_path, f"output_folder_{i}")
        os.makedirs(output_dir, exist_ok=True)

        # Save current grid snapshot
        append_grid_extxyz(
            grid_extxyz_opt, initial_structure.lattice, grid, tag=f"opt_iter_{i}"
        )

        print(structure)

        # --- Build energy matrix (QUBO) and mapping ---
        QUBO, li_indices, mn_indices, ewald_discrete, buckingham_discrete = build_QUBO(
            structure,
            threshold_li=threshold_li,
            prox_penalty=prox_penalty,
        )

        # CP-SAT core (one-hot, counts, charge balance)
        model, x, site_options, var2siteopt, li_sites, mn_sites = cpsat_core_from_indices(
            li_indices, mn_indices, N_li=N_li
        )

        li_coords_cart = extract_li_cartesian_coords(structure)
        if len(li_sites) != len(li_coords_cart):
            raise ValueError(
                f"Li site/count mismatch before proximity exclusions: "
                f"{len(li_sites)} site IDs vs {len(li_coords_cart)} grid coords."
            )

        groups, pairs = build_li_proximity_groups(
            li_grid_coords=li_coords_cart,          # Li positions aligned with CP sites
            threshold_ang=LI_LI_EXCLUSION_ANG,
            lattice=structure.lattice,              # PBC-aware distances
            coords_are_cartesian=True,
            site_ids=li_sites,                      # CP site IDs aligned to grid order
        )
        add_li_proximity_exclusions(model, x, groups)
        log(f"Li-Li proximity exclusions: {len(groups)} groups at {LI_LI_EXCLUSION_ANG} Å")

        # Objective from upper-triangular QUBO
        SCALE, info = add_ut_qubo_objective(model, x, var2siteopt, QUBO, scale=SOLVER_SCALE)
        log(f"Objective wiring: {info}")

        # --- Mn atom indices (Z=25). Align to mn_sites order if needed ---
        mn_atom_indices_all = np.where(
            np.array(initial_structure.atomic_numbers) == 25
        )[0]
        assert len(mn_atom_indices_all) >= len(mn_sites), (
            "Not enough Mn atoms in initial_structure to map mn_sites."
        )
        mn_atom_indices = list(mn_atom_indices_all[: len(mn_sites)])

        # --- Save run-level artifacts once per iteration ---
        solver_params = {"time": max_time, "workers": CP_SAT_WORKERS, "seed": SOLVER_SEED}
        _ = init_run_store(
            output_dir=output_dir,
            initial_structure=initial_structure,
            li_sites=li_sites,
            mn_sites=mn_sites,
            initial_grid_cart=grid,
            mn_atom_indices=mn_atom_indices,
            QUBO_ut=QUBO,
            SCALE=SCALE,
            solver_params=solver_params,
        )
        save_qubo_var_mapping(output_dir, var2siteopt, li_sites, mn_sites)
        save_energy_components(output_dir, ewald_discrete, buckingham_discrete)

        # --- Solver setup: ONLY care about FINAL BEST solution ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max_time
        solver.parameters.num_search_workers = CP_SAT_WORKERS
        if solver_params["seed"] is not None:
            solver.parameters.random_seed = solver_params["seed"]
        solver.parameters.log_search_progress = True
        if hasattr(solver.parameters, "use_lns"):
            solver.parameters.use_lns = True

        # Solve WITHOUT incumbents callback; just final best
        status = solver.Solve(model)
        log(f"Status: {solver.StatusName(status)}")

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            log("No feasible solution found (skipping this iteration).")
            continue  # assume "always feasible" but be safe

        # Decode final best assignment
        assignment = {
            s: next(a for a in opts if solver.Value(x[(s, a)]) == 1)
            for s, opts in site_options.items()
        }

        try:
            best_obj_value = solver.ObjectiveValue()
            best_E = best_obj_value / SCALE
        except Exception:
            best_obj_value = None
            best_E = None

        energy_str = f"{best_E:.3f}" if best_E is not None else "N/A"
        log(f"Final best objective (scaled): {best_obj_value}, E = {energy_str} eV")

        # Store FINAL incumbent only (so rebuild_incumbent_structures can find it)
        cfg_hash = append_incumbent(
            output_dir=output_dir,
            assignment=assignment,
            energy_ev=best_E,
            li_sites=li_sites,
            mn_sites=mn_sites,
            tags={"status": "FINAL", "solver_status": solver.StatusName(status)},
        )
        log(f"Saved FINAL incumbent cfg: {cfg_hash}")

        # Now rebuild ONLY this final/best structure and append to all_before_optimized
        try:
            best_structs, best_energies = rebuild_incumbent_structures(
                output_dir,
                keep_final_only=True,   # only the best
                encode_mn_as_tc=True,
                set_oxidation=False,
            )
            if best_structs:
                append_extxyz(
                    before_path,
                    structures=best_structs,
                    energies=best_energies,
                )
                log(f"Appended best structure of iter {i} to {before_path}")
            # if best_structs:
            #     append_unique_extxyz(
            #         before_path,
            #         structures=best_structs,
            #         energies=best_energies,
            #         tol_frac=1e-4,
            #         tol_lat=1e-3,
            #     )
            #     log(f"Appended best structure of iter {i} to {before_path}")
            else:
                log("rebuild_incumbent_structures returned no structures for FINAL config.")
        except Exception as exc:
            log(f"[warn] Failed to rebuild/append best structure for iteration {i}: {exc}")

    # ----------------------------------------------------------------
    # AFTER all iterations for this N_li:
    #   - read all_before_optimized.extxyz
    #   - write A0.gin, A1.gin, ... into gulp_root_dir
    # ----------------------------------------------------------------
    try:
        if not os.path.exists(before_path):
            log(f"No {before_path} found, skipping GULP export for N_li = {N_li}")
            continue

        # Read all best structures for this N_li as ASE Atoms
        best_atoms_list = read(before_path, index=":")
        if not isinstance(best_atoms_list, list):
            best_atoms_list = [best_atoms_list]
        adaptor = AseAtomsAdaptor()
        best_structures = [adaptor.get_structure(atoms) for atoms in best_atoms_list]

        def run_gulp_phase(structs, phase_dir, phase_name, template_lines=None, geom_lines=None, head_override=None, tail_override=None):
            os.makedirs(phase_dir, exist_ok=True)
            run_dir = os.path.join(phase_dir, "run")
            os.makedirs(run_dir, exist_ok=True)
            # Build gin files either from structures or from geometry + template
            if geom_lines is not None and template_lines is not None:
                head, tail = get_head_tail(template_lines, head_override=head_override, tail_override=tail_override)
                gin_path = os.path.join(run_dir, "A0.gin")
                log(f"[{phase_name}] Writing gin from geometry → {gin_path}")
                with open(gin_path, "w") as fh:
                    fh.writelines(head)
                    fh.writelines(geom_lines)
                    fh.writelines(tail)
                log(f"[{phase_name}] Wrote GULP input from gulp.res: {gin_path}")
                count = 1
            else:
                count = 0
                for idx, pmg_struct in enumerate(structs):
                    gin_path = os.path.join(run_dir, f"A{idx}.gin")
                    log(f"[{phase_name}] Writing gin from structure {idx} → {gin_path}")
                    write_gulp_input(structure=pmg_struct, filename=gin_path)
                    if phase_name == "phase1" and PHASE1_TAIL_APPEND:
                        with open(gin_path, "a") as fh:
                            fh.writelines(PHASE1_TAIL_APPEND)
                    count += 1
                log(f"[{phase_name}] Wrote {count} GULP inputs from structures.")
            # Ensure count matches actual gin files present
            gin_files = list(Path(run_dir).glob("A*.gin"))
            count = len(gin_files)
            log(f"[{phase_name}] Detected {count} gin files in {run_dir} for taskfarm.")
            # Write scripts in phase_dir and run there
            write_gulp_batch_scripts(
                dest_dir=run_dir,
                count=count,
                job_name=f"gulp_{phase_name}",
                account="e05-algor-smw",
                partition="standard",
                qos="short",
                exe_path="/work/e05/e05/bcamino/klmc_exe/klmc3.062024.x",
                ntasks=count,
                ntasks_per_node=count,
                cpus_per_task=1,
            )
            bash_script = os.path.join(run_dir, "SLURM_js.slurm")
            if os.path.exists(bash_script):
                log(f"[{phase_name}] Running SLURM script: {bash_script}")
                try:
                    t0 = time.time()
                    result = subprocess.run(
                        ["bash", bash_script],
                        cwd=run_dir,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    elapsed = time.time() - t0
                    log(f"[{phase_name}] GULP batch finished in {elapsed:.1f} s")
                    if result.stdout:
                        log(f"[{phase_name}] SLURM STDOUT:\n{result.stdout}")
                    if result.stderr:
                        log(f"[{phase_name}] SLURM STDERR:\n{result.stderr}")
                    # TEMP: stop after GULP run for debugging
                    raise SystemExit("DEBUG STOP: exiting immediately after GULP execution.")
                except subprocess.CalledProcessError as exc:
                    log(f"[{phase_name}] SLURM failed with code {exc.returncode}")
                    if exc.stdout:
                        log(f"Captured STDOUT:\n{exc.stdout}")
                    if exc.stderr:
                        log(f"Captured STDERR:\n{exc.stderr}")
                    raise
            else:
                log(f"[{phase_name}] SLURM script missing at {bash_script}, skipping.")
            opt_structs, opt_energy, opt_energy_init = read_opt_structures(
                os.path.join(run_dir, "result"),
                input_name,
                max_structures=N_structures_opt,
            )
            if len(opt_structs) == 0:
                # Fallback: glob gout files manually
                result_dir = os.path.join(run_dir, "result")
                log(f"[{phase_name}] read_opt_structures returned 0; scanning {result_dir} for gout files.")
                gout_files = sorted(Path(result_dir).rglob(f"{input_name[:-3]}gout"))
                for gf in gout_files:
                    try:
                        s, e_final, e_init = parse_gulp_to_pymatgen(str(gf))
                        opt_structs.append(s)
                        opt_energy.append(e_final)
                        opt_energy_init.append(e_init)
                        log(f"[{phase_name}] parsed gout: {gf}")
                    except Exception as exc:
                        log(f"[{phase_name}] failed to parse {gf}: {exc}")
            # Extract geometry lines from gulp.res for chaining
            res_path = os.path.join(run_dir, "gulp.res")
            geom_next = extract_geom_from_res(res_path)
            return opt_structs, opt_energy, opt_energy_init, geom_next

        # Phase 1
        phase1_dir = os.path.join(gulp_root_dir, "phase1")
        # Build template from the first gin we write in phase1
        template_lines = None
        p1_structs, p1_energy, p1_energy_init, geom_p1 = run_gulp_phase(
            best_structures, phase1_dir, "phase1", template_lines=None, geom_lines=None
        )
        log(f"[phase1] gin files written to {os.path.join(phase1_dir,'run')}")
        log(f"[phase1] opt_structs={len(p1_structs)}, geom_lines={len(geom_p1) if geom_p1 else 0}")
        # Capture template from the first gin
        first_gin = os.path.join(phase1_dir, "run", "A0.gin")
        if os.path.exists(first_gin):
            with open(first_gin, "r") as fh:
                template_lines = fh.readlines()

        # Phase 2 (use outputs of phase 1)
        phase2_dir = os.path.join(gulp_root_dir, "phase2")
        p2_structs, p2_energy, p2_energy_init, geom_p2 = run_gulp_phase(
            p1_structs, phase2_dir, "phase2",
            template_lines=template_lines,
            geom_lines=geom_p1 if geom_p1 else None,
            head_override=PHASE2_HEAD,
            tail_override=PHASE2_TAIL,
        )
        log(f"[phase2] gin files written to {phase2_dir} (geom_lines_in={len(geom_p1) if geom_p1 else 0})")
        log(f"[phase2] opt_structs={len(p2_structs)}, geom_lines={len(geom_p2) if geom_p2 else 0}")
        # Phase 3 (use outputs of phase 2)
        phase3_dir = os.path.join(gulp_root_dir, "phase3")
        p3_structs, p3_energy, p3_energy_init, geom_p3 = run_gulp_phase(
            p2_structs, phase3_dir, "phase3",
            template_lines=template_lines,
            geom_lines=geom_p2 if geom_p2 else None,
            head_override=PHASE3_HEAD,
            tail_override=PHASE3_TAIL,
        )
        log(f"[phase3] gin files written to {phase3_dir} (geom_lines_in={len(geom_p2) if geom_p2 else 0})")
        log(f"[phase3] opt_structs={len(p3_structs)}, geom_lines={len(geom_p3) if geom_p3 else 0}")

        # Save final optimized structures/energies (phase 3) to per-N_li folder
        opt_extxyz = os.path.join(folder_path, "all_optimized.extxyz")
        append_extxyz(opt_extxyz, structures=p3_structs, energies=p3_energy)

        energies_json = os.path.join(folder_path_root, "energies.json")
        energies_payload = {}
        if os.path.exists(energies_json):
            try:
                with open(energies_json, "r") as fh:
                    energies_payload = json.load(fh)
            except Exception:
                energies_payload = {}
        entry = energies_payload.get(str(N_li), {"final": [], "initial": []})
        if not isinstance(entry, dict):
            entry = {"final": [], "initial": []}
        entry["final"].extend([e for e in p3_energy if e is not None])
        entry["initial"].extend([e for e in p3_energy_init if e is not None])
        energies_payload[str(N_li)] = entry
        with open(energies_json, "w") as fh:
            json.dump(energies_payload, fh, indent=2)

        log(
            f"GULP phases complete for N_li = {N_li}: "
            f"{len(p3_structs)} final structures saved."
        )

    except Exception as exc:
        log(f"[warn] Failed GULP export for N_li = {N_li}: {exc}")

    # Also save the current params to a JSON alongside energies.json for traceability
    params_json = os.path.join(folder_path_root, "params_used.json")
    try:
        with open(params_json, "w") as fh:
            json.dump(PARAMS, fh, indent=2)
    except Exception as exc:
        log(f"[warn] Failed to write params_used.json: {exc}")

################################### THE END #########################################




#     prev_grid_for_metrics = None
#     if GRID_CONVERGENCE_MODE.lower() != "none" and grid is not None:
#         prev_grid_for_metrics = np.array(grid, copy=True)

#     if n is None:
#         structures_to_read = N_structures_opt
#     else:
#         structures_to_read = min(n, N_structures_opt) if N_structures_opt is not None else n
#     opt_structures_raw, all_opt_energy_raw = read_opt_structures(
#         os.path.join(output_dir, 'gulp/result'), input_name, max_structures=structures_to_read
#     )
#     opt_structures = opt_structures_raw
#     all_opt_energy = all_opt_energy_raw
#     log(f'Iteration {i} - GULP opt_structures = {len(opt_structures_raw)}')
#     if HOST_FRAC_DRIFT_TOL is not None:
#         total_before = len(opt_structures_raw)
#         opt_structures, all_opt_energy, rejected = filter_structures_by_host_drift(
#             opt_structures_raw, all_opt_energy_raw, initial_structure, HOST_FRAC_DRIFT_TOL
#         )
#         if rejected:
#             log(f"Host-drift filter: kept {len(opt_structures)}/{total_before} (rejected {rejected}, tol={HOST_FRAC_DRIFT_TOL})")
#     # Filter pre-GULP structures similarly
#     pre_structures_filtered = pre_structures
#     pre_energies_filtered = pre_energies
#     if pre_structures and HOST_FRAC_DRIFT_TOL is not None:
#         total_before_pre = len(pre_structures)
#         pre_structures_filtered, pre_energies_filtered, rejected_pre = filter_structures_by_host_drift(
#             pre_structures, pre_energies, initial_structure, HOST_FRAC_DRIFT_TOL
#         )
#         if rejected_pre:
#             log(f"Host-drift filter (pre-GULP): kept {len(pre_structures_filtered)}/{total_before_pre} (rejected {rejected_pre}, tol={HOST_FRAC_DRIFT_TOL})")
#     if GRID_QUALITY_RADIUS is not None:
#         quality_metrics = compute_grid_quality(opt_structures, grid, initial_structure.lattice, GRID_QUALITY_RADIUS)
#         write_grid_quality(os.path.join(folder_path, "grid_quality.jsonl"), quality_metrics, tag=f"opt_{i}")

#     grid_source_structs = pre_structures_filtered if pre_structures_filtered else opt_structures

#     structure, grid_frac, grid_meta = build_new_structural_model(
#         grid_source_structs,
#         M,
#         N_positions_final,
#         initial_structure,
#         threshold,
#         return_stats=False,
#         fix_angles=True,
#         grid_update_mode=GRID_UPDATE_MODE,
#         explore_target=GRID_EXPLORE_SIZE,
#         use_dual=GRID_USE_DUAL_GRID,
#         tracker=grid_tracker,
#         tracker_decay=GRID_TRACKER_DECAY,
#         tracker_min_visits=GRID_TRACKER_MIN_VISITS,
#         tracker_decimals=GRID_TRACKER_DECIMALS,
#         iteration=i,
#     )
#     if grid_meta:
#         grid_tracker = grid_meta.get("tracker", grid_tracker)
#     new_grid_cart = structure.lattice.get_cartesian_coords(grid_frac)
#     explore_frac = grid_meta.get("explore_grid_frac", grid_frac) if grid_meta else grid_frac
#     new_grid_explore = structure.lattice.get_cartesian_coords(explore_frac)
#     if GRID_USE_DECAY:
#         grid, grid_decay_records = update_grid_with_decay(
#             new_grid_cart, grid_decay_records, GRID_DECAY_THRESHOLD, GRID_DECAY_TTL
#         )
#     else:
#         if DAMP_GRID_UPDATES:
#             grid = damp_grid_update(prev_grid_for_metrics, new_grid_cart, DAMP_GRID_THRESHOLD)
#         else:
#             grid = new_grid_cart
#         grid_decay_records = []
#     if GRID_USE_DUAL_GRID:
#         grid_explore = new_grid_explore
#     else:
#         grid_explore = grid
#     report_grid_convergence(prev_grid_for_metrics, grid, GRID_CONVERGENCE_MODE, GRID_OVERLAP_THRESHOLD)
#     # folder_path = "runs/out_20251112_151804_526"
#     extxyz_path_all = f"{folder_path}/all_optimized.extxyz"  # all structures
#     extxyz_path_good = f"{folder_path}/all_optimized_good.extxyz"  # filtered by host drift
#     extxyz_path_all_iter = f"{folder_path}/all_optimized_{i}.extxyz"
#     extxyz_path_good_iter = f"{folder_path}/all_optimized_good_{i}.extxyz"

#     # Write raw (all) structures
#     added_all, skipped_old_all, skipped_dup_all = append_unique_extxyz(
#         extxyz_path_all,
#         structures=opt_structures_raw,
#         energies=all_opt_energy_raw,
#         tol_frac=1e-4,
#         tol_lat=1e-3
#     )
#     # Write filtered (good) structures
#     added_good, skipped_old_good, skipped_dup_good = append_unique_extxyz(
#         extxyz_path_good,
#         structures=opt_structures,
#         energies=all_opt_energy,
#         tol_frac=1e-4,
#         tol_lat=1e-3
#     )
#     # Per-iteration files
#     append_unique_extxyz(
#         extxyz_path_all_iter,
#         structures=opt_structures_raw,
#         energies=all_opt_energy_raw,
#         tol_frac=1e-4,
#         tol_lat=1e-3
#     )
#     append_unique_extxyz(
#         extxyz_path_good_iter,
#         structures=opt_structures,
#         energies=all_opt_energy,
#         tol_frac=1e-4,
#         tol_lat=1e-3
#     )

# log(f"extxyz (all) update → added: {added_all}, skipped(existing): {skipped_old_all}, skipped(batch-dup): {skipped_dup_all}")
# log(f"extxyz (good) update → added: {added_good}, skipped(existing): {skipped_old_good}, skipped(batch-dup): {skipped_dup_good}")

# log(f'************ End Iteration {i} ************\n')

# if RUN_PERTURB_SAMPLING:
#     if last_run_data is None or last_run_data.get("QUBO") is None:
#         log("Perturbation sampling skipped: no reference QUBO recorded.")
#     else:
#         log("===== Starting perturb-and-solve sampling pass =====")
#         rng = np.random.default_rng(PERTURB_RANDOM_SEED)
#         perturb_extxyz_path = os.path.join(folder_path, "all_perturbed.extxyz")
#         perturb_grid_extxyz = os.path.join(folder_path, "all_perturbed_grid.extxyz")
#         try:
#             lattice_assets, base_struct_assets, li_sites_assets, mn_sites_assets, li_grid_frac_assets, mn_atom_indices_assets = load_run_assets(last_run_data["output_dir"])
#         except Exception:
#             lattice_assets = structure.lattice
#             base_struct_assets = structure.copy()
#             li_sites_assets = last_run_data.get("li_indices", [])
#             mn_sites_assets = last_run_data.get("mn_indices", [])
#             li_grid_frac_assets = structure.lattice.get_fractional_coords(grid)
#             mn_atom_indices_assets = list(range(len(mn_sites_assets)))
#         li_grid_coords_assets = lattice_assets.get_cartesian_coords(li_grid_frac_assets)

#         successes = 0
#         for k in range(PERTURB_NUM_RUNS):
#             log(f"~~~~ Begin Perturbation Run {k} ~~~~")
#             output_dir = os.path.join(folder_path, f"perturb_folder_{k}")
#             os.makedirs(output_dir, exist_ok=True)

#             append_grid_extxyz(perturb_grid_extxyz, initial_structure.lattice, grid, tag=f"perturb_iter_{k}")
#             if GRID_USE_DUAL_GRID and grid_explore is not None:
#                 append_grid_extxyz(perturb_grid_extxyz, initial_structure.lattice, grid_explore, tag=f"perturb_iter_{k}_explore")

#             Q_pert = perturb_qubo(
#                 last_run_data["QUBO"],
#                 diag_noise_ev=PERTURB_DIAG_NOISE_EV,
#                 pair_noise_ev=PERTURB_PAIR_NOISE_EV,
#                 rng=rng,
#             )

#             model, x, site_options, var2siteopt, li_sites, mn_sites = cpsat_core_from_indices(
#                 last_run_data["li_indices"],
#                 last_run_data["mn_indices"],
#                 N_li=N_li,
#             )

#             groups, _ = build_li_proximity_groups(
#                 li_grid_coords=li_grid_coords_assets,
#                 threshold_ang=LI_LI_EXCLUSION_ANG,
#                 lattice=lattice_assets,
#                 coords_are_cartesian=True,
#                 site_ids=li_sites,
#             )
#             add_li_proximity_exclusions(model, x, groups)
#             log(f"Li-Li proximity exclusions (perturb): {len(groups)} groups at {LI_LI_EXCLUSION_ANG} Å")

#             SCALE_p, info_p = add_ut_qubo_objective(model, x, var2siteopt, Q_pert)
#             n_workers = mp.cpu_count()
#             solver_params = {"time": PERTURB_MAX_TIME, "workers": n_workers, "seed": PERTURB_RANDOM_SEED + k}
#             _ = init_run_store(
#                 output_dir=output_dir,
#                 initial_structure=structure,
#                 li_sites=li_sites,
#                 mn_sites=mn_sites,
#                 initial_grid_cart=grid,
#                 mn_atom_indices=mn_atom_indices_assets,
#                 QUBO_ut=Q_pert,
#                 SCALE=SCALE_p,
#                 solver_params=solver_params,
#             )
#             save_qubo_var_mapping(output_dir, var2siteopt, li_sites, mn_sites)
#             save_energy_components(output_dir, ewald_discrete=None, buckingham_discrete=None)

#             solver = cp_model.CpSolver()
#             solver.parameters.max_time_in_seconds = PERTURB_MAX_TIME
#             solver.parameters.num_search_workers = n_workers
#             solver.parameters.random_seed = solver_params["seed"]
#             solver.parameters.log_search_progress = True
#             status = solver.Solve(model)
#             log(f"Perturbation solver status: {solver.StatusName(status)}")
#             if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#                 continue

#             assignment = _decode_assignment_from_solver(site_options, solver, x)
#             try:
#                 energy_val = solver.ObjectiveValue() / SCALE_p
#             except Exception:
#                 energy_val = None

#             cfg_hash = append_incumbent(
#                 output_dir=output_dir,
#                 assignment=assignment,
#                 energy_ev=energy_val,
#                 li_sites=li_sites,
#                 mn_sites=mn_sites,
#                 tags={"status": "PERTURB", "iteration": k},
#             )
#             log(f"Stored perturb cfg {cfg_hash}, E = {energy_val if energy_val is not None else 'N/A'} eV")

#             dest_dir = os.path.join(output_dir, "gulp/run")
#             n = write_gulp_inputs_from_incumbents(
#                 output_dir=output_dir,
#                 dest_dir=dest_dir,
#                 limit=1 if KEEP_FINAL_INCUMBENTS_ONLY else PERTURB_GULP_LIMIT,
#                 keep_final_only=KEEP_FINAL_INCUMBENTS_ONLY,
#                 filename_pattern="A{idx}.gin",
#                 write_gulp_input_fn=write_gulp_input,
#                 dedup_by_hash=DEDUP_INCUMBENTS,
#             )
#             log(f"Perturb incumbents dumped: {n}")
#             try:
#                 before_path = os.path.join(folder_path, "all_before_optimized.extxyz")
#                 added_before = write_incumbent_structures_extxyz(
#                     output_dir,
#                     before_path,
#                     keep_final_only=KEEP_FINAL_INCUMBENTS_ONLY,
#                     encode_mn_as_tc=True,
#                     set_oxidation=False,
#                 )
#                 log(f"Saved {added_before} pre-GULP perturb structures to {before_path}")
#             except Exception as exc:
#                 log(f"[warn] Failed to save pre-GULP perturb structures: {exc}")

#             bash_script = os.path.join(output_dir, 'gulp', 'SLURM_js.slurm')
#             gulp_cwd = os.path.join(output_dir, 'gulp')
#             if os.path.exists(bash_script):
#                 try:
#                     t0 = time.time()
#                     result = subprocess.run(
#                         ["bash", bash_script],
#                         cwd=gulp_cwd,
#                         capture_output=True,
#                         text=True,
#                         check=True,
#                     )
#                     elapsed = time.time() - t0
#                     log(f"Perturbation GULP batch finished in {elapsed:.1f} s")
#                     if result.stdout:
#                         log(f"Perturb SLURM STDOUT:\n{result.stdout}")
#                     if result.stderr:
#                         log(f"Perturb SLURM STDERR:\n{result.stderr}")
#                 except subprocess.CalledProcessError as exc:
#                     log(f"Perturb SLURM failed with code {exc.returncode}")
#                     if exc.stdout:
#                         log(f"Captured STDOUT:\n{exc.stdout}")
#                     if exc.stderr:
#                         log(f"Captured STDERR:\n{exc.stderr}")
#                     continue
#             else:
#                 log(f"Perturb SLURM script missing at {bash_script}, skipping GULP run.")
#                 continue

#             prev_grid_pert = None
#             if GRID_CONVERGENCE_MODE.lower() != "none" and grid is not None:
#                 prev_grid_pert = np.array(grid, copy=True)

#             if n is None:
#                 structures_to_read = N_structures_opt
#             else:
#                 structures_to_read = min(n, N_structures_opt) if N_structures_opt is not None else n
#             opt_structures, all_opt_energy = read_opt_structures(
#                 os.path.join(output_dir, 'gulp/result'), input_name, max_structures=structures_to_read
#             )
#             log(f'Perturb iteration {k} - GULP opt_structures = {len(opt_structures)}')
#             if HOST_FRAC_DRIFT_TOL is not None:
#                 total_before = len(opt_structures)
#                 opt_structures, all_opt_energy, rejected = filter_structures_by_host_drift(
#                     opt_structures, all_opt_energy, initial_structure, HOST_FRAC_DRIFT_TOL
#                 )
#                 if rejected:
#                     log(f"Host-drift filter: kept {len(opt_structures)}/{total_before} (rejected {rejected}, tol={HOST_FRAC_DRIFT_TOL})")
#             if GRID_QUALITY_RADIUS is not None:
#                 quality_metrics = compute_grid_quality(opt_structures, grid, initial_structure.lattice, GRID_QUALITY_RADIUS)
#                 write_grid_quality(os.path.join(folder_path, "grid_quality.jsonl"), quality_metrics, tag=f"perturb_{k}")

#             if opt_structures:
#                 structure, grid_frac, grid_meta = build_new_structural_model(
#                     opt_structures,
#                     M,
#                     N_positions_final,
#                     initial_structure,
#                     threshold,
#                     return_stats=False,
#                     fix_angles=True,
#                     grid_update_mode=GRID_UPDATE_MODE,
#                     explore_target=GRID_EXPLORE_SIZE,
#                     use_dual=GRID_USE_DUAL_GRID,
#                     tracker=grid_tracker,
#                     tracker_decay=GRID_TRACKER_DECAY,
#                     tracker_min_visits=GRID_TRACKER_MIN_VISITS,
#                     tracker_decimals=GRID_TRACKER_DECIMALS,
#                     iteration=num_iterations + k + 1,
#                 )
#                 if grid_meta:
#                     grid_tracker = grid_meta.get("tracker", grid_tracker)
#                 new_grid_cart = structure.lattice.get_cartesian_coords(grid_frac)
#                 explore_frac = grid_meta.get("explore_grid_frac", grid_frac) if grid_meta else grid_frac
#                 new_grid_explore = structure.lattice.get_cartesian_coords(explore_frac)
#                 if GRID_USE_DECAY:
#                     grid, grid_decay_records = update_grid_with_decay(
#                         new_grid_cart, grid_decay_records, GRID_DECAY_THRESHOLD, GRID_DECAY_TTL
#                     )
#                 else:
#                     if DAMP_GRID_UPDATES:
#                         grid = damp_grid_update(prev_grid_pert, new_grid_cart, DAMP_GRID_THRESHOLD)
#                     else:
#                         grid = new_grid_cart
#                     grid_decay_records = []
#                 if GRID_USE_DUAL_GRID:
#                     grid_explore = new_grid_explore
#                 else:
#                     grid_explore = grid
#                 report_grid_convergence(prev_grid_pert, grid, GRID_CONVERGENCE_MODE, GRID_OVERLAP_THRESHOLD)
#                 append_unique_extxyz(
#                     perturb_extxyz_path,
#                     structures=opt_structures,
#                     energies=all_opt_energy,
#                     tol_frac=1e-4,
#                     tol_lat=1e-3,
#                 )
#                 append_grid_extxyz(perturb_grid_extxyz, structure.lattice, grid, tag=f"perturb_iter_{k}_post")
#                 if GRID_USE_DUAL_GRID and grid_explore is not None:
#                     append_grid_extxyz(perturb_grid_extxyz, structure.lattice, grid_explore, tag=f"perturb_iter_{k}_post_explore")
#                 successes += 1
#             else:
#                 log("No optimized structures returned from GULP in perturb run.")
#         log(f"Perturbation sampling complete → {successes} solutions recorded.")

# if RUN_BOLTZMANN_LOOP:
#     if global_best_energy_scaled is None:
#         log("Boltzmann pass skipped: no baseline energy available.")
#     elif not (BOLTZMANN_USE_ENERGY_WINDOW or BOLTZMANN_USE_DIVERSITY_OBJECTIVE):
#         log("Boltzmann pass skipped: enable energy window and/or diversity objective.")
#     else:
#         log("===== Starting Boltzmann sampling pass =====")
#         boltz_extxyz_path = os.path.join(folder_path, "all_boltzmann.extxyz")
#         boltzmann_assignments = []

#         for j in range(BOLTZMANN_NUM_ITERATIONS):
#             log(f"~~~~ Begin Boltzmann Iteration {j} ~~~~")
#             output_dir = os.path.join(folder_path, f"boltzmann_folder_{j}")
#             os.makedirs(output_dir, exist_ok=True)
#             append_grid_extxyz(grid_extxyz_boltz, initial_structure.lattice, grid, tag=f"boltz_iter_{j}")
#             if GRID_USE_DUAL_GRID and grid_explore is not None:
#                 append_grid_extxyz(grid_extxyz_boltz_explore, initial_structure.lattice, grid_explore, tag=f"boltz_iter_{j}_explore")

#             QUBO, li_indices, mn_indices, ewald_discrete, buckingham_discrete = build_QUBO(
#                 structure,
#                 threshold_li=threshold_li,
#                 prox_penalty=prox_penalty
#             )

#             model, x, site_options, var2siteopt, li_sites, mn_sites = cpsat_core_from_indices(
#                 li_indices, mn_indices, N_li=N_li
#             )
#             save_qubo_var_mapping(output_dir, var2siteopt, li_sites, mn_sites)
#             save_energy_components(output_dir, ewald_discrete, buckingham_discrete)

#             li_coords_boltz = extract_li_cartesian_coords(structure)
#             if len(li_sites) != len(li_coords_boltz):
#                 raise ValueError(
#                     f"Li site/count mismatch before proximity exclusions (boltz): "
#                     f"{len(li_sites)} site IDs vs {len(li_coords_boltz)} grid coords."
#                 )
#             groups, _ = build_li_proximity_groups(
#                 li_grid_coords=li_coords_boltz,
#                 threshold_ang=LI_LI_EXCLUSION_ANG,
#                 lattice=structure.lattice,
#                 coords_are_cartesian=True,
#                 site_ids=li_sites
#             )
#             add_li_proximity_exclusions(model, x, groups)
#             log(f"Li-Li proximity exclusions (boltz): {len(groups)} groups at {LI_LI_EXCLUSION_ANG} Å")

#             SCALE_b, energy_terms_b, _ = _build_qubo_energy_terms(
#                 model, x, var2siteopt, QUBO
#             )
#             energy_expr = sum(c * var for c, var in energy_terms_b)

#             if BOLTZMANN_USE_ENERGY_WINDOW:
#                 window_int = int(round(BOLTZMANN_ENERGY_WINDOW_EV * SCALE_b))
#                 lower = int(global_best_energy_scaled)
#                 model.Add(energy_expr <= lower + window_int)
#                 # lower bound
#                 #model.Add(energy_expr >= lower)

#             if BOLTZMANN_MIN_HAMMING_DISTANCE > 0:
#                 for prev in boltzmann_assignments:
#                     _add_hamming_distance_constraint(
#                         model, x, site_options, prev, BOLTZMANN_MIN_HAMMING_DISTANCE
#                     )

#             if BOLTZMANN_USE_DIVERSITY_OBJECTIVE and boltzmann_assignments:
#                 prev_assignment = boltzmann_assignments[-1]
#                 diversity_terms = []
#                 for s, opts in site_options.items():
#                     prev_opt = prev_assignment.get(s)
#                     if prev_opt is None:
#                         continue
#                     diversity_terms.append(1 - x[(s, prev_opt)])
#                 if diversity_terms:
#                     model.Maximize(sum(diversity_terms))
#             elif not BOLTZMANN_USE_ENERGY_WINDOW:
#                 model.Minimize(energy_expr)

#             n_workers = mp.cpu_count()
#             solver_params = {"time": BOLTZMANN_MAX_TIME, "workers": n_workers, "seed": 2025021}
#             solver = cp_model.CpSolver()
#             solver.parameters.max_time_in_seconds = BOLTZMANN_MAX_TIME
#             solver.parameters.num_search_workers = n_workers
#             solver.parameters.random_seed = solver_params["seed"]
#             solver.parameters.log_search_progress = True

#             status = solver.Solve(model)
#             log(f"Boltzmann solver status: {solver.StatusName(status)}")
#             if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
#                 log("No further solutions found within requested criteria.")
#                 break

#             assignment = _decode_assignment_from_solver(site_options, solver, x)
#             energy_val = solver.Value(energy_expr) / SCALE_b
#             cfg_hash = append_incumbent(
#                 output_dir=output_dir,
#                 assignment=assignment,
#                 energy_ev=energy_val,
#                 li_sites=li_sites,
#                 mn_sites=mn_sites,
#                 tags={"status": "BOLTZMANN", "iteration": j}
#             )
#             boltzmann_assignments.append(assignment)
#             log(f"Stored Boltzmann cfg {cfg_hash} with E = {energy_val:.3f} eV")

#             dest_dir = os.path.join(output_dir, "gulp/run")
#             n = write_gulp_inputs_from_incumbents(
#                 output_dir=output_dir,
#                 dest_dir=dest_dir,
#                 limit=1 if KEEP_FINAL_INCUMBENTS_ONLY else BOLTZMANN_GULP_LIMIT,
#                 keep_final_only=KEEP_FINAL_INCUMBENTS_ONLY,
#                 filename_pattern="A{idx}.gin",
#                 write_gulp_input_fn=write_gulp_input,
#                 dedup_by_hash=DEDUP_INCUMBENTS,
#             )
#             log(f"Boltzmann incumbents dumped: {n}")
#             try:
#                 before_path = os.path.join(folder_path, "all_before_optimized.extxyz")
#                 added_before = write_incumbent_structures_extxyz(
#                     output_dir,
#                     before_path,
#                     keep_final_only=KEEP_FINAL_INCUMBENTS_ONLY,
#                     encode_mn_as_tc=True,
#                     set_oxidation=False,
#                 )
#                 log(f"Saved {added_before} pre-GULP boltzmann structures to {before_path}")
#             except Exception as exc:
#                 log(f"[warn] Failed to save pre-GULP boltzmann structures: {exc}")

#             bash_script = os.path.join(output_dir, 'gulp', 'SLURM_js.slurm')
#             gulp_cwd = os.path.join(output_dir, 'gulp')
#             if os.path.exists(bash_script):
#                 try:
#                     t0 = time.time()
#                     result = subprocess.run(
#                         ["bash", bash_script],
#                         cwd=gulp_cwd,
#                         capture_output=True,
#                         text=True,
#                         check=True,
#                     )
#                     elapsed = time.time() - t0
#                     log(f"Boltzmann GULP batch finished in {elapsed:.1f} s")
#                     if result.stdout:
#                         log(f"Boltzmann SLURM STDOUT:\n{result.stdout}")
#                     if result.stderr:
#                         log(f"Boltzmann SLURM STDERR:\n{result.stderr}")
#                 except subprocess.CalledProcessError as exc:
#                     log(f"Boltzmann SLURM failed with code {exc.returncode}")
#                     if exc.stdout:
#                         log(f"Captured STDOUT:\n{exc.stdout}")
#                     if exc.stderr:
#                         log(f"Captured STDERR:\n{exc.stderr}")
#                     break
#             else:
#                 log(f"Boltzmann SLURM script missing at {bash_script}, skipping GULP run.")

#             prev_grid_boltz = None
#             if GRID_CONVERGENCE_MODE.lower() != "none" and grid is not None:
#                 prev_grid_boltz = np.array(grid, copy=True)

#             if n is None:
#                 structures_to_read = N_structures_opt
#             else:
#                 structures_to_read = min(n, N_structures_opt) if N_structures_opt is not None else n
#             opt_structures, all_opt_energy = read_opt_structures(
#                 os.path.join(output_dir, 'gulp/result'), input_name, max_structures=structures_to_read
#             )
#             log(f'Boltzmann iteration {j} - GULP opt_structures = {len(opt_structures)}')
#             if HOST_FRAC_DRIFT_TOL is not None:
#                 total_before = len(opt_structures)
#                 opt_structures, all_opt_energy, rejected = filter_structures_by_host_drift(
#                     opt_structures, all_opt_energy, initial_structure, HOST_FRAC_DRIFT_TOL
#                 )
#                 if rejected:
#                     log(f"Host-drift filter: kept {len(opt_structures)}/{total_before} (rejected {rejected}, tol={HOST_FRAC_DRIFT_TOL})")
#             if GRID_QUALITY_RADIUS is not None:
#                 quality_metrics = compute_grid_quality(opt_structures, grid, initial_structure.lattice, GRID_QUALITY_RADIUS)
#                 write_grid_quality(os.path.join(folder_path, "grid_quality.jsonl"), quality_metrics, tag=f"boltz_{j}")

#             if opt_structures:
#                 structure, grid_frac, grid_meta = build_new_structural_model(
#                     opt_structures,
#                     M,
#                     N_positions_final,
#                     initial_structure,
#                     threshold,
#                     return_stats=False,
#                     fix_angles=True,
#                     grid_update_mode=GRID_UPDATE_MODE,
#                     explore_target=GRID_EXPLORE_SIZE,
#                     use_dual=GRID_USE_DUAL_GRID,
#                     tracker=grid_tracker,
#                     tracker_decay=GRID_TRACKER_DECAY,
#                     tracker_min_visits=GRID_TRACKER_MIN_VISITS,
#                     tracker_decimals=GRID_TRACKER_DECIMALS,
#                     iteration=num_iterations + PERTURB_NUM_RUNS + j + 1,
#                 )
#                 if grid_meta:
#                     grid_tracker = grid_meta.get("tracker", grid_tracker)
#                 new_grid_cart = structure.lattice.get_cartesian_coords(grid_frac)
#                 explore_frac = grid_meta.get("explore_grid_frac", grid_frac) if grid_meta else grid_frac
#                 new_grid_explore = structure.lattice.get_cartesian_coords(explore_frac)
#                 if GRID_USE_DECAY:
#                     grid, grid_decay_records = update_grid_with_decay(
#                         new_grid_cart, grid_decay_records, GRID_DECAY_THRESHOLD, GRID_DECAY_TTL
#                     )
#                 else:
#                     if DAMP_GRID_UPDATES:
#                         grid = damp_grid_update(prev_grid_boltz, new_grid_cart, DAMP_GRID_THRESHOLD)
#                     else:
#                         grid = new_grid_cart
#                     grid_decay_records = []
#                 if GRID_USE_DUAL_GRID:
#                     grid_explore = new_grid_explore
#                 else:
#                     grid_explore = grid
#                 report_grid_convergence(prev_grid_boltz, grid, GRID_CONVERGENCE_MODE, GRID_OVERLAP_THRESHOLD)
#                 added, skipped_old, skipped_dup = append_unique_extxyz(
#                     boltz_extxyz_path,
#                     structures=opt_structures,
#                     energies=all_opt_energy,
#                     tol_frac=1e-4,
#                     tol_lat=1e-3
#                 )
#                 log(f"boltzmann extxyz update → added: {added}, skipped(existing): {skipped_old}, skipped(batch-dup): {skipped_dup}")
#             else:
#                 log("No optimized structures returned from GULP in Boltzmann run.")

#             log(f"~~~~ End Boltzmann Iteration {j} ~~~~\n")

#         log("===== Boltzmann sampling complete =====")

# if RUN_LOGGER is not None:
#     RUN_LOGGER.close()
