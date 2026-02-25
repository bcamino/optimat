#!/usr/bin/env python
# coding: utf-8

# # Fujitsu GULP

# #### Parameters
# INITIAL GRID:
# - N_initial_grid=500
# - min_dist_grid=1.
# - one_hot_value = 500
# - N_li = 2
# - weight = 1000
# - N_structures_opt

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# from QG_functions import *

# === Core Python ===
import sys
import time
import copy
import itertools
import subprocess
import pickle
import re
import shutil as sh
import os

# === Scientific Libraries ===
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy import constants
from scipy.spatial import KDTree, distance_matrix, cKDTree
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
# from sklearn.metrics import mean_squared_error as mse

# === Plotting ===
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('tableau-colorblind10')

# === ASE and Visualization ===
from ase.visualize import view
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write

# === Pymatgen ===
from pymatgen.core import Element
from pymatgen.core.structure import Structure, Lattice, PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.analysis.ewald import EwaldSummation

# === Optimization and Parallelization ===
from joblib import Parallel, delayed
from tqdm import tqdm

# === Fujitsu DADK ===
from dadk.BinPol import *
from dadk.QUBOSolverCPU import *

# === Janus ===
# from janus_core.calculations.single_point import SinglePoint
# from janus_core.calculations.geom_opt import GeomOpt

# === Constants ===
TO_EV = 14.39964390675221758120
k_b = constants.physical_constants['Boltzmann constant in eV/K'][0]

# === Utility ===
np.seterr(divide='ignore')

# === Helper Functions ===
def vview(structure):
    """Visualize a pymatgen Structure object using ASE."""
    view(AseAtomsAdaptor().get_atoms(structure))


# ## Functions

# In[2]:


def generate_filtered_grid(structure, N_initial_grid=1000, min_dist_grid=1.5):
    lattice = structure.lattice.matrix         # 3x3 array
    cart_coords = structure.cart_coords        # (N_atoms, 3)

    # Estimate the number of points per dimension
    volume = np.abs(np.linalg.det(lattice))
    spacing = (volume / N_initial_grid) ** (1/3)
    
    # Determine the number of grid points along each lattice vector
    lengths = np.linalg.norm(lattice, axis=1)
    num_points = np.maximum(np.round(lengths / spacing).astype(int), 1)
    
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


# In[3]:


def join_structure_grid(structure,initial_grid):

    initial_grid_pmg = Structure(structure.lattice,[3]*len(initial_grid),initial_grid,coords_are_cartesian=True)
    
    return Structure.from_sites(initial_grid_pmg.sites+structure.sites)


# In[4]:


# def precompute_translations(lattice_vectors, R_max):
#     max_shift = np.ceil(R_max / np.linalg.norm(lattice_vectors, axis=1)).astype(int)
#     shifts = []
#     for rnx in range(-max_shift[0], max_shift[0] + 1):
#         for rny in range(-max_shift[1], max_shift[1] + 1):
#             for rnz in range(-max_shift[2], max_shift[2] + 1):
#                 shift = rnx * lattice_vectors[0] + rny * lattice_vectors[1] + rnz * lattice_vectors[2]
#                 if norm(shift) < R_max:
#                     shifts.append(shift)
#     return np.array(shifts)

# def compute_real_contrib(i, j, cart_coords, shifts, Sigma):
#     dr_init = cart_coords[i] - cart_coords[j]
#     acc = 0.0
#     for shift in shifts:
#         dr = dr_init + shift
#         d = norm(dr)
#         if d > 1e-12:
#             acc += 0.5 / d * math.erfc(d / Sigma)
#         elif i == j:
#             acc += -1 / Sigma / math.sqrt(np.pi)
#     return i, j, acc * TO_EV

# def compute_ewald_matrix_fast(structure, sigma=None, R_max=None, charge=None, w=1, triu=True):
#     frac_coords = structure.frac_coords
#     lattice_vectors = structure.lattice.matrix
#     cart_coords = frac_coords @ lattice_vectors

#     N = len(frac_coords)
#     V = np.linalg.det(lattice_vectors)

#     if sigma is None:
#         Sigma = ((N * w * np.pi ** 3) / V ** 2) ** (-1 / 6)
#     else:
#         Sigma = sigma

#     if R_max is None:
#         A = 1e-17
#         R_max = np.sqrt(-np.log(A) * Sigma ** 2)

#     shifts = precompute_translations(lattice_vectors, R_max)

#     results = Parallel(n_jobs=-1)(
#         delayed(compute_real_contrib)(i, j, cart_coords, shifts, Sigma)
#         for i in range(N) for j in range(i, N)
#     )

#     Ewald_full = np.zeros((N, N))
#     for i, j, val in results:
#         Ewald_full[i, j] = val
#         if i != j:
#             Ewald_full[j, i] = val

#     if triu is True:
#         Ewald_full = np.triu(Ewald_full)

#     return Ewald_full


# In[36]:


def compute_ewald_matrix_fast(structure, real_depth=5, recip_depth=5, alpha=None, print_info=False, triu=False):
    """
    Computes the Ewald interaction matrix for a given structure using a fixed real and reciprocal space depth.
    Works for arbitrary lattice vectors and returns an NxN unit-charge interaction matrix (no charges applied).

    Parameters:
        structure: Pymatgen Structure
        real_depth: integer number of real-space lattice vector shells
        recip_depth: integer number of reciprocal-space vector shells
        alpha: optional Ewald parameter. If None, computed automatically
        print_info: whether to print debugging information

    Returns:
        NxN Ewald interaction matrix
    """
    N = len(structure)
    lattice = structure.lattice
    positions = structure.frac_coords
    volume = lattice.volume
    vecs = lattice.matrix
    recip = 2 * np.pi * np.linalg.inv(vecs).T

    # Convert fractional positions to Cartesian
    pos_cart = positions @ vecs

    # Determine alpha if not provided
    if alpha is None:
        alpha = 2 / (volume ** (1 / 3))

    # Prepare real space lattice vector shifts
    real_range = range(-real_depth, real_depth + 1)
    real_shifts = np.array(np.meshgrid(real_range, real_range, real_range)).T.reshape(-1, 3)
    real_shifts = real_shifts[np.any(real_shifts != 0, axis=1)]  # remove [0,0,0]
    real_shifts_cart = real_shifts @ vecs

    # Prepare reciprocal lattice vector shifts
    recip_range = range(-recip_depth, recip_depth + 1)
    recip_shifts = np.array(np.meshgrid(recip_range, recip_range, recip_range)).T.reshape(-1, 3)
    recip_shifts = recip_shifts[np.any(recip_shifts != 0, axis=1)]
    recip_shifts_cart = recip_shifts @ recip

    # Initialize matrix
    ewald_matrix = np.zeros((N, N))

    # Real space contribution
    for i in tqdm(range(N), desc="Real space"):
        for j in range(i, N):
            rij = pos_cart[i] - pos_cart[j]
            r0 = norm(rij)
            if i != j:
                ewald_matrix[i, j] += math.erfc(alpha * r0) / (2 * r0)
            for shift in real_shifts_cart:
                r = norm(rij + shift)
                ewald_matrix[i, j] += math.erfc(alpha * r) / (2 * r)

    # Self term
    for i in range(N):
        ewald_matrix[i, i] -= alpha / math.sqrt(math.pi)

    # Reciprocal space contribution
    for i in tqdm(range(N), desc="Reciprocal space"):
        for j in range(i, N):
            v = pos_cart[j] - pos_cart[i]
            for k in recip_shifts_cart:
                k2 = np.dot(k, k)
                coeff = (4 * math.pi ** 2) / k2
                coeff *= math.exp(-k2 / (4 * alpha ** 2))
                coeff *= math.cos(np.dot(k, v))
                ewald_matrix[i, j] += coeff / (2 * math.pi * volume)

    # Unit conversion to eV
    ewald_matrix *= TO_EV

    # Symmetrize
    for i in range(N):
        for j in range(i):
            ewald_matrix[i, j] = ewald_matrix[j, i]

    if print_info:
        print(f"alpha = {alpha}")
        print(f"volume = {volume}")
        print(f"max (eV): {np.max(ewald_matrix)}, min (eV): {np.min(ewald_matrix)}")

    if triu is True:
        ewald_matrix = np.triu(ewald_matrix)

    return ewald_matrix


# In[6]:


def compute_discrete_ewald_matrix(structure, charge_options_by_Z, ewald_matrix=None):
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
    if ewald_matrix is None:
        ewald_matrix = compute_ewald_matrix_fast(structure,triu=True)

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


# In[7]:


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


# In[8]:


def build_qubo_discrete_from_Ewald_IP(ewald_discrete,buckingham_matrix):
    Q = ewald_discrete + buckingham_matrix

    return Q


# In[9]:


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


# In[10]:


def build_QUBO(structure, threshold_li=0, prox_penalty=0):
    
    structure_tmp = copy.deepcopy(structure)
    structure_tmp.add_site_property("charge", [1.0] * len(structure))
    # ewald_matrix = compute_ewald_matrix_fast(structure,triu=True)
    ewald = EwaldSummation(structure_tmp, eta=None, w=1)

    ewald_matrix = ewald.total_energy_matrix
    ewald_matrix = np.triu(ewald_matrix,1)


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
  
    ewald_discrete, expanded_charges, expanded_matrix = compute_discrete_ewald_matrix(structure, charges, ewald_matrix=ewald_matrix)

    species_dict = {'Mn': ['Mn', 'Tc']}  # Mn sites can be either Mn4+ (Mn) or Mn3+ (Tc)
    buckingham_dict = {'Li-O':[426.480 ,    0.3000  ,   0.00],
                        'Mn-O':[3087.826    ,   0.2642 ,    0.00], # This is the Mn4+
                        'Tc-O':[1686.125  ,    0.2962 ,    0.00], # This is the Mn3+
                        'O-O' : [22.410  ,     0.6937,   32.32]
                        }
    buckingham_discrete, species_vector = compute_buckingham_matrix_discrete(
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

    return QUBO, li_indices, mn_indices


# In[11]:


# def mace_optgeom(structures,mace_io_path):

#     if not os.path.exists(mace_io_path):
#         os.makedirs(mace_io_path)

#     energies = []
#     opt_structures = []

#     for i, structure in enumerate(structures):
#         structure = AseAtomsAdaptor().get_atoms(structure)
#         # Set up the single-point calculator
#         sp_mace = SinglePoint(
#             struct=structure.copy(),
#             arch="mace_mp",
#             device="cpu",
#             model_path="small",
#             calc_kwargs={"default_dtype": "float64"},
#             properties="energy"
#         )

#         # Run geometry optimization
#         optimized = GeomOpt(
#             struct=sp_mace.struct,
#             fmax=0.1,
#             filter_func=None
#         )
#         optimized.run()

#         # Write to CIF
#         cif_filename = os.path.join(mace_io_path,f"optimized_structure_{i}.cif")
#         opt_structures.append(AseAtomsAdaptor().get_structure(optimized.struct))
#         write(cif_filename, optimized.struct)

#         # Store energy
#         energy = optimized.struct.info.get("energy", None)
#         energies.append(energy)

#     return opt_structures, energies

    


# In[12]:


def add_contsraint_to_QUBO(QUBO,N_li,one_hot_value,weight,li_indices, mn_indices):

    

    # One-hot
    QUBO_constraints = np.zeros(QUBO.shape)

    for i in mn_indices[::2]:
        QUBO_constraints[i][i] += -one_hot_value
        QUBO_constraints[i][i+1] += 2*one_hot_value


    for i in li_indices:
        QUBO_constraints[i][i] += weight*(1-2*N_li)
        for j in li_indices:
            if j > i:
                QUBO_constraints[i][j] += weight*2

    # (10) = Mn4+ (01) = Mn3+
    for i in mn_indices[1::2]:
        QUBO_constraints[i][i] += weight*(1-2*N_li)
        for j in mn_indices:
            if j > i:
                QUBO_constraints[i][j] += weight*2

    QUBO_full = QUBO + QUBO_constraints

    return QUBO_full


# In[13]:


def build_bin_pol(QUBO_full):

    N = QUBO_full.shape[0]

    limno2_poly = BinPol()

    for i in range(N):
        limno2_poly.set_term(QUBO_full[i][i],(i,))
        for j in range(i,N):
            limno2_poly.set_term(QUBO_full[i][j],(i,j))
    
    return limno2_poly


# In[14]:


def solve_QUBO(limno2_poly,number_iterations,number_runs):
    
    solver = QUBOSolverCPU(
    number_iterations=number_iterations,
    number_runs=number_runs,
    graphics=GraphicsDetail.ALL,
    scaling_action=ScalingAction.AUTO_SCALING,
    scaling_bit_precision=62)

    solution_list = solver.minimize(limno2_poly)

    return solution_list

# print('duration', solution_list.solver_times.duration_execution)


# In[15]:


def classical_energy(x,q):
    # x is the binary vector
    # q is the qubo matrix

    E_tmp = np.matmul(x,q)
    E_classical = np.sum(x*E_tmp)
    
    return E_classical


# In[16]:


def filter_valid_structures(df, N_li, N_structures_opt):
    """
    Filters a DataFrame of solutions to retain only rows where:
    - num_li == num_mn == N_li
    - not_one_hot == 0

    Parameters:
    - df: pd.DataFrame, output from make_df_solutions_all
    - N_li: int, desired number of Li and Mn sites

    Returns:
    - pd.DataFrame with valid solutions, sorted by energy
    """
    filtered_df = df[
        (df['num_li'] == N_li) &
        (df['num_mn'] == N_li) &
        (df['not_one_hot'] == 0)
    ].copy()

    N_rows = filtered_df.shape[0]

    if N_rows < N_structures_opt:
        return filtered_df.sort_values(by='QUBO_energy').reset_index(drop=True), N_rows
    else:
        return filtered_df.sort_values(by='QUBO_energy').reset_index(drop=True), N_structures_opt


# In[17]:


def make_df_solutions_all(solution_list,QUBO, li_indices, mn_indices):
    records = []

    for sol in solution_list.solutions:
        config = np.array(sol.configuration)
        energy = sol.energy
        num_li = np.sum(config[li_indices])
        num_mn = np.sum(config[mn_indices[1::2]])
        not_one_hot = np.sum(config[mn_indices[1::2]] + config[mn_indices[::2]] != 1)
        
        records.append({
            'energy': float(energy),
            'num_li': int(num_li),
            'num_mn': int(num_mn),
            'not_one_hot': int(not_one_hot),
            'configuration': tuple(config),  # Make it hashable for grouping
        })

    # Create DataFrame
    df = pd.DataFrame(records)

    # Group by configuration and count how many times each appears
    grouped_df = (
        df.groupby('configuration')
        .agg({
            'energy': 'mean',
            'num_li': 'first',
            'num_mn': 'first',
            'not_one_hot': 'first',
        })
        .reset_index()
    )

    # Add count column
    counts = df['configuration'].value_counts().rename('count')
    grouped_df = grouped_df.merge(counts, left_on='configuration', right_index=True)

    # Add QUBO_energy
    grouped_df['QUBO_energy'] = grouped_df['configuration'].apply(lambda x: classical_energy(np.array(x), QUBO))

    # Compute relative energy wrt QUBO_energy
    min_qubo_energy = grouped_df['QUBO_energy'].min()
    grouped_df['relative_energy'] = grouped_df['QUBO_energy'] - min_qubo_energy

    # Reorder columns
    cols = list(grouped_df.columns)
    energy_idx = cols.index('energy')
    cols.insert(energy_idx + 1, cols.pop(cols.index('QUBO_energy')))
    cols.insert(energy_idx + 2, cols.pop(cols.index('relative_energy')))
    grouped_df = grouped_df[cols]

    # Sort by increasing energy
    grouped_df = grouped_df.sort_values(by='energy').reset_index(drop=True)

    return grouped_df


# In[18]:


def select_configurations(df,N_structures_opt):
    selected_df = df.head(N_structures_opt)

    selected_configs = [np.array(cfg) for cfg in selected_df['configuration']]

    return selected_configs


# In[19]:


def map_to_chemical_numbers(selected_configs, li_indices, mn_indices):
    li_keep = []
    mn_list = []
    tc_list = []

    for config in selected_configs:  # config_list is your list of binary arrays
        # Li positions
        li_keep_single = [i for i in li_indices if config[i] == 1]
        li_keep.append(li_keep_single)

        # Mn/Tc one-hot decoding
        mn_list_single = []
        tc_list_single = []
        for mn_idx, (i0, i1) in enumerate(zip(mn_indices[::2], mn_indices[1::2])):
            if config[i0] == 1 and config[i1] == 0:
                mn_list_single.append(mn_idx)
            elif config[i0] == 0 and config[i1] == 1:
                tc_list_single.append(mn_idx)
            # Else: not one-hot → invalid, can skip or log

        mn_list.append(mn_list_single)
        tc_list.append(tc_list_single)
    
    li_keep = np.array(li_keep)
    mn_list = np.array(mn_list)
    tc_list = np.array(tc_list)

    return li_keep, mn_list, tc_list


# In[20]:


def build_structures_to_opt(initial_structure, initial_grid, li_keep, tc_list, N_structures_opt):

    structure_original = copy.deepcopy(initial_structure)
    lattice = structure_original.lattice

    tc_indices_abs = tc_list - len(li_keep[0])

    all_structures_new = []

    for i in range(N_structures_opt):

        structure_new = copy.deepcopy(initial_structure)

        li_grid_coord_new = initial_grid[li_keep[i]]
        li_grid_new = Structure(lattice, [3]*len(li_grid_coord_new),li_grid_coord_new,coords_are_cartesian=False)
        
        for tc_site in tc_indices_abs[i]:
            structure_new.replace(tc_site,43)
        
        all_structures_new.append(Structure.from_sites(li_grid_new.sites+structure_new.sites))
    
    return all_structures_new


# In[21]:


def write_gulp_input(structure, filename="gulp_input.gin"):
    with open(filename, "w") as f:
        f.write("sp opti fbfgs conp #property phon comp\n")
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


# In[22]:


def write_all_gulp_input(all_structures_new,iteration,input_name='gulp_klmc.gin',gulp_io_path='data/gulp_input_files/'):
    submission_lines = []
    submission_lines.append(f'cd {gulp_io_path}')
    
    for i, structure_new in enumerate(all_structures_new):
        folder = os.path.join(gulp_io_path,f'structure_{i}')
        os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
        file_name = os.path.join(folder, input_name)
        write_gulp_input(structure_new, file_name)
        
        
        # Add GULP execution commands to the submission script
        submission_lines.append(f"cd structure_{i}")
        submission_lines.append("/work/e05/e05/bcamino/klmc/KLMC3-libgulp-6.1.2/Src/_build_libgulp/gulp.x")
        submission_lines.append("cd ../")  # Adjust if your script is in a subfolder

    # Write all the lines to a shell script
    with open(os.path.join(gulp_io_path,"gulp_submission.sh"), "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("\n".join(submission_lines))
        f.write("\n")


# In[48]:


def write_all_gulp_input_klmc(all_structures_new,gulp_io_path):
    
    run_folder = os.path.join(gulp_io_path,'run')

    for i, structure_new in enumerate(all_structures_new):
        os.makedirs(run_folder, exist_ok=True)  # Create folder if it doesn't exist
        file_name = os.path.join(run_folder, f'A{i}.gin')
        write_gulp_input(structure_new, file_name)
    EXE = "CHANGE THIS"    
    if len(structure_new) > 127:
        n = 128
    else:
        n = len(structure_new)+1
    submission_lines = []
    submission_lines += [
    "export OMP_NUM_THREADS=1",
    'EXE="/work/e05/e05/bcamino/klmc_exe/klmc3.062024.x"',
    f"srun -n {n} --ntasks-per-node=128 --cpus-per-task=1 --distribution=block:block --hint=nomultithread --exact ${EXE} 1> stdout 2> stderr",
    "mkdir result",
    "mv A* ./result",
    "",
    "mkdir log",
    "mv master.log ./log",
    "mv workgroup*.log ./log"
]
    
    # Write all the lines to a shell script
    with open(os.path.join(gulp_io_path,"SLURM_js.sh"), "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("\n".join(submission_lines))
        f.write("\n")
    
    taskfarm_file = []
    taskfarm_file += [
    "task_start 0",
    f"task_end {len(structure_new)-1}",
    "cpus_per_worker 1",
    "application gulp"
]
    with open(os.path.join(gulp_io_path,"taskfarm.config"), "w") as f:
        f.write("\n".join(taskfarm_file))
    


# In[24]:


def parse_gulp_to_pymatgen(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines[::-1]:
        if 'Total lattice energy ' and ' eV ' in line:
            energy = float(line.split()[-2])
            break
 
    for line in lines:
        if 'Total number atoms/shells' in line:
            n_atoms = int(line.strip().split()[-1])
            break
    
    # --- 1. Extract lattice parameters ---
    a = b = c = alpha = beta = gamma = None
    for line in lines:
        if "Final cell parameters and derivatives" in line:
            continue
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
            break  # we’re done once all 3 are found
 
    if None in (a, b, c, alpha, beta, gamma):
        raise ValueError("Could not parse complete lattice parameters.")

    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    # --- 2. Extract atomic positions ---
    species = []
    coords = []
    parsing_atoms = False
    for i,line in enumerate(lines):
        if "Final fractional coordinates of atoms" in line:
            parsing_atoms = True
            continue
        if parsing_atoms:
            # if not line.strip():
                # break  # blank line → end of section
            tokens = line.strip().split()
            if len(tokens) > 0:
                if tokens[0] == '1':
                    for j in range(n_atoms):
                        tokens = lines[i+j].strip().split()
                        species.append(tokens[1])

                        coords.append([float(tokens[3]), float(tokens[4]), float(tokens[5])])
                    break
                      

    structure = Structure(lattice, species, coords, coords_are_cartesian=False)
    structure.translate_sites(np.arange(structure.num_sites),[1,1,1],to_unit_cell=True)
    
    return structure, energy


# In[46]:


def read_opt_structures(N_structures_opt, folder_path, input_name='gulp_klmc.gin'):

    output_name = input_name[:-3]+'gout'
    all_opt_structures = []
    all_opt_energy = []
    for i in range(N_structures_opt):
        file_path = os.path.join(folder_path,f'A{i}')
        file_path = os.path.join(file_path,output_name)  
        print(file_path)
        if os.path.exists(file_path):
            structure, energy = parse_gulp_to_pymatgen(file_path)
            all_opt_structures.append(structure)
            all_opt_energy.append(energy)

    return all_opt_structures, all_opt_energy
        


# In[26]:


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


# In[27]:


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


# In[28]:


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


# In[29]:


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


# In[30]:


def average_close_points(symmetrised_coords, threshold):
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

    for i in range(len(symmetrised_coords)):
        if visited[i]:  # Skip if already processed
            continue

        # Find all close points (including itself)
        close_points = np.where(distances[i] < threshold)[0]
        visited[close_points] = True  # Mark as processed

        # Compute the average of these points
        avg_coord = np.mean(symmetrised_coords[close_points], axis=0)
        averaged_coords.append(avg_coord)

    return np.array(averaged_coords)


# In[31]:


def build_new_structural_model(opt_structures, M, N_positions_final, initial_structure, threshold, return_stats = False, fix_angles=True):


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

        lattice_new = Lattice.from_lengths_and_angles(
            [a_mean, b_mean, c_mean],
            [alpha, beta, gamma]
        )

    else:
        lattice_all = []
        for structure in opt_structures:
            lattice_all.append(structure.lattice.matrix)
        lattice_new = np.mean(lattice_all,axis=0)

    mn_coord_new = []
    o_coord_new = []

    for structure in opt_structures:
        structure.replace_species({'Tc':'Mn'})
        mn_indices_new = np.where(np.array(structure.atomic_numbers)==25)[0]
        o_indices_new = np.where(np.array(structure.atomic_numbers)==8)[0]

        mn_coord_new.append(structure.frac_coords[mn_indices_new]%1)
        o_coord_new.append(structure.frac_coords[o_indices_new]%1)

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
        
        li_index = np.where(np.array(structure.atomic_numbers) == 3)[0]

        li_coords = structure.frac_coords[li_index]
        li_coords_all.extend(li_coords)
            
    li_coords_all = unwrap_frac_coords(li_coords_all)

    li_coords_all = np.array(li_coords_all)
    grid = compute_probability_grid(li_coords_all, M)

    # plot_probability_grid(grid)

    centers = find_fractional_centers(M)
    top_centers = find_top_x_points(grid,centers,N_positions_final)

    coord_top = []
    for line in top_centers:
        coord_top.append(line[0])
    coord_top = np.array(coord_top)

    #Symmetrise
    symmops = SpacegroupAnalyzer(initial_structure).get_symmetry_operations()
    num_symmops = len(symmops)

    symmetrised_coords = []
    for symmop in symmops:
        for coord in coord_top:
            symmetrised_coords.append(symmop.operate(coord)%1)


    symmetrised_coords = np.array(symmetrised_coords)

    averaged_symmetrised_coords = average_close_points(symmetrised_coords, threshold)
    # vview(Structure(initial_structure.lattice.matrix,[1]*N_positions_final,coord_top))
    # vview(Structure(initial_structure.lattice.matrix,[1]*len(averaged_symmetrised_coords),averaged_symmetrised_coords))

    mn_sites = []
    o_sites = []
    li_sites = []

    # atomic_numbers = [3]*len(averaged_symmetrised_coords)+[24]*len(mn_coord_new)+[8]*

    li_sites = []
    mn_sites = []
    o_sites = []

    lattice = Lattice(lattice_new)  # Wrap your matrix once

    # Then use it everywhere:
    li_sites = [PeriodicSite('Li', coord, lattice) for coord in averaged_symmetrised_coords]
    mn_sites = [PeriodicSite('Mn', coord, lattice) for coord in mn_coord_average]
    o_sites = [PeriodicSite('O', coord, lattice) for coord in o_coord_average]

    # Combine and build
    all_sites = li_sites + mn_sites + o_sites
    structure = Structure.from_sites(all_sites)

    return structure, averaged_symmetrised_coords
    


# In[32]:


def mace_optgeom(structures,mace_io_path):

    if not os.path.exists(mace_io_path):
        os.makedirs(mace_io_path)

    energies = []
    opt_structures = []

    for i, structure in enumerate(structures):
        structure = AseAtomsAdaptor().get_atoms(structure)
        # Set up the single-point calculator
        sp_mace = SinglePoint(
            struct=structure.copy(),
            arch="mace_mp",
            device="cpu",
            model_path="small",
            calc_kwargs={"default_dtype": "float64"},
            properties="energy"
        )

        # Run geometry optimization
        optimized = GeomOpt(
            struct=sp_mace.struct,
            fmax=0.1,
            filter_func=None
        )
        optimized.run()

        # Write to CIF
        cif_filename = os.path.join(mace_io_path,f"optimized_structure_{i}.cif")
        opt_structures.append(AseAtomsAdaptor().get_structure(optimized.struct))
        write(cif_filename, optimized.struct)

        # Store energy
        energy = optimized.struct.info.get("energy", None)
        energies.append(energy)

    return opt_structures, energies

    


# # THE PROGRAM

# ## The parameters

# In[41]:


N_li = 2

N_initial_grid=10
min_dist_grid=1.

threshold_li=1.5
prox_penalty=1000

one_hot_value = 200
weight = 500

N_structures_opt = 2

number_iterations = 1000
number_runs = 100

input_name='gulp_klmc.gin'
gulp_io_path='klmc/'
mace_io_path='mace_io_files'

M = 20 #grid definition
N_positions_final = 10

threshold = 0.1  # THIS IS AN IMPORTANT PARAMETER

num_iterations = 5


# ## Loop

# In[34]:


def save_output(output_dir, df, new_structure, QUBO, QUBO_full):
    
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(os.path.join(output_dir,'df_results.csv'), index=False)

    new_structure.to(filename=os.path.join(output_dir,'new_structure.cif'))

    np.savetxt(os.path.join(output_dir,'QUBO.csv'), QUBO, delimiter=',')
    np.savetxt(os.path.join(output_dir,'QUBO_full.csv'), QUBO_full, delimiter=',')


    


# ### GULP

# In[ ]:


# def add_proximity_constraint_li(structure, QUBO, li_indices, threshold_li=1.,prox_penalty=200):
#     # THIS OONLY WORKS IF Li ATOMS ARE FIRST IN THE STRUCTURE
    
#     dm = structure.distance_matrix
#     num_sites = QUBO.shape[0]
    
#     # Create a mask for all (i,j) pairs where both i and j are Li
#     li_mask = np.zeros((num_sites, num_sites), dtype=bool)
#     li_mask[np.ix_(li_indices, li_indices)] = True

#     # Apply the distance threshold
#     below_thresh_mask = dm < threshold

#     # Combine masks
#     final_mask = li_mask & below_thresh_mask

#     # Create constraint matrix
#     prox_constraint = np.where(final_mask, prox_penalty, 0)
#     np.fill_diagonal(prox_constraint,0)

#     return QUBO + prox_constraint


# In[ ]:


