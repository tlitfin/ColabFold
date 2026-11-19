"""Interface confidence scores."""

# Reimplemented from https://github.com/DunbrackLab/IPSAE (MIT); nucleic
# acids, ligands and ions are scored following
# https://github.com/DunbrackLab/IPSAE2 (MIT).

import numpy as np

# Angstrom.
DEFAULT_PAE_CUTOFF = 15.0
PDOCKQ_DIST_CUTOFF = 8.0

# alphafold.common.residue_constants.atom_order of the alphafold2 package,
# whose atom_types start ("N", "CA", "C", "CB", "O", ...): kept literal so
# this module needs no alphafold2 install.
CA_IDX = 1
CB_IDX = 3
# alphafold.common.protein.PDB_CHAIN_IDS.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# ipSAE2's molecule types. A pair of different types is scored from the
# alignment frames of the more complex molecule, and an ion offers no
# alignment frame at all.
TYPE_ORDER = {"prot": 0, "nuc": 1, "lig": 2, "ion": 3}
# ipSAE2 keeps a fixed d0 for scored chains that hold no protein.
NON_PROTEIN_D0 = 2.0
# Elements whose lone atoms ipSAE2 scores as ions.
ION_ELEMENTS = frozenset(
    "LI NA K RB CS MG CA SR BA MN FE CO NI CU ZN CD AL GA HG PB F CL BR I".split())


def _ptm_score(pae, d0):
    """TM-score kernel."""
    return 1.0 / (1.0 + (pae / d0) ** 2.0)


def _calc_d0_array(n_res):
    """TM-score d0."""
    length = np.maximum(26.0, np.asarray(n_res, dtype=np.float64))
    return np.maximum(1.0, 1.24 * (length - 15.0) ** (1.0 / 3.0) - 1.8)


def get_interface_scores(pae, plddt, asym_id, atom_positions, atom_mask,
                         pae_cutoff=DEFAULT_PAE_CUTOFF,
                         chain_labels=None, chain_types=None):
    """Return ipSAE, pDockQ and pDockQ2 per chain pair.

    ``chain_labels`` and ``chain_types`` map a chain of ``asym_id`` to its
    output label and molecule type ("prot", "nuc", "lig", "ion"), for backends
    whose scoring units are not alphafold2's protein residues. Without them
    every chain is a protein and the original ipSAE is computed.
    """
    asym_id = np.asarray(asym_id)
    unique_chains = np.unique(asym_id)
    if len(unique_chains) < 2:
        return {}

    pae = np.asarray(pae)
    plddt = np.asarray(plddt, dtype=np.float64)
    atom_positions = np.asarray(atom_positions, dtype=np.float64)
    atom_mask = np.asarray(atom_mask)

    # Use CA for glycine.
    has_cb = atom_mask[:, CB_IDX] > 0.5
    cb_coords = np.where(has_cb[:, None],
                         atom_positions[:, CB_IDX], atom_positions[:, CA_IDX])
    # Exclude unplaced residues.
    placed = atom_mask[:, CA_IDX] > 0.5

    if chain_labels is None:
        chain_labels = {chain: PDB_CHAIN_IDS[int(chain)] for chain in unique_chains}
    chain_index = {chain: np.flatnonzero(asym_id == chain) for chain in unique_chains}
    types = {chain: (chain_types or {}).get(int(chain), "prot")
             for chain in unique_chains}

    ipsae, pdockq, pdockq2 = {}, {}, {}
    for chain_1 in unique_chains:
        for chain_2 in unique_chains:
            if chain_1 == chain_2:
                continue
            type_1, type_2 = types[chain_1], types[chain_2]
            # ipSAE2 scores a mixed-type pair from the more complex molecule's
            # frames only, and never from an ion's.
            if type_1 == "ion" or TYPE_ORDER[type_1] > TYPE_ORDER[type_2]:
                continue
            key = f"{chain_labels[chain_1]}-{chain_labels[chain_2]}"
            # only this pair's block: masking the whole matrix made it chains^2 * L^2
            rows, cols = np.ix_(chain_index[chain_1], chain_index[chain_2])
            pair_pae = pae[rows, cols].astype(np.float64)

            # ipSAE. The PAE cutoff applies to polymer scored units only, and a
            # scored chain without protein keeps a fixed d0 (ipSAE2).
            if type_2 in ("prot", "nuc"):
                valid_pairs = pair_pae < pae_cutoff
            else:
                valid_pairs = np.ones(pair_pae.shape, dtype=bool)
            n0_res = valid_pairs.sum(axis=1)
            if type_2 == "prot":
                d0_res = _calc_d0_array(n0_res)
            else:
                d0_res = np.full(n0_res.shape, NON_PROTEIN_D0)
            ptm_sums = (_ptm_score(pair_pae, d0_res[:, None]) * valid_pairs).sum(axis=1)
            ipsae_by_res = np.divide(ptm_sums, n0_res,
                                     out=np.zeros_like(ptm_sums), where=n0_res > 0)
            ipsae[key] = round(float(ipsae_by_res.max()), 6)

            # pDockQ and pDockQ2 are calibrated on protein interfaces only.
            if type_1 != "prot" or type_2 != "prot":
                continue

            contacts = (cb_coords[chain_index[chain_1], None, :]
                        - cb_coords[chain_index[chain_2]][None, :, :])
            distances = np.sqrt((contacts ** 2).sum(-1))
            distances[~placed[chain_index[chain_1]], :] = np.inf
            distances[:, ~placed[chain_index[chain_2]]] = np.inf
            contacts = distances <= PDOCKQ_DIST_CUTOFF
            n_contacts = contacts.sum()
            if n_contacts > 0:
                interface_plddt = np.concatenate([plddt[chain_index[chain_1]][contacts.any(axis=1)],
                                                  plddt[chain_index[chain_2]][contacts.any(axis=0)]])
                mean_plddt = interface_plddt.mean()
                pdockq_val = 0.724 / (1.0 + np.exp(
                    -0.052 * (mean_plddt * np.log10(n_contacts) - 152.611))) + 0.018
                mean_ptm = _ptm_score(pair_pae[contacts], 10.0).mean()
                pdockq2_val = 1.31 / (1.0 + np.exp(
                    -0.075 * (mean_plddt * mean_ptm - 84.733))) + 0.005
            else:
                pdockq_val, pdockq2_val = 0.0, 0.0
            if chain_1 < chain_2:  # Symmetric.
                pdockq[key] = round(float(pdockq_val), 4)
            pdockq2[key] = round(float(pdockq2_val), 4)

    return {"ipsae": ipsae, "pdockq": pdockq, "pdockq2": pdockq2}


def format_ipsae(ipsae_scores):
    """Format per-interface maxima."""
    pair_max = {}
    for key, value in ipsae_scores.items():
        chain_1, chain_2 = key.split("-")
        pair = key if chain_1 < chain_2 else f"{chain_2}-{chain_1}"
        pair_max[pair] = max(pair_max.get(pair, 0.0), value)
    if len(pair_max) == 1:
        return f"{next(iter(pair_max.values())):.3g}"
    return ",".join(f"{pair}:{value:.3g}" for pair, value in sorted(pair_max.items()))
