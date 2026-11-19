"""
Run an alphafold3-open model and write results in ColabFold's layout.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from colabfold import ipsae
from colabfold.backend import metrics_line

logger = logging.getLogger(__name__)


def _residue_groups(chains, res_ids):
    """Contiguous (chain, residue) groups of an ordered index, as (start, end)."""
    change = np.r_[True, (chains[1:] != chains[:-1]) | (res_ids[1:] != res_ids[:-1])]
    starts = np.flatnonzero(change)
    return list(zip(starts, np.append(starts[1:], len(chains))))


def _molecule_types(structure) -> Dict[str, str]:
    """ipSAE2's molecule type per output chain, from alphafold3's chain types."""
    from alphafold3.constants import mmcif_names

    protein = set(mmcif_names.PEPTIDE_CHAIN_TYPES) - {mmcif_names.PEPTIDE_NUCLEIC_ACID_CHAIN}
    nucleic = set(mmcif_names.NUCLEIC_ACID_CHAIN_TYPES)
    chains = {}
    for label, chain_type, element in zip(structure.chain_id, structure.chain_type,
                                          structure.atom_element):
        chain = chains.setdefault(label, [chain_type, 0, set()])
        chain[1] += 1
        chain[2].add(element)
    types = {}
    for label, (chain_type, num_atoms, elements) in chains.items():
        if chain_type in protein:
            types[label] = "prot"
        elif chain_type in nucleic:
            types[label] = "nuc"
        elif chain_type != mmcif_names.WATER:
            # a lone ionic atom is an ion, which offers no alignment frame
            if num_atoms == 1 and elements & ipsae.ION_ELEMENTS:
                types[label] = "ion"
            else:
                types[label] = "lig"
    return types


def _interface_scores(inference_result, full) -> Dict[str, Any]:
    """ipSAE, pDockQ and pDockQ2 per chain pair, as the alphafold2 backends
    compute them (colabfold.ipsae).

    The scoring units are alphafold3's tokens, one per polymer residue and one
    per atom of a flattened residue or ligand; nucleic acids and ligands are
    scored the way ipSAE2 does it.
    """
    structure = inference_result.predicted_structure
    token_chains = np.asarray(full.token_chain_ids)
    token_res = np.asarray(full.token_res_ids)
    num_tokens = len(token_chains)

    types = _molecule_types(structure)
    labels = [label for label in dict.fromkeys(token_chains.tolist()) if label in types]
    if len(labels) < 2:
        return {}
    keep = np.array([chain in types for chain in token_chains])
    if not keep.all():
        # tokens of a chain without a molecule type, water mostly, cannot be scored
        pae = full.pae[np.ix_(keep, keep)]
        token_chains = token_chains[keep]
        token_res = token_res[keep]
        num_tokens = int(keep.sum())
    else:
        pae = full.pae

    # atoms of chains that carry no tokens, water mostly, would shift the
    # residue order the tokens follow
    atom_chains = np.asarray(structure.chain_id)
    atom_keep = np.array([chain in types for chain in atom_chains])
    atom_chains = atom_chains[atom_keep]
    atom_res = np.asarray(structure.res_id)[atom_keep]
    atom_names = np.asarray(structure.atom_name)[atom_keep]
    coords = np.stack([structure.atom_x, structure.atom_y, structure.atom_z],
                      axis=1)[atom_keep]
    b_factors = np.asarray(structure.atom_b_factor, dtype=np.float64)[atom_keep]

    token_groups = _residue_groups(token_chains, token_res)
    atom_groups = _residue_groups(atom_chains, atom_res)
    if len(token_groups) != len(atom_groups):
        raise ValueError(f"{len(token_groups)} token residues, but "
                         f"{len(atom_groups)} residues in the structure")

    # per token: its chain, its pLDDT, and the CA/CB of a one-token residue
    codes = np.zeros(num_tokens, dtype=np.int64)
    token_plddt = np.zeros(num_tokens)
    atom_positions = np.zeros((num_tokens, 37, 3))  # 37: alphafold2's atom_type_num
    atom_mask = np.zeros((num_tokens, 37), dtype=bool)
    chain_labels, chain_types = {}, {}
    for code, label in enumerate(labels, 1):
        sel = token_chains == label
        codes[sel] = code
        chain_labels[code] = label
        chain_types[code] = types[label]
    for (token_start, token_end), (atom_start, atom_end) in zip(token_groups, atom_groups):
        if token_chains[token_start] != atom_chains[atom_start]:
            raise ValueError(f"token chain {token_chains[token_start]} comes where "
                             f"structure chain {atom_chains[atom_start]} is expected")
        atom_plddt = b_factors[atom_start:atom_end]
        if token_end - token_start == atom_end - atom_start:
            # one token per atom: a flattened residue or ligand
            token_plddt[token_start:token_end] = atom_plddt
        else:
            # one token for the whole residue: its atoms share the pLDDT
            token_plddt[token_start:token_end] = atom_plddt.mean() if atom_end > atom_start else 0.0
        if token_end - token_start == 1 and chain_types[int(codes[token_start])] == "prot":
            residue_names = atom_names[atom_start:atom_end]
            ca = np.flatnonzero(residue_names == "CA")
            # (0, 0, 0) marks an atom the model did not predict
            if len(ca) and coords[atom_start + ca[0]].any():
                atom_positions[token_start, ipsae.CA_IDX] = coords[atom_start + ca[0]]
                atom_mask[token_start, ipsae.CA_IDX] = True
                cb = np.flatnonzero(residue_names == "CB")
                if len(cb) and coords[atom_start + cb[0]].any():
                    atom_positions[token_start, ipsae.CB_IDX] = coords[atom_start + cb[0]]
                    atom_mask[token_start, ipsae.CB_IDX] = True

    return ipsae.get_interface_scores(
        pae=pae,
        plddt=token_plddt,
        asym_id=codes,
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        chain_labels=chain_labels,
        chain_types=chain_types)


def per_residue_plddt(structure) -> List[float]:
    """Mean atom b-factor per residue, in chain order."""
    b = np.asarray(structure.atom_b_factor, dtype=float)
    chain = np.asarray(structure.chain_id)
    res = np.asarray(structure.res_id)
    if len(b) == 0:
        return []
    keys = np.asarray([f"{c}\t{r}" for c, r in zip(chain, res)])
    starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    return [float(v) for v in np.add.reduceat(b, starts) / np.diff(np.append(starts, len(b)))]


def scores_of(inference_result, ranking_score: float) -> Dict[str, Any]:
    from alphafold3.model import confidence_types

    full = confidence_types.StructureConfidenceFull.from_inference_result(inference_result)
    meta = inference_result.metadata
    pae = np.asarray(full.pae, dtype=float)
    scores = {
        "plddt": np.around(per_residue_plddt(inference_result.predicted_structure), 2).tolist(),
        "pae": np.around(pae, 2).tolist(),
        "max_pae": float(pae.max()),
        "ranking_score": float(ranking_score),
        "token_chain_ids": list(full.token_chain_ids),
    }
    for key, name in (("predicted_tm_score", "ptm"),
                      ("interface_predicted_tm_score", "iptm"),
                      ("has_clash", "has_clash"),
                      ("fraction_disordered", "fraction_disordered")):
        value = meta.get(key)
        if value is not None and np.isfinite(float(value)):
            scores[name] = float(value)
    if len(set(full.token_chain_ids)) > 1:
        try:
            interface_scores = _interface_scores(inference_result, full)
            if interface_scores:
                scores.update(interface_scores)
        except Exception as e:
            logger.warning(f"Could not compute ipSAE/pDockQ interface scores: {e}")
    return scores


def _print_line(tag: str, scores: Dict[str, Any], took: float) -> str:
    summary = {k: scores[k] for k in ("ptm", "iptm", "ranking_score") if k in scores}
    if scores["plddt"]:
        summary["mean_plddt"] = float(np.mean(scores["plddt"]))
    line = f"{tag} took {took:.1f}s" + metrics_line(summary)
    if scores.get("ipsae"):
        line += f" ipSAE={ipsae.format_ipsae(scores['ipsae'])}"
    if scores.get("pdockq2"):
        line += f" pDockQ2={ipsae.format_ipsae(scores['pdockq2'])}"
    return line


def rank_key(rank_by: str, scores: Dict[str, Any]) -> float:
    """The metric to rank on. ``auto`` is alphafold3's own ranking score."""
    if rank_by in ("auto", "ranking_score"):
        return float(scores["ranking_score"])
    if rank_by == "plddt":
        return float(np.mean(scores["plddt"])) if scores["plddt"] else float("nan")
    if rank_by == "multimer":
        rank_by = "iptm" if "iptm" in scores else "ptm"
    return float(scores.get(rank_by, scores["ranking_score"]))


def predict_structure(
    prefix: str,
    result_dir: Path,
    fold_input,
    model_runner,
    model_type: str,
    featurised_examples: List[Any],
    save_all: bool = False,
    rank_by: str = "auto",
    stop_at_score: float = 100.0,
    prediction_callback=None,
) -> Dict[str, Any]:
    import jax
    from alphafold3.model import post_processing

    ranked, best = [], -float("inf")
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        if best >= stop_at_score:
            logger.info(f"stopping early, {rank_by} reached {best:.3f}")
            break
        start = time.time()
        result = model_runner.run_inference(example, jax.random.PRNGKey(seed))
        inference_results = model_runner.extract_inference_results(
            batch=example, result=result, target_name=fold_input.name
        )
        took = time.time() - start
        for sample, inference_result in enumerate(inference_results):
            processed = post_processing.post_process_inference_result(inference_result)
            tag = f"{model_type}_seed_{seed:03d}_sample_{sample}"
            scores = scores_of(inference_result, processed.ranking_score)
            num_tokens = int(np.sum(example["seq_mask"]))
            logits = model_runner.extract_logits(result, sample, num_tokens) if save_all else None
            ranked.append((rank_key(rank_by, scores), tag, processed, scores, took,
                           model_runner.extract_embeddings(result, num_tokens), logits))
            best = max(best, ranked[-1][0])
            if prediction_callback is not None:
                prediction_callback(inference_result.predicted_structure, None,
                                    scores, example, (tag, False))

    logger.info(f"reranking models by '{rank_by}' metric")
    ranked.sort(key=lambda row: row[0], reverse=True)

    rank, metric, result_files = [], [], []
    for n, (_, tag, processed, scores, took, embeddings, logits) in enumerate(ranked):
        new_tag = f"rank_{(n + 1):03d}_{tag}"
        rank.append(new_tag)
        metric.append(scores)
        logger.info(_print_line(new_tag, scores, took))
        cif = result_dir.joinpath(f"{prefix}_{new_tag}.cif")
        cif.write_bytes(processed.cif)
        result_files.append(cif)
        scores_file = result_dir.joinpath(f"{prefix}_scores_{new_tag}.json")
        scores_file.write_text(json.dumps(scores))
        result_files.append(scores_file)
        if save_all:
            full = result_dir.joinpath(f"{prefix}_confidences_{new_tag}.json")
            full.write_bytes(processed.structure_full_data_json)
            result_files.append(full)
        if logits:
            npz = result_dir.joinpath(f"{prefix}_logits_{new_tag}.npz")
            np.savez_compressed(npz, **logits)
            result_files.append(npz)
        if embeddings:
            npz = result_dir.joinpath(f"{prefix}_representations_{new_tag}.npz")
            np.savez_compressed(npz, **embeddings)
            result_files.append(npz)
    return {"rank": rank, "metric": metric, "result_files": result_files}
