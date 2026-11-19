"""ipSAE/pDockQ scoring: the kernel and the alphafold3 token preparation."""
import numpy as np

from colabfold import ipsae


def test_kernel_matches_original_ipSAE():
    """Pure-protein chains reproduce the original ipSAE per chain pair."""
    # two chains, 3 residues each, all PAE under the cutoff
    asym_id = np.array([1, 1, 1, 2, 2, 2])
    pae = np.full((6, 6), 2.0)
    plddt = np.full(6, 90.0)
    atom_positions = np.zeros((6, 37, 3))
    atom_positions[:, ipsae.CA_IDX, :] = np.arange(6)[:, None] * [1.0, 0.0, 0.0]
    atom_positions[:, ipsae.CB_IDX, :] = np.arange(6)[:, None] * [1.0, 0.0, 0.0] + [0.0, 1.0, 0.0]
    atom_mask = np.zeros((6, 37), dtype=bool)
    atom_mask[:, ipsae.CA_IDX] = True
    atom_mask[:, ipsae.CB_IDX] = True

    scores = ipsae.get_interface_scores(pae, plddt, asym_id, atom_positions, atom_mask)
    # every cross-chain PAE is 2.0, so each row's mean TM score is
    # 1 / (1 + (2 / d0) ** 2) with d0 from 3 qualifying residues; alphafold2's
    # 1-based asym ids map chain 1 to B and chain 2 to C
    d0 = max(1.0, 1.24 * (26.0 - 15.0) ** (1 / 3) - 1.8)
    expected = round(1.0 / (1.0 + (2.0 / d0) ** 2), 6)
    assert scores["ipsae"] == {"B-C": expected, "C-B": expected}
    # the residues sit within 8 A of each other, so there are contacts
    assert 0.0 < scores["pdockq"]["B-C"] < 1.0
    assert scores["pdockq2"]["B-C"] > 0.5


def test_kernel_unplaced_and_glycine():
    """Unplaced residues make no contacts, glycine falls back to its CA."""
    asym_id = np.array([1, 1, 2, 2])
    pae = np.full((4, 4), 2.0)
    plddt = np.full(4, 90.0)
    atom_positions = np.zeros((4, 37, 3))
    atom_positions[:, ipsae.CA_IDX, :] = np.arange(4)[:, None] * [50.0, 0.0, 0.0]
    atom_positions[0, ipsae.CB_IDX, :] = [50.0, 1.0, 0.0]
    atom_mask = np.zeros((4, 37), dtype=bool)
    atom_mask[:, ipsae.CA_IDX] = True
    atom_mask[0, ipsae.CB_IDX] = True  # residue 3 (glycine-like) has no CB

    scores = ipsae.get_interface_scores(pae, plddt, asym_id, atom_positions, atom_mask)
    # chain 1's residues sit 50 A away from chain 2's, and residue 2 is unplaced
    assert scores["pdockq2"] == {"B-C": 0.0, "C-B": 0.0}
    assert scores["pdockq"] == {"B-C": 0.0}


def test_kernel_nucleic_fixed_d0():
    """A nucleic scored chain keeps ipSAE2's fixed d0, under the PAE cutoff."""
    asym_id = np.array([1, 1, 2, 2])
    pae = np.full((4, 4), 11.0)
    np.fill_diagonal(pae, 0.0)
    plddt = np.full(4, 90.0)
    atom_positions = np.zeros((4, 37, 3))
    atom_mask = np.zeros((4, 37), dtype=bool)

    scores = ipsae.get_interface_scores(
        pae, plddt, asym_id, atom_positions, atom_mask,
        chain_labels={1: "A", 2: "B"},
        chain_types={1: "prot", 2: "nuc"})
    # only the protein-aligned direction is scored, at the fixed d0 of 2 A
    expected = round(1.0 / (1.0 + (11.0 / ipsae.NON_PROTEIN_D0) ** 2), 6)
    assert scores["ipsae"] == {"A-B": expected}
    # pDockQ/pDockQ2 stay protein-only
    assert scores["pdockq"] == {} and scores["pdockq2"] == {}


def test_kernel_ligand_without_cutoff():
    """Ligand scored units keep every PAE and score at the fixed d0."""
    asym_id = np.array([1, 1, 2, 2])
    pae = np.full((4, 4), 30.0)  # far beyond any cutoff
    np.fill_diagonal(pae, 0.0)
    plddt = np.full(4, 90.0)
    atom_positions = np.zeros((4, 37, 3))
    atom_mask = np.zeros((4, 37), dtype=bool)

    scores = ipsae.get_interface_scores(
        pae, plddt, asym_id, atom_positions, atom_mask,
        chain_labels={1: "A", 2: "B"},
        chain_types={1: "prot", 2: "lig"})
    expected = round(1.0 / (1.0 + (30.0 / ipsae.NON_PROTEIN_D0) ** 2), 6)
    assert scores["ipsae"] == {"A-B": expected}


def test_kernel_mixed_pair_direction():
    """A mixed-type pair is scored from the more complex molecule's frames."""
    asym_id = np.array([1, 1, 2, 2])
    # aligning on chain 2 would give the better score, but only chain 1's
    # frames may be used for a protein-ligand pair
    pae = np.full((4, 4), 1.0)
    pae[2:, :2] = 10.0  # bad PAE from chain 2's frames onto chain 1
    np.fill_diagonal(pae, 0.0)
    plddt = np.full(4, 90.0)
    atom_positions = np.zeros((4, 37, 3))
    atom_mask = np.zeros((4, 37), dtype=bool)

    scores = ipsae.get_interface_scores(
        pae, plddt, asym_id, atom_positions, atom_mask,
        chain_labels={1: "P", 2: "L"},
        chain_types={1: "prot", 2: "lig"})
    expected = round(1.0 / (1.0 + (1.0 / ipsae.NON_PROTEIN_D0) ** 2), 6)
    assert scores["ipsae"] == {"P-L": expected}


def test_kernel_ion_has_no_frames():
    asym_id = np.array([1, 2])
    pae = np.full((2, 2), 1.0)
    np.fill_diagonal(pae, 0.0)
    plddt = np.full(2, 90.0)
    atom_positions = np.zeros((2, 37, 3))
    atom_mask = np.zeros((2, 37), dtype=bool)

    scores = ipsae.get_interface_scores(
        pae, plddt, asym_id, atom_positions, atom_mask,
        chain_labels={1: "P", 2: "I"},
        chain_types={1: "prot", 2: "ion"})
    assert scores["ipsae"] == {"P-I": round(1.0 / (1.0 + 0.25), 6)}


def test_format_ipsae_maxes_directions():
    assert ipsae.format_ipsae({"A-B": 0.4, "B-A": 0.7, "A-C": 0.2}) == "A-B:0.7,A-C:0.2"
    assert ipsae.format_ipsae({"A-B": 0.412345}) == "0.412"


class _FakeFull:
    def __init__(self, pae, token_chain_ids, token_res_ids):
        self.pae = pae
        self.token_chain_ids = token_chain_ids
        self.token_res_ids = token_res_ids


class _FakeInferenceResult:
    def __init__(self, predicted_structure, full):
        self.predicted_structure = predicted_structure
        self.full = full


def _cif(chains, atoms):
    """A minimal mmCIF: ``chains`` maps a label to (entity id, chain type), and
    ``atoms`` lists (chain, res id, res name, atom name, x, y, z, b-factor) with
    an optional element."""
    rows = []
    for i, atom in enumerate(atoms, 1):
        chain, res_id, res_name, name, x, y, z, b = atom[:8]
        element = atom[8] if len(atom) > 8 else (
            "N" if name.startswith("N") else "P" if name == "P" else "O" if name.startswith("O") else "C")
        entity_id = chains[chain][0]
        rows.append(f"ATOM {i} {element} {name} . {res_name} "
                    f"{chain} {entity_id} {res_id} ? {x:.3f} {y:.3f} {z:.3f} 1.00 {b:.2f} "
                    f"{res_id} {chain} 1")
    entities = "\n".join(f"{entity} {kind}" for entity, (kind, _) in
                         [(c[0], (c[1], c[2])) for c in chains.values()])
    asym = "\n".join(f"{label} {entity}" for label, entity in
                     [(label, c[0]) for label, c in chains.items()])
    polymers = "\n".join(f"{entity} {poly}" for entity, poly in
                         [(c[0], c[2]) for c in chains.values() if c[1] == "polymer"])
    return f"""data_test
#
loop_
_entity.id
_entity.type
{entities}
#
loop_
_entity_poly.entity_id
_entity_poly.type
{polymers}
#
loop_
_struct_asym.id
_struct_asym.entity_id
{asym}
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
{chr(10).join(rows)}
#
"""


# one glycine and one alanine on chain A, DNA on chain B, a two-atom ligand on C
_PROTEIN_DNA_LIGAND_CIF = _cif(
    {"A": (1, "polymer", "polypeptide(L)"),
     "B": (2, "polymer", "polydeoxyribonucleotide"),
     "C": (3, "non-polymer", "non-polymer")},
    [("A", 1, "GLY", "CA", 1.0, 0.0, 0.0, 90.0),
     ("A", 1, "GLY", "N", 0.0, 0.0, 0.0, 90.0),
     ("A", 2, "ALA", "CA", 5.0, 0.0, 0.0, 90.0),
     ("A", 2, "ALA", "CB", 5.0, 1.0, 0.0, 90.0),
     ("B", 1, "DA", "P", 0.0, 5.0, 0.0, 90.0),
     ("B", 2, "DA", "P", 2.0, 5.0, 0.0, 90.0),
     ("C", 1, "LIG", "C", 8.0, 0.0, 0.0, 90.0),
     ("C", 1, "LIG", "O", 9.0, 0.0, 0.0, 90.0)])


# the protein chain twice, plus the two-atom ligand
_PROTEIN_PROTEIN_LIGAND_CIF = _cif(
    {"A": (1, "polymer", "polypeptide(L)"),
     "B": (1, "polymer", "polypeptide(L)"),
     "C": (2, "non-polymer", "non-polymer")},
    [("A", 1, "GLY", "CA", 1.0, 0.0, 0.0, 90.0),
     ("A", 2, "ALA", "CA", 5.0, 0.0, 0.0, 90.0),
     ("A", 2, "ALA", "CB", 5.0, 1.0, 0.0, 90.0),
     ("B", 1, "GLY", "CA", 1.0, 3.0, 0.0, 90.0),
     ("B", 2, "ALA", "CA", 5.0, 3.0, 0.0, 90.0),
     ("B", 2, "ALA", "CB", 5.0, 4.0, 0.0, 90.0),
     ("C", 1, "LIG", "C", 8.0, 0.0, 0.0, 90.0),
     ("C", 1, "LIG", "O", 9.0, 0.0, 0.0, 90.0)])


def test_af3_tokens_map_to_residues():
    from colabfold.alphafold3.predict import _interface_scores, _molecule_types

    structure = _structure_of(_PROTEIN_DNA_LIGAND_CIF)
    assert _molecule_types(structure) == {"A": "prot", "B": "nuc", "C": "lig"}

    # tokens: 2 protein residues, 2 nucleotides, 2 ligand atoms
    pae = np.full((6, 6), 1.0)
    np.fill_diagonal(pae, 0.0)
    full = _FakeFull(pae, ["A", "A", "B", "B", "C", "C"], [1, 2, 1, 2, 1, 1])

    scores = _interface_scores(_FakeInferenceResult(structure, full), full)
    # protein-DNA and protein-ligand only from the protein's frames,
    # DNA-ligand only from the DNA's frames
    assert set(scores["ipsae"]) == {"A-B", "A-C", "B-C"}
    assert scores["pdockq"] == {} and scores["pdockq2"] == {}


def test_af3_water_tokens_are_dropped():
    """Tokens of a chain with no molecule type must not shift the others."""
    from colabfold.alphafold3.predict import _interface_scores

    structure = _structure_of(_PROTEIN_DNA_LIGAND_CIF)

    # the water tokens have no residue in the structure to anchor to
    pae = np.full((8, 8), 1.0)
    np.fill_diagonal(pae, 0.0)
    full = _FakeFull(pae, ["A", "A", "B", "B", "C", "C", "W", "W"],
                     [1, 2, 1, 2, 1, 1, 1, 2])

    scores = _interface_scores(_FakeInferenceResult(structure, full), full)
    assert set(scores["ipsae"]) == {"A-B", "A-C", "B-C"}


def test_af3_lone_metal_is_an_ion():
    from colabfold.alphafold3.predict import _molecule_types

    structure = _structure_of(_cif(
        {"A": (1, "polymer", "polypeptide(L)"), "B": (2, "non-polymer", "non-polymer")},
        [("A", 1, "GLY", "CA", 1.0, 0.0, 0.0, 90.0),
         ("A", 2, "ALA", "CA", 5.0, 0.0, 0.0, 90.0),
         ("A", 2, "ALA", "CB", 5.0, 1.0, 0.0, 90.0),
         ("B", 1, "MG", "MG", 8.0, 0.0, 0.0, 90.0, "MG")]))
    assert _molecule_types(structure) == {"A": "prot", "B": "ion"}


def test_af3_protein_tokens_get_ca_and_cb():
    from colabfold.alphafold3.predict import _interface_scores

    structure = _structure_of(_PROTEIN_PROTEIN_LIGAND_CIF)

    # tokens: 2 + 2 protein residues, 2 ligand atoms
    pae = np.full((6, 6), 1.0)
    np.fill_diagonal(pae, 0.0)
    full = _FakeFull(pae, ["A", "A", "B", "B", "C", "C"], [1, 2, 1, 2, 1, 1])

    scores = _interface_scores(_FakeInferenceResult(structure, full), full)
    # glycine falls back to its CA, alanine has a CB; all residues are within
    # 8 A across the A-B interface, so it is a confident contact
    assert scores["pdockq2"]["A-B"] > 0.7
    assert set(scores["pdockq2"]) == {"A-B", "B-A"}  # no pDockQ2 for the ligand


def _structure_of(mmcif_string):
    from alphafold3.structure import parsing

    return parsing.from_mmcif(mmcif_string)
