# Build a phylogenetic tree from TemPhD AMR sequences
# and compute its likelihood with Felsenstein's algorithm

# Pipeline:
#   1) Use DataFormat and use AMR-positive phage sequences
#   2) Align with MAFFT
#   3) Build a UPGMA tree from pairwise p-distances.
#   4) Use Felsenstein pruning to compute log-likelihood of
#      the alignment on that tree.

import os
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio import SeqIO

from DataFormat import DataFormat


# Configuraion of paths

DATA_ROOT = Path(r"C:\Users\gabri\OneDrive\Desktop\CompBio Project\TemPhD")
DB_NAME = "TemPhD"

OUTPUT_DIR = DATA_ROOT / "phylo_out_felsenstein"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FASTA_PATH = OUTPUT_DIR / "amr_for_felsenstein.fasta"
ALIGN_PATH = OUTPUT_DIR / "amr_for_felsenstein.aln.fasta"

# How many distinct AMR proteins to use
MAX_TAXA_FOR_TREE = 8

# MAFFT .bat 
MAFFT_BIN = r"C:\Program Files\mafft-win\mafft.bat" 

# Jukes-Cantor rate
jukes_rate = 1.0


# Load Dataformat with only AMR positive files, write the fastas

def build_temphd_fasta(max_taxa: int = MAX_TAXA_FOR_TREE) -> None:

    data_obj = DataFormat(str(DATA_ROOT), DB_NAME)

    # Not all the data - kept small for now
    data_obj.create_data_file(
        number_of_AMR_seqs=10,
        Number_of_control_seqs=None,
        high_quality_only=False,
        high_med_quality_only=True,
        save_fname=None,
        AMR_proteins_Only=True
    )

    df = data_obj.data_df

    # Require sequence + Protein_ID
    df = df.dropna(subset=["sequence", "Protein_ID"])
    df = df.drop_duplicates(subset=["Protein_ID"]).reset_index(drop=True)
    print(f"[DATA] Unique AMR Protein_ID rows: {len(df)}")

    # Take only first max_taxa proteins for the  ML tree
    df = df.iloc[:max_taxa].copy()
    print(f"[DATA] Subset to first {len(df)} Protein_IDs for tree.")

    n_written = 0
    with FASTA_PATH.open("w") as fout:
        for _, row in df.iterrows():
            prot_id = str(row["Protein_ID"])
            # drop stop codons if present
            seq = str(row["sequence"]).replace("*", "")  

            if not seq:
                continue

            fout.write(f">{prot_id}\n")
            for i in range(0, len(seq), 60):
                fout.write(seq[i:i + 60] + "\n")
            n_written += 1

    if n_written < 3:
        raise ValueError("Need at least 3 sequences to build a tree; got fewer.")

    print(f"[FASTA] Wrote {n_written} sequences to {FASTA_PATH}")

# Alignment using MAAFT 

def run_mafft_if_configured() -> Path:

    if MAFFT_BIN is None:
        print("[MAFFT] MAFFT_BIN is None; skipping alignment. "
              "Using unaligned FASTA directly (toy/demo only).")
        return FASTA_PATH

    if not os.path.exists(MAFFT_BIN):
        raise FileNotFoundError(
            f"[MAFFT] Could not find MAFFT .bat at:\n  {MAFFT_BIN}\n"
            "Update MAFFT_BIN or set MAFFT_BIN=None to skip alignment."
        )

    cmd_str = f'"{MAFFT_BIN}" --auto "{FASTA_PATH}"'
    print("[MAFFT] Running:", cmd_str)

    with ALIGN_PATH.open("w") as out_f:
        subprocess.run(cmd_str, check=True, shell=True, stdout=out_f)

    print(f"[MAFFT] Alignment written to: {ALIGN_PATH}")
    return ALIGN_PATH


def load_alignment_from_fasta(path: Path) -> Dict[str, str]:
    aln: Dict[str, str] = {}
    for record in SeqIO.parse(str(path), "fasta"):
        name = str(record.id)
        seq = str(record.seq).upper()
        aln[name] = seq
    print(f"[ALIGN] Loaded alignment with {len(aln)} taxa and "
          f"{len(next(iter(aln.values())))} sites from {path}")
    return aln


# Implementation of Felselsteins---------------------------------------

NUC_STATES = ["A", "C", "G", "T"]
STATE_INDEX = {b: i for i, b in enumerate(NUC_STATES)}


def jukes_transition_matrix(t: float, mu: float = jukes_rate) -> np.ndarray:

    lam = 4.0 * mu / 3.0
    e = np.exp(-lam * t)
    P_same = 0.25 + 0.75 * e
    P_diff = 0.25 - 0.25 * e
    P = np.full((4, 4), P_diff, dtype=float)
    np.fill_diagonal(P, P_same)
    return P


@dataclass(eq=False)
class Node:
    name: Optional[str] = None
    children: List["Node"] = field(default_factory=list)
    parent: Optional["Node"] = None
    branch_length: float = 0.0  # distance to parent

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def postorder(node: Node) -> List[Node]:
    order: List[Node] = []

    def _rec(n: Node):
        for c in n.children:
            _rec(c)
        order.append(n)

    _rec(node)
    return order


def felsenstein_log_likelihood(root: Node,
                               alignment: Dict[str, str],
                               mu: float = jukes_rate) -> float:
    # All sequences must be aligned and same length
    L_sites = len(next(iter(alignment.values())))
    for s in alignment.values():
        assert len(s) == L_sites, "All sequences must have same length (aligned)."

    # Precompute leaf partial likelihoods: (L_sites, 4)
    leaf_partials: Dict[str, np.ndarray] = {}
    for taxon, seq in alignment.items():
        arr = np.zeros((L_sites, 4), dtype=float)
        for i, base in enumerate(seq):
            if base == "-":
                # Unknown/gap: all states equally possible
                arr[i, :] = 1.0
            else:
                idx = STATE_INDEX.get(base.upper(), None)
                if idx is None:
                    arr[i, :] = 1.0
                else:
                    arr[i, idx] = 1.0
        leaf_partials[taxon] = arr

    # Stationary distribution for JC69
    pi = np.full(4, 0.25, dtype=float)

    partials: Dict[Node, np.ndarray] = {}

    # Bottom-up recursion
    for node in postorder(root):
        if node.is_leaf():
            if node.name not in leaf_partials:
                raise KeyError(f"No sequence for leaf {node.name}")
            partials[node] = leaf_partials[node.name]
        else:
            node_like = np.ones((L_sites, 4), dtype=float)
            for child in node.children:
                child_like = partials[child]
                P = jukes_transition_matrix(child.branch_length, mu)
                # child_like: (L_sites, 4), P.T: (4,4) -> (L_sites, 4)
                contrib = child_like @ P.T
                node_like *= contrib
            partials[node] = node_like

    # Root likelihood and total log-likelihood
    root_like = partials[root]
    site_likes = root_like @ pi  # (L_sites,)
    site_likes = np.clip(site_likes, 1e-300, 1.0)
    logL = float(np.sum(np.log(site_likes)))
    return logL


# Design tree topology
def p_distance(seq1: str, seq2: str) -> float:
    assert len(seq1) == len(seq2)
    diffs = 0
    valid = 0
    for a, b in zip(seq1, seq2):
        if a == "-" or b == "-":
            continue
        valid += 1
        if a != b:
            diffs += 1
    if valid == 0:
        return 0.0
    return diffs / valid


def upgma_tree(aln: Dict[str, str]) -> Node:
    taxa = list(aln.keys())
    clusters: Dict[int, Node] = {}
    heights: Dict[int, float] = {}
    sizes: Dict[int, int] = {}
    next_id = 0

    # Initialize each taxon as its own cluster
    for name in taxa:
        clusters[next_id] = Node(name=name)
        heights[next_id] = 0.0
        sizes[next_id] = 1
        next_id += 1

    # distance matrix: keys are (i,j) with i<j
    D: Dict[Tuple[int, int], float] = {}
    ids = list(range(next_id))
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = p_distance(aln[taxa[i]], aln[taxa[j]])
            D[(ids[i], ids[j])] = d

    # Main UPGMA loop
    while len(clusters) > 1:
        ids = list(clusters.keys())
        min_pair: Optional[Tuple[int, int]] = None
        min_d = float("inf")

        # Find closest pair of clusters
        for i_idx in range(len(ids)):
            for j_idx in range(i_idx + 1, len(ids)):
                i = ids[i_idx]
                j = ids[j_idx]
                key = (i, j) if i < j else (j, i)
                d = D[key]
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)

        i, j = min_pair
        # New cluster height
        h_u = min_d / 2.0

        # Create internal node
        u_node = Node(name=None, children=[clusters[i], clusters[j]])
        clusters[i].parent = u_node
        clusters[j].parent = u_node
        clusters[i].branch_length = h_u - heights[i]
        clusters[j].branch_length = h_u - heights[j]

        new_id = next_id
        next_id += 1
        clusters[new_id] = u_node
        heights[new_id] = h_u
        sizes[new_id] = sizes[i] + sizes[j]

        # Update distances: d(u, k) = weighted average of d(i,k), d(j,k)
        for k in list(clusters.keys()):
            if k in (i, j, new_id):
                continue
            key_ik = (i, k) if i < k else (k, i)
            key_jk = (j, k) if j < k else (k, j)
            d_ik = D[key_ik]
            d_jk = D[key_jk]
            d_uk = (sizes[i] * d_ik + sizes[j] * d_jk) / (sizes[i] + sizes[j])
            key_uk = (min(new_id, k), max(new_id, k))
            D[key_uk] = d_uk

        # Remove distances involving i or j
        D = { (a, b): val
              for (a, b), val in D.items()
              if a not in (i, j) and b not in (i, j) }

        # Drop old clusters
        del clusters[i], clusters[j]
        del heights[i], heights[j], sizes[i], sizes[j]

    # Remaining node is the root
    root_id = next(iter(clusters.keys()))
    root = clusters[root_id]
    root.branch_length = 0.0
    return root


# Print tree

def tree_to_newick(node: Node) -> str:
    if node.is_leaf():
        name = node.name if node.name is not None else "leaf"
        return f"{name}:{node.branch_length:.5f}"
    else:
        children_str = ",".join(tree_to_newick(c) for c in node.children)
        return f"({children_str}):{node.branch_length:.5f}"


# Main

def main():
    print("=== Felsenstein ML (JC69) on TemPhD AMR sequences ===")
    print(f"Data root : {DATA_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")

    # 1) Build FASTA from TemPhD
    build_temphd_fasta(max_taxa=MAX_TAXA_FOR_TREE)

    # 2) Align with MAFFT (or skip, depending on MAFFT_BIN)
    aln_path = run_mafft_if_configured()

    # 3) Load alignment
    alignment = load_alignment_from_fasta(aln_path)

    # 4) Build tree topology + branch lengths via UPGMA
    root = upgma_tree(alignment)

    # 5) Compute log-likelihood via Felsenstein pruning algorithm
    logL = felsenstein_log_likelihood(root, alignment, mu=jukes_rate)
    print(f"[ML] Log-likelihood under JC69 (Felsenstein): {logL:.3f}")
    
    # Save Newick to file for visualization
    tree_out = OUTPUT_DIR / "felsenstein_tree.nwk"
    with tree_out.open("w") as f:
        f.write(tree_to_newick(root) + ";\n")
    print(f"[TREE] Newick tree written to: {tree_out}")



if __name__ == "__main__":
    main()
