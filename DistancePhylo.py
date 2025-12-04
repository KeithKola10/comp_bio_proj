# -*- coding: utf-8 -*-
"""
Build a phylogenetic tree for AMR genes from the TemPhD dataset.

Workflow:
1. Load or create the AMR dataset using DataFormat (amr_dataset.pickle).
2. Extract AMR gene/protein sequences and write them to a FASTA file.
3. Align sequences with MAFFT.
4. Build a Neighbor-Joining tree (distance-based) in pure Python using Biopython.
"""

import subprocess
from pathlib import Path

import pandas as pd
from DataFormat import DataFormat

from Bio import AlignIO, Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor


# ---------- PATHS & CONFIG ----------

BASE_PATH = r"C:\Users\gabri\OneDrive\Desktop\CompBio Project\TemPhD"
DB_NAME = "TemPhD"
PICKLE_NAME = "amr_dataset"          # created once by RunDataFormat.py

FASTA_NAME = "amr_genes_for_tree.fasta"
ALIGN_NAME = "amr_genes_for_tree.aln.fasta"
TREE_NAME  = "amr_genes_for_tree.nj_tree.nwk"

# IMPORTANT: point this to your MAFFT Windows install (mafft.bat)
MAFFT_EXE = r"C:\\Program Files\\mafft-win\\mafft.bat"   # <-- change if your path is different


# ---------- UTILS ----------

def run_and_log(cmd, cwd=None, stdout_file=None):
    """Run a command, optionally writing stdout to a file, and print progress."""
    print("\nRunning:", " ".join(cmd))

    if stdout_file is None:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(result.stdout)
    else:
        with open(stdout_file, "w") as fh:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                text=True,
                stdout=fh,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.stderr:
                print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {result.returncode}: "
            + " ".join(cmd)
        )


def load_or_build_dataset(base_path: str, db_name: str, pickle_name: str) -> pd.DataFrame:
    """
    Load amr_dataset.pickle if it exists; otherwise build it with create_data_file().
    Returns the dataframe and ensures DataFormat.data_df is set.
    """
    base = Path(base_path)
    data_obj = DataFormat(base_path, db_name)

    pkl_path = base / f"{pickle_name}.pickle"
    if pkl_path.exists():
        print(f"Loading existing dataset from {pkl_path}")
        df = data_obj.load_data_pickle(pickle_name)
        data_obj.data_df = df
    else:
        print(f"{pkl_path} not found. Building dataset with create_data_file()...")
        data_obj.create_data_file(
            number_of_AMR_seqs=10,           # adjust if you want more/less
            Number_of_control_seqs=None,
            high_quality_only=False,
            high_med_quality_only=True,
            save_fname=pickle_name,
            AMR_proteins_Only=True,
        )
        df = data_obj.data_df

    return df


def write_fasta_from_df(df: pd.DataFrame, fasta_path: Path,
                        id_col: str = "Protein_ID", seq_col: str = "sequence"):
    """
    Write sequences from dataframe to a FASTA file.

    FASTA headers look like:
        >ProteinID|Phage_ID|Best-hit ARO
    (Only fields that exist in df are used.)
    """
    cols = df.columns
    with open(fasta_path, "w") as fh:
        for _, row in df.dropna(subset=[seq_col, id_col]).iterrows():
            seq = str(row[seq_col]).replace(" ", "").replace("\n", "")
            if not seq:
                continue

            header_parts = [str(row[id_col])]

            if "Phage_ID" in cols:
                header_parts.append(str(row["Phage_ID"]))

            # Add an AMR annotation column if available
            for aro_col in ["Best-hit ARO", "ARO_Accession", "ARO_term"]:
                if aro_col in cols:
                    header_parts.append(str(row[aro_col]))
                    break

            header = "|".join(header_parts)

            fh.write(f">{header}\n")
            # wrap sequence lines at 60 characters
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")

    print(f"FASTA written to: {fasta_path}")


def build_alignment_and_tree(
    fasta_path: Path,
    align_path: Path,
    tree_path: Path,
    mafft_exec: str = MAFFT_EXE,
):
    """
    1) Multiple sequence alignment with MAFFT
    2) Neighbor-Joining tree using Biopython (no external FastTree needed)
    """
    # --- 1. MAFFT alignment ---
    mafft_cmd = [
        mafft_exec,
        "--auto",
        str(fasta_path),
    ]
    run_and_log(mafft_cmd, cwd=fasta_path.parent, stdout_file=align_path)

    # --- 2. NJ tree in Python ---
    print("\nBuilding Neighbor-Joining tree in Python (Biopython)...")

    # Read alignment produced by MAFFT
    alignment = AlignIO.read(str(align_path), "fasta")

    # For protein sequences, use BLOSUM62 distance; for nucleotide, "identity" also works.
    calculator = DistanceCalculator("identity")
    constructor = DistanceTreeConstructor(calculator, "nj")
    tree = constructor.build_tree(alignment)

    # Write tree to Newick file
    Phylo.write(tree, str(tree_path), "newick")

    print("\nDONE.")
    print(f"Alignment saved to: {align_path}")
    print(f"Tree saved to:      {tree_path}")


def main():
    base = Path(BASE_PATH)
    base.mkdir(parents=True, exist_ok=True)

    # 1) Load or create the AMR dataset
    df = load_or_build_dataset(BASE_PATH, DB_NAME, PICKLE_NAME)

    # 2) Deduplicate by Protein_ID to avoid repeated sequences in the tree
    if "Protein_ID" in df.columns:
        df = df.drop_duplicates(subset=["Protein_ID"]).reset_index(drop=True)

    # 3) Write FASTA from dataframe
    fasta_path = base / FASTA_NAME
    write_fasta_from_df(df, fasta_path, id_col="Protein_ID", seq_col="sequence")

    # 4) MAFFT alignment + NJ tree
    align_path = base / ALIGN_NAME
    tree_path = base / TREE_NAME
    build_alignment_and_tree(fasta_path, align_path, tree_path)


if __name__ == "__main__":
    main()
