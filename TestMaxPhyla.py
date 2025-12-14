# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path

from DataFormat import DataFormat


# Root folder with TemPhD data
DATA_ROOT = Path(r"C:\Users\gabri\OneDrive\Desktop\CompBio Project\TemPhD")
DB_NAME = "TemPhD"

# Output folder
OUTPUT_DIR = DATA_ROOT / "phylo_out"

# How many AMR phages to include
N_AMR_PHAGES = 10

# MAFFT: Windows .bat path 
MAFFT_BIN = r"C:\Program Files\mafft-win\mafft.bat"

# IQ-TREE3: full path to iqtree3.exe
IQTREE_BIN = r"C:\Program Files\iqtree-3.0.1-Windows\bin\iqtree3.exe"

# File paths
FASTA_PATH = OUTPUT_DIR / "amr_proteins_for_tree.fasta"
ALIGN_PATH = OUTPUT_DIR / "amr_proteins_for_tree.aln.fasta"
# IQ-TREE prefix; tree will be amr_proteins_for_tree.treefile
IQTREE_PREFIX = OUTPUT_DIR / "amr_proteins_for_tree"

# STEP 1: Build FASTA of AMR sequence
def prepare_fasta(n_amr_phages=N_AMR_PHAGES):
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    data_obj = DataFormat(str(DATA_ROOT), DB_NAME)
    data_obj.create_data_file(
        number_of_AMR_seqs=n_amr_phages,
        AMR_proteins_Only=True
    )

    df = data_obj.data_df
    df = df.dropna(subset=["sequence", "Protein_ID"])

    seen = set()
    with FASTA_PATH.open("w") as fout:
        for _, row in df.iterrows():
            prot_id = str(row["Protein_ID"])
            if prot_id in seen:
                continue
            seen.add(prot_id)

            seq = str(row["sequence"]).replace("*", "")
            phage_id = str(row.get("Phage_ID", "NA"))

            header = f">{prot_id}|{phage_id}"
            fout.write(header + "\n")

            # wrap sequence
            for i in range(0, len(seq), 60):
                fout.write(seq[i:i+60] + "\n")

    print(f"[FASTA] Wrote {len(seen)} unique AMR sequences to: {FASTA_PATH}")

# STEP 2: Multiple sequence alignment with MAFFT (.bat)
def run_mafft():
    if not FASTA_PATH.exists():
        raise FileNotFoundError(f"FASTA not found: {FASTA_PATH}")

    if not os.path.exists(MAFFT_BIN):
        raise FileNotFoundError(
            f"Could not find MAFFT .bat at:\n  {MAFFT_BIN}\n"
            "Update MAFFT_BIN in this script to the correct path."
        )

    cmd_str = f'"{MAFFT_BIN}" --auto "{FASTA_PATH}"'
    print("[MAFFT] Running:", cmd_str)

    with ALIGN_PATH.open("w") as out_f:
        subprocess.run(cmd_str, check=True, shell=True, stdout=out_f)

    print(f"[MAFFT] Alignment written to: {ALIGN_PATH}")


# -------------------------------------------------------------------------
# STEP 3: Felsenstein ML tree with IQ-TREE3 (GTR+G nucleotide model)
def run_iqtree():
    if not ALIGN_PATH.exists():
        raise FileNotFoundError(f"Alignment not found: {ALIGN_PATH}")

    if not os.path.exists(IQTREE_BIN):
        raise FileNotFoundError(
            f"Could not find IQ-TREE executable at:\n  {IQTREE_BIN}\n"
            "Check that iqtree3.exe is installed there or update IQTREE_BIN."
        )

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    prefix_str = str(IQTREE_PREFIX)
    cmd = [
        IQTREE_BIN,
        "-s", str(ALIGN_PATH),   # alignment (FASTA, nucleotide)
        "-st", "DNA",            # explicitly tell IQ-TREE it's DNA
        "-m", "GTR+G",           # nucleotide ML model
        "-bb", "1000",           # ultrafast bootstrap replicates (>= 1000)
        "-nt", "AUTO",           # auto-detect cores
        "-pre", prefix_str       # outputs: amr_proteins_for_tree.*
    ]

    print("[IQ-TREE] Running:", " ".join(cmd))

    # Capture output so you can see IQ-TREE's own messages
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True
    )

    if result.stdout:
        print("===== IQ-TREE stdout =====")
        print(result.stdout)
    if result.stderr:
        print("===== IQ-TREE stderr =====")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"IQ-TREE failed with exit code {result.returncode}. "
            "Check the IQ-TREE stderr output above for details."
        )

    treefile = IQTREE_PREFIX.with_suffix(".treefile")
    if treefile.exists():
        print(f"[IQ-TREE] ML tree (Felsenstein) written to: {treefile}")
    else:
        print("[IQ-TREE] Finished, but .treefile not found automatically.")
        print(f"          Look in {OUTPUT_DIR} for *.treefile outputs.")


# MAIN
def main():
    print("=== Building ML phylogenetic tree (Felsenstein, nucleotide GTR+G) from AMR sequences ===")
    prepare_fasta()
    run_mafft()
    run_iqtree()
    print("\nDone.")
    print("Visualize amr_proteins_for_tree.treefile with your Visualize Tree script, iTOL, or FigTree.")


if __name__ == "__main__":
    main()
