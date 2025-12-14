# Visualize_Felsenstein_Tree.py

from pathlib import Path

import matplotlib.pyplot as plt
from Bio import Phylo

from DataFormat import DataFormat

# Set up paths
DATA_ROOT = Path(r"C:\Users\gabri\OneDrive\Desktop\CompBio Project\TemPhD")
DB_NAME = "TemPhD"

# Newick tree file produced by from Felsenstein
TREE_PATH = DATA_ROOT / "phylo_out_felsenstein" / "felsenstein_tree.nwk"

# These should match the parameters in the Felsenstein script
N_AMR_PHAGES = 10
AMR_ONLY = True

# Load meta data in for labels on tree diagram
def load_metadata() -> dict:
    print("[META] Rebuilding DataFormat dataset for metadata...")
    data_obj = DataFormat(str(DATA_ROOT), DB_NAME)

    data_obj.create_data_file(
        number_of_AMR_seqs=N_AMR_PHAGES,
        Number_of_control_seqs=None,
        high_quality_only=False,
        high_med_quality_only=True,
        save_fname=None,
        AMR_proteins_Only=AMR_ONLY,
    )

    df = data_obj.data_df

    # Only need Protein_ID + Host
    needed_cols = [c for c in ["Protein_ID", "Host"] if c in df.columns]
    if "Host" not in needed_cols:
        raise KeyError("Column 'Host' not found in DataFormat.data_df. "
                       "Check that temphd_phage_meta_data.tsv is present.")

    df_meta = (
        df[needed_cols]
        .dropna(subset=["Protein_ID", "Host"])
        .drop_duplicates(subset=["Protein_ID"])
    )

    label_map = dict(zip(df_meta["Protein_ID"], df_meta["Host"]))
    print(f"[META] Built label map for {len(label_map)} Protein_IDs.")
    return label_map


# Relabel tree leaves so that readable with host/protein
def relabel_tree_with_host(tree, label_map: dict):
    missing = 0
    for clade in tree.get_terminals():
        prot_id = clade.name
        if prot_id in label_map:
            host = label_map[prot_id]
            clade.name = f"{host} | {prot_id}"
        else:
            missing += 1
    if missing > 0:
        print(f"[WARN] Host label missing for {missing} tips (kept original names).")
    return tree


# Main
def main():
    print("Visualizing")
    print(f"Tree file : {TREE_PATH}")

    # 1) Load tree
    tree = Phylo.read(str(TREE_PATH), "newick")
    print(f"[TREE] Loaded tree with {len(tree.get_terminals())} tips.")

    # 2) Build label map and relabel tips
    label_map = load_metadata()
    relabel_tree_with_host(tree, label_map)

    # 3) Plot tree
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    Phylo.draw(tree, do_show=False, axes=ax)
    plt.tight_layout()
    plt.show()

    # Also print ASCII version to the console
    print("\n[TREE ASCII]")
    Phylo.draw_ascii(tree)


if __name__ == "__main__":
    main()
