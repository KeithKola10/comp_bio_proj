# -*- coding: utf-8 -*-

"""
View a Newick tree from the TemPhD AMR project.

- Loads a .nwk file (Newick format)
- Prints basic summary
- Displays the tree using Biopython + matplotlib

Usage (in Spyder/IPython):
    %run "C:/Users/gabri/OneDrive/Desktop/CompBio Project/TemPhD/ViewAMRTree.py"
"""

from pathlib import Path

import matplotlib.pyplot as plt
from Bio import Phylo


# -------------------------------------------------------------------
# CONFIG: set path to your tree file here
# -------------------------------------------------------------------
BASE_PATH = Path(r"C:\Users\gabri\OneDrive\Desktop\CompBio Project\TemPhD")

# If you used the FastTree version earlier:
# TREE_FILE = "amr_genes_for_tree.ml_tree.nwk"

# If you're using the NJ version from the latest script:
# TREE_FILE = "amr_genes_for_tree.nj_tree.nwk"

TREE_FILE = "amr_genes_for_tree.nj_tree.nwk"   # <-- change to .nj_tree.nwk if needed
# -------------------------------------------------------------------


def main():
    tree_path = BASE_PATH / TREE_FILE

    if not tree_path.exists():
        raise FileNotFoundError(f"Tree file not found: {tree_path}")

    print(f"Loading tree from: {tree_path}")

    # Read Newick tree
    tree = Phylo.read(str(tree_path), "newick")

    # Basic info
    terminals = tree.get_terminals()
    print(f"\nTree summary:")
    print(f"  Number of leaves (tips): {len(terminals)}")
    print(f"  Rooted: {tree.rooted}")
    print(f"  First few leaf names:")
    for clade in terminals[:10]:
        print(f"   - {clade.name}")

    # Matplotlib figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    # Draw the tree (rectangular)
    Phylo.draw(
        tree,
        do_show=False,       # we'll call plt.show() explicitly
        axes=ax,
    )

    ax.set_title(TREE_FILE, fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
