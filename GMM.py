import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score,silhouette_score, calinski_harabasz_score, davies_bouldin_score,adjusted_rand_score
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

#Amino Acid dict
CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W'}


amino_acids = sorted(set(CODON_TABLE.values()) - {"*"})


#Colors
# 50 reasonably distinct colors (hex)
COLOR50 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#6b6ecf", "#9c9ede", "#3182bd", "#6baed6",
    "#9ecae1", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2",
    "#31a354", "#74c476", "#a1d99b", "#c7e9c0", "#756bb1",
    "#9e9ac8", "#bcbddc", "#dadaeb", "#636363", "#969696",
    "#bdbdbd", "#d9d9d9", "#e7ba52", "#e7cb94", "#ad494a",
    "#d6616b", "#e7969c", "#ce6dbd", "#de9ed6", "#3182bd",
    "#6baed6", "#9ecae1", "#c6dbef", "#9e9ac8"
]

#Data formattting functions
def label_generator(names):
    mapping = {}
    counter = 0
    result = []

    for n in names:
        if n not in mapping:
            mapping[n] = counter
            counter += 1
        result.append(mapping[n])
        
    return result, mapping 

def translate_dna(dna_seq):
    dna_seq = dna_seq.upper().replace("U", "T")
    protein = []

    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i:i+3]
        aa = CODON_TABLE.get(codon, 'X')
        protein.append(aa)

    return "".join(protein)
    


def count_amino_acids_list(seq_list):
    rows = []
    
    for seq in seq_list:
        protein = translate_dna(seq)

        # initialize AA count row
        counts = {aa: 0 for aa in amino_acids}
        for aa in protein:
            if aa in counts:
                counts[aa] += 1
        
        rows.append(counts)
    return pd.DataFrame(rows)

def format_data_GMM(data,cols_include):
    
    if 'protein_counts' in cols_include:
        seqs = data['Protein Sequence'].tolist()
        df_amino = count_amino_acids_list(seqs)
        cols_include.remove('protein_counts')
        df_rest = pd.DataFrame(data, columns=cols_include)
        df_out = pd.concat([df_rest, df_amino], axis=1)
    else:
        df_out = pd.DataFrame(data, columns=cols_include)
        
    df_out['labels'], mapping = label_generator(data['Host'])
        
    return df_out, mapping

#
def pca_then_gmm(df,n_pcs=2,n_clusters=3,scale=True,random_state=0,min_cluster_distance=None):


    #Numeric data only
    df_num = df.select_dtypes(include="number")
    X = df_num.values
    
    #Data scaling
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        scaler = None
    #PCA
    pca = PCA(n_components=n_pcs, random_state=random_state)
    X_pca = pca.fit_transform(X)

    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
    pc_scores = pd.DataFrame(X_pca, columns=pc_cols, index=df.index)

    #GMM on PCA scores
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state
    )
    gmm.fit(X_pca)

    # Original labels / probabilities
    orig_labels = gmm.predict(X_pca)
    orig_labels_series = pd.Series(orig_labels, index=df.index, name="cluster")

    orig_probs = gmm.predict_proba(X_pca)
    probs_df = pd.DataFrame(
        orig_probs,
        index=df.index,
        columns=[f"cluster_{k}" for k in range(n_clusters)]
    )

    # merging based on min_cluster_distance

    if min_cluster_distance is None or n_clusters <= 1:
        merge_groups = [set([k]) for k in range(n_clusters)]
        return pc_scores, orig_labels_series, probs_df, pca, gmm, merge_groups

    means_2d = gmm.means_[:, :n_pcs]
    n_comp = gmm.n_components
    adjacency = {k: set() for k in range(n_comp)}
    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            dist_ij = np.linalg.norm(means_2d[i] - means_2d[j])
            if dist_ij < min_cluster_distance:
                adjacency[i].add(j)
                adjacency[j].add(i)
    visited = set()
    merge_groups = []

    for k in range(n_comp):
        if k in visited:
            continue
        stack = [k]
        group = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            group.add(node)
            for neigh in adjacency[node]:
                if neigh not in visited:
                    stack.append(neigh)
        merge_groups.append(group)

    if len(merge_groups) == n_comp and all(len(g) == 1 for g in merge_groups):
        return pc_scores, orig_labels_series, probs_df, pca, gmm, merge_groups
    comp_to_merged = {}
    for new_idx, group in enumerate(merge_groups):
        for comp_idx in group:
            comp_to_merged[comp_idx] = new_idx

    #Remap labels
    merged_label_values = np.vectorize(comp_to_merged.get)(orig_labels)
    merged_labels = pd.Series(
        merged_label_values,
        index=df.index,
        name="cluster"
    )

    # 8. Merge probability columns by summing within groups
    n_merged = len(merge_groups)
    merged_probs = np.zeros((orig_probs.shape[0], n_merged))

    for new_idx, group in enumerate(merge_groups):
        group = list(group)
        merged_probs[:, new_idx] = orig_probs[:, group].sum(axis=1)

    merged_probs_df = pd.DataFrame(
        merged_probs,
        index=df.index,
        columns=[f"cluster_{k}" for k in range(n_merged)]
    )

    return pc_scores, merged_labels, merged_probs_df, pca, gmm, merge_groups


def evaluate_gmm_unsupervised(X, gmm_labels, gmm):
    metrics = {
        "log_likelihood_per_sample": gmm.score(X),
        "AIC": gmm.aic(X),
        "BIC": gmm.bic(X),
        "silhouette": silhouette_score(X, gmm_labels),
        "calinski_harabasz": calinski_harabasz_score(X, gmm_labels),
        "davies_bouldin": davies_bouldin_score(X, gmm_labels),
    }
    return metrics

def probabilistic_rand_index(true_labels, prob_matrix):
    true_labels = np.asarray(true_labels)
    prob_matrix = np.asarray(prob_matrix)
    n = len(true_labels)

    same_true = np.equal.outer(true_labels, true_labels).astype(float)
    same_cluster_prob = prob_matrix @ prob_matrix.T

    # PRI = average agreement probability
    pri = np.mean(same_true * same_cluster_prob +
                  (1 - same_true) * (1 - same_cluster_prob))
    return pri


def evaluate_gmm_with_ground_truth(true_labels, gmm_labels, gmm_probabilities):
 
    metrics = {
        "ARI": adjusted_rand_score(true_labels, gmm_labels),
        "AMI": adjusted_mutual_info_score(true_labels, gmm_labels),
        "homogeneity": homogeneity_score(true_labels, gmm_labels),
        "completeness": completeness_score(true_labels, gmm_labels),
        "v_measure": v_measure_score(true_labels, gmm_labels),
        "PRI": probabilistic_rand_index(true_labels, gmm_probabilities)}
    return metrics



def plot_gmm_clusters_with_ellipses(X,gmm,labels=None,merge_groups=None,title="GMM Clusters with Ellipses (PC1 vs PC2)",
                                    n_std=2.0,alpha_points=0.8,alpha_ellipse=0.9,legend=True,):



    X = np.asarray(X)
    x0 = X[:, 0]
    x1 = X[:, 1]
    x_label = "PC1"
    y_label = "PC2"

    n_components = gmm.n_components
    if merge_groups is None:
        merge_groups = [set([k]) for k in range(n_components)]

    if labels is None:
        comp_labels = gmm.predict(X)
        comp_to_merged = {}
        for new_idx, group in enumerate(merge_groups):
            for comp_idx in group:
                comp_to_merged[comp_idx] = new_idx
        labels = np.vectorize(comp_to_merged.get)(comp_labels)

    labels = np.asarray(labels)
    n_groups = len(merge_groups)

    cluster_colors = [COLOR50[i % len(COLOR50)] for i in range(n_groups)]
    point_colors = [cluster_colors[int(l)] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x0,x1,c=point_colors,alpha=alpha_points,edgecolor="k",linewidth=0.5,)

    cov_type = gmm.covariance_type
    means_2d = gmm.means_[:, :2]

    def get_cov_2d_for_component(k):
        if cov_type == "full":
            return gmm.covariances_[k][:2, :2]
        elif cov_type == "tied":
            return gmm.covariances_[:2, :2]
        elif cov_type == "diag":
            diag_vals = gmm.covariances_[k][:2]
            return np.diag(diag_vals)
        elif cov_type == "spherical":
            return np.eye(2) * gmm.covariances_[k]
        else:
            raise ValueError(f"Unsupported covariance_type: {cov_type}")

    #  Draw ellipses for each merged group 
    for g_idx, group in enumerate(merge_groups):
        group = list(group)

        if len(group) == 1:
            k = group[0]
            mean = means_2d[k]
            cov = get_cov_2d_for_component(k)
        else:
            weights = gmm.weights_[group]
            weights = weights / weights.sum()

            comp_means = means_2d[group] 
            comp_covs = np.array([get_cov_2d_for_component(k) for k in group])  

            mean = np.sum(weights[:, None] * comp_means, axis=0)

            cov = np.zeros((2, 2))
            for w, mu_j, cov_j in zip(weights, comp_means, comp_covs):
                diff = (mu_j - mean).reshape(2, 1)
                cov += w * (cov_j + diff @ diff.T)

        # Eigen-decomposition to get ellipse axes
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Angle of ellipse
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        color = cluster_colors[g_idx]

        ell = Ellipse(xy=mean,width=width,height=height,angle=angle,edgecolor=color,facecolor="none",linewidth=2,alpha=alpha_ellipse)
        ax.add_patch(ell)

        ax.scatter(mean[0],mean[1],marker="x",color=color,s=80,linewidth=2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    legend_handles = []
    legend_labels = []

    for g_idx, group in enumerate(merge_groups):
        color = cluster_colors[g_idx]

        handle = Line2D([0], [0],marker='o',linestyle='',markerfacecolor=color,markeredgecolor='k', alpha=alpha_points, )
        legend_handles.append(handle)

        comps_str = ",".join(str(c) for c in sorted(group))
        legend_labels.append(f"{g_idx}: [{comps_str}]")

    if legend:
        ax.legend(legend_handles,legend_labels,title="Merged clusters\n(new: [orig comps])",loc="center left",bbox_to_anchor=(1.02, 0.5),borderaxespad=0.0,)

    ax.grid(True)
    plt.tight_layout()
    plt.show()


def plot_gmm_ellipses_with_true_labels(X,gmm,true_labels,merge_groups=None,title="GMM Ellipses with True-Label Coloring (PC1 vs PC2)",n_std=2.0, alpha_points=0.8,alpha_ellipse=0.9,legend=True,):


    X = np.asarray(X)
    x0 = X[:, 0]
    x1 = X[:, 1]
    x_label = "PC1"
    y_label = "PC2"

    true_labels = np.asarray(true_labels)
    n_components = gmm.n_components


    if merge_groups is None:
        merge_groups = [set([k]) for k in range(n_components)]

    if not isinstance(merge_groups, (list, tuple)):
        raise ValueError("merge_groups must be a list/tuple of sets, "
                         "as returned by pca_then_gmm.")

    n_groups = len(merge_groups)
    unique_true = np.unique(true_labels)
    true_colors = {lab: COLOR50[i % len(COLOR50)] for i, lab in enumerate(unique_true)}
    point_colors = [true_colors[lab] for lab in true_labels]

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(x0,x1, c=point_colors,alpha=alpha_points,edgecolor="k",linewidth=0.5)
    cov_type = gmm.covariance_type
    means_2d = gmm.means_[:, :2]

    cluster_colors = [COLOR50[i % len(COLOR50)] for i in range(n_groups)]

    def get_cov_2d_for_component(k):
        """Get 2x2 covariance for component k in first two dims."""
        if cov_type == "full":
            return gmm.covariances_[k][:2, :2]
        elif cov_type == "tied":
            return gmm.covariances_[:2, :2]
        elif cov_type == "diag":
            diag_vals = gmm.covariances_[k][:2]
            return np.diag(diag_vals)
        elif cov_type == "spherical":
            return np.eye(2) * gmm.covariances_[k]
        else:
            raise ValueError(f"Unsupported covariance_type: {cov_type}")

    for g_idx, group in enumerate(merge_groups):
        group = list(group)

        if len(group) == 1:
            k = group[0]
            mean = means_2d[k]
            cov = get_cov_2d_for_component(k)
        else:
            weights = gmm.weights_[group]
            weights = weights / weights.sum()

            comp_means = means_2d[group]  
            comp_covs = np.array([get_cov_2d_for_component(k) for k in group]) 

            mean = np.sum(weights[:, None] * comp_means, axis=0)
            cov = np.zeros((2, 2))
            for w, mu_j, cov_j in zip(weights, comp_means, comp_covs):
                diff = (mu_j - mean).reshape(2, 1)
                cov += w * (cov_j + diff @ diff.T)

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigvals)

        color = cluster_colors[g_idx]

        ell = Ellipse(xy=mean,width=width,height=height,angle=angle,edgecolor=color,facecolor="none",linewidth=2,alpha=alpha_ellipse)
        ax.add_patch(ell)
        ax.scatter(mean[0],mean[1],marker="x",color=color,s=80,linewidth=2,)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    point_handles = []
    point_labels = []
    for lab in unique_true:
        color = true_colors[lab]
        handle = Line2D([0], [0],marker='o',linestyle='',markerfacecolor=color, markeredgecolor='k',alpha=alpha_points)
        point_handles.append(handle)
        point_labels.append(str(lab))

    cluster_handles = []
    cluster_labels = []
    for g_idx, group in enumerate(merge_groups):
        color = cluster_colors[g_idx]
        handle = Line2D(
            [0], [0],
            linestyle='-',
            color=color,
            linewidth=2,
        )
        comps_str = ",".join(str(c) for c in sorted(group))
        cluster_handles.append(handle)
        cluster_labels.append(f"{g_idx}: [{comps_str}]")

    first_legend = ax.legend(
        point_handles,
        point_labels,
        title="True labels",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    ax.add_artist(first_legend)

    if legend:
        ax.legend(cluster_handles,cluster_labels,title="GMM clusters\n(new: [orig comps])",loc="lower left", bbox_to_anchor=(1.02, 0.0),borderaxespad=0.0)

    ax.grid(True)
    plt.tight_layout()
    plt.show()





def plot_pc_loading_bar(pca, feature_names, pc_index=0, top_n=None,
                        figsize=(10,5), color="steelblue", title=None):
    """
    Plot bar chart of loading magnitudes for a given principal component.

    Parameters
    ----------
    pca : fitted sklearn.decomposition.PCA object
        Must contain pca.components_
    feature_names : list-like
        Names of original variables / features
    pc_index : int
        Zero-based index of principal component to plot (0=PC1, 1=PC2, ...)
    top_n : int or None
        If given, only show top N strongest contributing variables
    figsize : tuple
        Figure size (width, height)
    color : str or matplotlib color
        Bar color
    title : str or None
        Custom title (otherwise generated automatically)

    Returns
    -------
    pandas.Series: sorted PC loadings (abs value sorted)
    """
    
    # Extract loadings
    loadings = pca.components_[pc_index]

    # Store signed values (for interpretation)
    signed_series = pd.Series(loadings, index=feature_names)

    # Sort by absolute magnitude
    sorted_series = signed_series.reindex(signed_series.abs().sort_values(ascending=False).index)

    # Apply top_n if specified
    if top_n is not None:
        sorted_series = sorted_series.iloc[:top_n]

    # Plot bar chart
    plt.figure(figsize=figsize)
    sorted_series.abs().plot(kind="bar", color=color, edgecolor="k")

    if title is None:
        title = f"PC{pc_index+1} variable contributions"

    plt.title(title)
    plt.ylabel("Absolute loading magnitude")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return sorted_series

