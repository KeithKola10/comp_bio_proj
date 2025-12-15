

#%% imports 

from DataFormat import DataFormat
import GMM
import numpy as np
import pandas as pd
import ipdb
#%% load in data 


path = '/Users/kolak/comp_bio_proj/analysis_data' #path to data file
db_name = 'TemPhD' #database name

data_obj = DataFormat(path,db_name)

#%% load in data 
raw_data = data_obj.load_data_pickle('TemPhD_All')

#Filter for unique seqs
raw_data = raw_data.drop_duplicates(subset=['Protein Sequence','Host'], keep='first').reset_index(drop=True)
raw_data = raw_data[(raw_data['Host'] != 'human gut') & (raw_data['Host'] != 'bacterium')].reset_index(drop=True)
print(len(raw_data))
print(raw_data.keys())
print(len(raw_data['Host'].unique()))
# get releven data
cols_include = ['Aromaticity','Helix_fraction','Turn_fraction','Sheet_fraction' ,'Host'] #'protein_counts'
df_formatted, mapping = GMM.format_data_GMM(raw_data,cols_include)
print(mapping)

#drop host column
df_formatted_mod = df_formatted.drop(columns=['Host','labels'])
df_formatted_mod = df_formatted_mod.apply(pd.to_numeric, errors='coerce')


#run PCA + GMM Coluns
flag = True
clusters = len(mapping)
while flag is True:
    pc_scores, merged_labels, merged_probs_df, pca, gmm, merge_groups = GMM.pca_then_gmm(df_formatted_mod,n_pcs=2,n_clusters=clusters,scale=True,random_state=0,min_cluster_distance=.5) #.75 for protein_counts
    
    merge_groups_sorted = sorted(merge_groups, key=lambda s: min(s))
    old_to_new = {}
    for new_lab, grp in enumerate(merge_groups_sorted):
        for old_lab in grp:
            old_to_new[old_lab] = new_lab

    merged_labels = np.array([old_to_new[int(l)] for l in merged_labels], dtype=int)
    merge_groups = merge_groups_sorted 

    print('Started with %s, Ended with %s' % ( str(clusters), str(len(merge_groups))))
    if clusters != len(merge_groups):
        clusters = len(merge_groups)
    else:
        flag = False

print(merge_groups)
print('merged labels:', merged_labels)



#%% Evaluate GMM on Training Data

un_sup_met = GMM.evaluate_gmm_unsupervised(pc_scores.values, merged_labels, gmm)
print("Unsupervised Metrics:")
for key, value in un_sup_met.items():
    print(f"{key}: {value}")

#Group Relabel
true_labels = df_formatted['labels'].values.copy()

# Build mapping: each original label -> merged label
label_map = {}
for group in merge_groups:
    rep = list(group)[0]  # representative (first element)
    for lbl in group:
        label_map[lbl] = rep

# Apply mapping to true_labels
# --- Supervised metrics: TRUE species labels vs PREDICTED cluster labels ---
true_labels = df_formatted['labels'].values.astype(int)   # species IDs from mapping
pred_labels = np.asarray(merged_labels).astype(int)       # cluster IDs from GMM

soft_probs = merged_probs_df.values  # should align with rows of pc_scores

sup_met = GMM.evaluate_gmm_with_ground_truth(true_labels, pred_labels, soft_probs)
print("\nSupervised Metrics:")
for key, value in sup_met.items():
    print(f"{key}: {value}")

# %% Raw GMM labels
print(type(pc_scores.values))
GMM.plot_gmm_clusters_with_ellipses(pc_scores.values, gmm, labels=merged_labels, merge_groups=merge_groups, title="GMM Clusters with Group Labels", legend = False)
print(merge_groups)
print(mapping)
# %% True GMM Labels
GMM.plot_gmm_ellipses_with_true_labels(pc_scores.values, gmm, true_labels=df_formatted['Host'], merge_groups=merge_groups, title="GMM Clusters with True Species Labels", legend = False)


# %%
