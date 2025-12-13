import numpy as np
import pandas as pd
from collections import Counter
import itertools
import warnings
import matplotlib.pyplot as plt
import random
import re
import os

# Sklearn & HMM Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from hmmlearn import hmm

warnings.filterwarnings('ignore')

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def reverse_complement(seq):
    if seq is None:
        return None
    seq = str(seq)
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]

def get_kmers(sequence, k=4):
    if not sequence:
        return {}
    sequence = sequence.upper().replace('N', '')
    if len(sequence) < k:
        return {}
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
    counts = Counter([sequence[i:i+k] for i in range(len(sequence)-k+1)])
    total = sum(counts.values())
    if total == 0:
        return {kmer: 0 for kmer in all_kmers}
    return {kmer: counts.get(kmer, 0)/total for kmer in all_kmers}

def get_gc(sequence):
    if not sequence:
        return 0.0
    s = sequence.upper()
    return (s.count('G') + s.count('C')) / len(s)

def is_mobile_element_annotation(annot_str, keywords=None):
    """
    Conservative mobile-element annotation checker using word-boundary regex.
    """
    if not isinstance(annot_str, str) or annot_str.strip() == "":
        return False
    if keywords is None:
        keywords = [
            'integrase', 'transposase', 'recombinase', 'resolvase',
            'insertion sequence', 'insertion-sequence', 'transposon',
            'integron', 'cassette'
        ]
    s = annot_str.lower()
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw.lower()) + r'\b', s):
            return True
    return False

def visualize_decoding(phage_id, windows, y_true, y_pred_viterbi, y_prob_posterior):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.fill_between(windows, 0, y_true, color='green', alpha=0.3, step='mid', label='Ground Truth (Actual ARG / Island)')
    ax1.step(windows, y_pred_viterbi, where='mid', color='red', linestyle='--', linewidth=2, label='Viterbi Prediction')
    ax1.set_ylabel("State (0/1)")
    ax1.set_title(f"Phage: {phage_id} - Viterbi Decoding (Best Path)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(windows, 0, y_true, color='green', alpha=0.3, label='Ground Truth')
    ax2.plot(windows, y_prob_posterior, linewidth=1.5, label='Posterior Probability')
    ax2.axhline(y=0.5, color='gray', linestyle=':', label='Decision Threshold')
    ax2.set_ylabel("Prob(Island)")
    ax2.set_xlabel("Genome Position (bp)")
    ax2.set_title("Posterior Decoding (Classification Uncertainty)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def merge_windows_to_ranges(positions, window_size, step):
    """Merge contiguous/touching window start positions into ranges (start, end)."""
    if len(positions) == 0:
        return []
    pos = np.sort(np.array(positions))
    ranges = []
    start = pos[0]
    prev = pos[0]
    for p in pos[1:]:
        if p <= prev + step:
            prev = p
            continue
        else:
            ranges.append((int(start), int(prev + window_size)))
            start = p
            prev = p
    ranges.append((int(start), int(prev + window_size)))
    return ranges

# ==========================================
# 2. MAIN PROGRAM (strict island definition + HMM)
# ==========================================
def run_pipeline(
    folder_path, db_name="TemPhD",
    window_size=1000, step=250, k=4,
    island_radius=3000,            # +/- bp to group ARG-near windows
    min_args_for_island=1,         # require >= this many ARGs in candidate region (we keep default 1)
    min_mobile_count=2,            # require >= this many mobile annotations in region OR gc shift
    gc_shift_threshold=0.1,       # absolute GC difference threshold
    min_island_length_bp=2000,     # minimum merged island length
    mobile_kw_list=None,
    verbose=True
):
    """
    Detect ARG-enriched genomic islands: merged contiguous regions with >=1 ARG AND (mobile_count >= min_mobile_count OR GC shift >= gc_shift_threshold).
    Returns dictionary of results (no CSV writes).
    """
    if mobile_kw_list is None:
        mobile_kw_list = [
            'integrase', 'transposase', 'recombinase', 'resolvase',
            'insertion sequence', 'insertion-sequence', 'transposon',
            'integron', 'cassette'
        ]

    if verbose:
        print("--- STARTING STRICT ARG-ENRICHED ISLAND DETECTION + HMM PIPELINE (strand-aware) ---")

    if 'DataFormat' not in globals():
        print("CRITICAL ERROR: 'DataFormat' class not found. Please run Part 1 code first.")
        return

    data = DataFormat(folder_path, db_name)

    try:
        amr_df = data.__load_tsv__(data.amr_tsv)
        meta_df = data.__load_tsv__(data.meta_tsv)
        prot_df = data.__load_tsv__(data.protein_tsv)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # normalize strand
    if 'Strand' in prot_df.columns:
        prot_df['Strand'] = prot_df['Strand'].astype(str).str.strip()
    elif 'strand' in prot_df.columns:
        prot_df['Strand'] = prot_df['strand'].astype(str).str.strip()
    else:
        prot_df['Strand'] = '+'
        if verbose:
            print("WARNING: No 'Strand' column found — assuming '+' for all proteins.")

    # prepare phage lists
    hq_phages = meta_df[meta_df['Completeness'].isin(['High-quality', 'Complete'])]['Phage_ID'].unique()
    arg_phages = amr_df['Phage_id'].unique()
    valid_phages = list(set(hq_phages) & set(arg_phages))

    if len(valid_phages) == 0:
        print("No valid phages found after filtering completeness & ARG presence.")
        return

    train_phages, test_phages = train_test_split(valid_phages, test_size=0.20, random_state=42)

    if verbose:
        print(f"\n[CHECKPOINT 1] DATASET PREPARATION")
        print(f"Total Valid Phages: {len(valid_phages)}")
        print(f"Training Set: {len(train_phages)} | Test Set: {len(test_phages)}")
        print("--------------------------------------------------")

    # collect global features + island summaries
    features_list = []
    island_summary = []

    if verbose:
        print(f"Extracting per-window features from {len(valid_phages)} phages...")

    for idx, phage_id in enumerate(valid_phages):
        df_seq = data.__load_fasta__(phage_id)
        if df_seq is None or df_seq.empty:
            continue
        seq = df_seq.iloc[0]['sequence']
        seq_len = len(seq)
        genome_gc = get_gc(seq)

        my_args = amr_df[amr_df['Phage_id'] == phage_id].copy()
        my_prots = prot_df[prot_df['Phage_ID'] == phage_id].copy()

        # arg protein ids
        if 'Protein_id' in my_args.columns:
            arg_prot_ids = set(my_args['Protein_id'].astype(str))
        else:
            arg_prot_ids = set()

        # build coordinates for ARG-associated proteins (if present)
        arg_coords = []
        for _, r in my_prots[my_prots['Protein_ID'].astype(str).isin(arg_prot_ids)].iterrows():
            try:
                s0 = max(0, int(r['Start']) - 1)
                e0 = min(seq_len, int(r['Stop']))
                arg_coords.append((s0, e0))
            except:
                continue
        arg_midpoints = [(s+e)//2 for (s,e) in arg_coords]

        # compute mobile positions from annotations
        mobile_positions = []
        for _, prow in my_prots.iterrows():
            annot_fields = []
            for col in ['Product','Annotation','Function','Description','product','annotation','function','description','Name']:
                if col in my_prots.columns and pd.notna(prow.get(col, None)):
                    annot_fields.append(str(prow.get(col,'')))
            annot_concat = " ".join(annot_fields).strip()
            if is_mobile_element_annotation(annot_concat, mobile_kw_list):
                try:
                    ps0 = max(0, int(prow['Start']) - 1)
                    pe0 = min(seq_len, int(prow['Stop']))
                    mobile_positions.append((ps0, pe0))
                except:
                    continue

        # per-phage per-window temporary store
        phage_rows = []

        # sliding windows: collect per-window stats but do NOT finalize label yet
        for i in range(0, max(1, seq_len - window_size + 1), step):
            w_seq = seq[i:i+window_size]
            w_start, w_end = i, min(i + window_size, seq_len)

            # ARG overlaps in window (count)
            arg_overlap_count = 0
            for (a0,a1) in arg_coords:
                if a0 < w_end and a1 > w_start:
                    arg_overlap_count += 1

            # mobile overlaps in window (count)
            mobile_overlap_count = 0
            for (m0,m1) in mobile_positions:
                if m0 < w_end and m1 > w_start:
                    mobile_overlap_count += 1

            # near_any_arg: center within island_radius of any ARG midpoint
            center = (w_start + w_end)//2
            near_any_arg = any(abs(mp - center) <= island_radius for mp in arg_midpoints) if len(arg_midpoints)>0 else False

            row = {
                'Phage_ID': phage_id,
                'Window_Start': w_start,
                'Window_End': w_end,
                'Local_GC': get_gc(w_seq),
                'kmer': None,  # placeholder, we'll expand k-mers later if needed
                'arg_overlap_count': arg_overlap_count,
                'mobile_overlap_count': mobile_overlap_count,
                'near_any_arg': int(near_any_arg),
                # label to be set after merged-range evaluation
                'Label': 0
            }
            phage_rows.append(row)

        # convert to df for this phage and find candidate windows near any ARG
        phage_df = pd.DataFrame(phage_rows)
        candidate_positions = phage_df[phage_df['near_any_arg'] == 1]['Window_Start'].values
        merged_candidate_ranges = merge_windows_to_ranges(candidate_positions, window_size=window_size, step=step)

        # Evaluate each merged candidate range and accept only those that satisfy final island criteria:
        #   (1) region contains >= min_args_for_island ARGs
        #   (2) region length >= min_island_length_bp
        #   (3) and (mobile_count >= min_mobile_count OR GC shift >= gc_shift_threshold)
        accepted_ranges = []
        for (rs, re) in merged_candidate_ranges:
            # count ARGs whose midpoints fall in [rs, re)
            arg_count_in_region = sum(1 for mp in arg_midpoints if (mp >= rs and mp < re))
            # count mobile hits overlapping region
            mobile_count_in_region = sum(1 for (m0,m1) in mobile_positions if not (m1 <= rs or m0 >= re))
            # compute GC shift for the full region (extract sequence safely)
            try:
                region_seq = seq[rs:re]
                region_gc = get_gc(region_seq)
            except:
                region_gc = genome_gc
            gc_shift = abs(region_gc - genome_gc)
            region_len = re - rs

            # acceptance criteria
            meets_arg = arg_count_in_region >= max(1, min_args_for_island)  # ensure at least 1
            meets_mobile_or_gc = (mobile_count_in_region >= min_mobile_count) or (gc_shift >= gc_shift_threshold)
            meets_length = region_len >= min_island_length_bp

            if meets_arg and meets_mobile_or_gc and meets_length:
                accepted_ranges.append({
                    'start': rs,
                    'end': re,
                    'length': region_len,
                    'arg_count': arg_count_in_region,
                    'mobile_count': mobile_count_in_region,
                    'gc_shift': gc_shift
                })

        # set labels for windows that fall inside accepted ranges
        if len(accepted_ranges) > 0:
            for ar in accepted_ranges:
                mask = (phage_df['Window_Start'] >= ar['start']) & (phage_df['Window_End'] <= ar['end'])
                phage_df.loc[mask, 'Label'] = 1

        # record island summary info
        island_summary.append({
            'Phage_ID': phage_id,
            'num_accepted_islands': len(accepted_ranges),
            'accepted_islands': accepted_ranges,
            'num_args': len(arg_midpoints),
            'num_mobile_positions': len(mobile_positions),
            'genome_gc': genome_gc
        })

        # expand k-mers and other features now and append to global list
        for _, prow in phage_df.iterrows():
            wstart = int(prow['Window_Start'])
            wend = int(prow['Window_End'])
            wseq = seq[wstart:wend]
            row_features = {
                'Phage_ID': phage_id,
                'Window_Start': wstart,
                'Label': int(prow['Label']),
                'Local_GC': prow['Local_GC'],
                'Arg_Count_Window': prow['arg_overlap_count'],
                'Mobile_Count_Window': prow['mobile_overlap_count'],
                'Protein_Density': 0  # we'll fill protein density below
            }
            # add kmer features (k up to k param)
            kmer_feats = get_kmers(wseq, k=k)
            row_features.update(kmer_feats)

            # compute overlapped proteins to approximate protein density and protein stats
            overlaps = []
            for _, p in my_prots.iterrows():
                try:
                    ps0 = max(0, int(p['Start']) - 1)
                    pe0 = min(seq_len, int(p['Stop']))
                except:
                    continue
                if ps0 < wend and pe0 > wstart:
                    overlaps.append(p)
            overlapped_prots = pd.DataFrame(overlaps) if len(overlaps) else pd.DataFrame(columns=my_prots.columns)
            orf_seqs = []
            for _, op in overlapped_prots.iterrows():
                try:
                    ps0 = max(0, int(op['Start']) - 1)
                    pe0 = min(seq_len, int(op['Stop']))
                except:
                    continue
                subseq = seq[ps0:pe0]
                if str(op.get('Strand', '+')).strip() in ['-', '-1']:
                    subseq = reverse_complement(subseq)
                orf_seqs.append(subseq)
            row_features['Protein_Density'] = len(orf_seqs)

            # simple protein property means if present
            for c in ['Aromaticity', 'Instability_index', 'Isoelectric_point']:
                if c in overlapped_prots.columns and len(overlapped_prots) > 0:
                    vals = pd.to_numeric(overlapped_prots[c], errors='coerce').dropna()
                    row_features[c] = vals.mean() if len(vals) > 0 else 0.0
                else:
                    row_features[c] = 0.0

            features_list.append(row_features)

    # create features dataframe
    df = pd.DataFrame(features_list)
    if df.empty:
        print("No features extracted. Exiting.")
        return

    # create island summary dataframe for diagnostics
    island_df = pd.DataFrame(island_summary)

    # Diagnostic printout
    if verbose:
        total_with_islands = island_df[island_df['num_accepted_islands'] > 0].shape[0]
        print("\n=== DIAGNOSTIC: ACCEPTED ISLANDS (post-merge + criteria) ===")
        print(f"Phages analyzed: {len(island_df)}. Phages with ≥1 accepted island: {total_with_islands}")
        if total_with_islands > 0:
            print("Examples (first 10):")
            sample_rows = island_df[island_df['num_accepted_islands'] > 0].head(10)
            for _, r in sample_rows.iterrows():
                print(f"- {r['Phage_ID']}: {r['accepted_islands']} | #args={r['num_args']} mobile_hits={r['num_mobile_positions']} genome_GC={r['genome_gc']:.3f}")

    # Now split into train/test by phage using earlier train_phages/test_phages
    df_train = df[df['Phage_ID'].isin(train_phages)]
    df_test = df[df['Phage_ID'].isin(test_phages)]

    # Preprocess: drop id columns and impute/scale
    drop_cols = ['Phage_ID', 'Label', 'Window_Start']
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    Xtrain_raw = df_train.drop(columns=drop_cols)
    Xtest_raw = df_test.drop(columns=drop_cols)

    Xtrain_imp = imputer.fit_transform(Xtrain_raw)
    Xtest_imp  = imputer.transform(Xtest_raw)

    Xtrain_scaled = scaler.fit_transform(Xtrain_imp)
    Xtest_scaled  = scaler.transform(Xtest_imp)

    # LASSO for feature selection
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(Xtrain_scaled, df_train['Label'].values)

    coef_mask = np.abs(lasso.coef_) > 1e-8
    if coef_mask.sum() < 5:
        idxs = np.argsort(-np.abs(lasso.coef_))[:5]
        coef_mask = np.zeros_like(lasso.coef_, dtype=bool)
        coef_mask[idxs] = True

    selected_indices = np.where(coef_mask)[0]
    X_train = Xtrain_scaled[:, selected_indices]
    X_test  = Xtest_scaled[:, selected_indices]

    if verbose:
        print(f"\n[CHECKPOINT] LASSO selected {len(selected_indices)} features.")

    # PCA
    n_components = min(50, X_train.shape[1])
    if n_components < 1:
        print("Not enough features selected for PCA. Exiting.")
        return
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    y_train = df_train['Label'].values
    X_test = pca.transform(X_test)
    y_test = df_test['Label'].values

    if verbose:
        print(f"\n[CHECKPOINT 2] DIMENSIONALITY REDUCTION COMPLETE")
        print(f"PCA Variance Explained: {np.sum(pca.explained_variance_ratio_):.4f}")
        print("--------------------------------------------------")

    # HMM training
    X_bg = X_train[y_train == 0]
    X_island = X_train[y_train == 1]

    if len(X_island) < 5:
        print("CRITICAL WARNING: not enough island examples in training; aborting.")
        return

    model = hmm.GaussianHMM(n_components=2, covariance_type="full", init_params="", verbose=False)
    model.startprob_ = np.array([0.99, 0.01])
    model.transmat_ = np.array([[0.98, 0.02], [0.05, 0.95]])
    model.means_ = np.array([X_bg.mean(axis=0), X_island.mean(axis=0)])
    cov_bg = np.cov(X_bg.T) + np.eye(model.means_.shape[1]) * 0.1
    cov_island = np.cov(X_island.T) + np.eye(model.means_.shape[1]) * 0.1
    model.covars_ = np.array([cov_bg, cov_island])
    model.n_features = model.means_.shape[1]

    if verbose:
        print(f"\n[CHECKPOINT 3] MODEL TRAINED")

    # predict on test
    lengths_test = df_test.groupby('Phage_ID').size().values
    if X_test.shape[0] == 0:
        print("No test data available after split.")
        return

    y_pred_viterbi = model.predict(X_test, lengths_test)

    if verbose:
        print("\n=== EVALUATION ON TEST DATASET ===")
        print(classification_report(y_test, y_pred_viterbi, target_names=['Background', 'Island']))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred_viterbi))
        print("\n[CHECKPOINT 4] VISUALIZING POSTERIOR DECODING (Uncertainty Analysis)")

    # Visualize some phages
    test_phages_with_island = df_test[df_test['Label'] == 1]['Phage_ID'].unique()
    if len(test_phages_with_island) > 0 and verbose:
        viz_phages = random.sample(list(test_phages_with_island), min(3, len(test_phages_with_island)))
        for pid in viz_phages:
            mask = (df_test['Phage_ID'] == pid).values
            X_p = X_test[mask]
            windows = df_test[mask]['Window_Start'].values
            y_true_p = df_test[mask]['Label'].values
            y_vit = model.predict(X_p, [len(X_p)])
            try:
                y_post = model.predict_proba(X_p, [len(X_p)])[:, 1]
            except:
                y_post = np.zeros(len(X_p))
            visualize_decoding(pid, windows, y_true_p, y_vit, y_post)
    else:
        if verbose:
            print("No ARG-enriched islands found in Test Set to visualize.")

    # merge predicted windows into islands (post-HMM) and filter by min length
    df_test = df_test.copy().reset_index(drop=True)
    df_test['Predicted'] = y_pred_viterbi.astype(int)
    all_pred_islands = []
    for pid in df_test['Phage_ID'].unique():
        df_p = df_test[df_test['Phage_ID'] == pid]
        pred_positions = df_p[df_p['Predicted'] == 1]['Window_Start'].values
        merged = merge_windows_to_ranges(pred_positions, window_size=window_size, step=step)
        merged_filtered = [r for r in merged if (r[1] - r[0]) >= min_island_length_bp]
        for (s,e) in merged_filtered:
            all_pred_islands.append({'Phage_ID': pid, 'pred_island_start': s, 'pred_island_end': e, 'length': e-s})
    pred_islands_df = pd.DataFrame(all_pred_islands)

    if verbose:
        if pred_islands_df.empty:
            print("\nNo predicted islands after merging + length filtering.")
        else:
            print(f"\nPredicted islands after merging (count = {len(pred_islands_df)}). Example:")
            print(pred_islands_df.head())

    if verbose:
        print("\nPIPELINE COMPLETE. Returning results (no CSVs written).")

    return {
        'features_df': df,
        'accepted_island_summary': island_df,
        'predicted_islands': pred_islands_df,
        'hmm_model': model,
        'pca': pca,
        'lasso': lasso,
        'scaler': scaler,
        'imputer': imputer
    }

# Example run (adjust path)
my_folder = r"C:\Users\subak\OneDrive\Documents\Comp Bio"
outputs = run_pipeline(
    my_folder,
    window_size=1000,
    step=250,
    k=4,
    island_radius=3000,
    min_args_for_island=1,
    min_mobile_count=2,
    gc_shift_threshold=0.1,
    min_island_length_bp=2000,
    verbose=True
)
