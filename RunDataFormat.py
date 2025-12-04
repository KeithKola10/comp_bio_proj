# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 10:58:38 2025

@author: gabri
"""

'''Example script to organize and merge all of the for a data set from PhageScope
For this code to woork the following files are required:

    -TemPhD # folder with all of the fasta files
    -TemPhD_antimicrobial_resistance_gene_data.tsv
    -temphd_phage_annotated_protein_meta_data.tsv
    -temphd_phage_meta_data.tsv
    -TemPhD.gff3


Data can be saved and loaded as .pickle files. However, There is an option to export it to a .csv/
'''
#%%
from DataFormat import DataFormat

#%% Create Data obj

path = 'C:\\Users\\gabri\\OneDrive\\Desktop\\CompBio Project\\TemPhD' #path to data file
db_name = 'TemPhD' #database name

data_obj = DataFormat(path,db_name) #object



# %% Generate Data

number_of_AMR_seqs = None # number of phage AMR sequences in the dataset, default ia All
Number_of_control_seqs = None # number of phage non-AMR sequences, default is 0
high_quality_only = False #Filter for reads that are high-quality only
high_med_quality_only = True #filter for reads that are medium and high quality
save_fname = None #export file name, Also doubles as True for save data, data is not saved automatically
MR_proteins_Only = True #only AMR protein sequences, overrides number_of_AMR_seqs and Number_of_control_seqs

#data_obj.create_data_file(number_of_AMR_seqs=10) #Main function for creating a dataset

data_obj = DataFormat(path, db_name)

# Generate + SAVE dataframe
data_obj.create_data_file(
    number_of_AMR_seqs=10,
    Number_of_control_seqs=None,
    high_quality_only=False,
    high_med_quality_only=True,
    save_fname="amr_dataset",      # <-- this will create amr_dataset.pickle
    AMR_proteins_Only=True
)

# %%