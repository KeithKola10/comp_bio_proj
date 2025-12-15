
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

path = '/Users/kolak/comp_bio_proj/analysis_data' #path to data file
db_name = 'TemPhD' #database name

data_obj = DataFormat(path,db_name) #object



# %% Generate Data


number_of_AMR_seqs = None # number of phage AMR sequences in the dataset, default ia All
Number_of_control_seqs = None # number of phage non-AMR sequences, default is 0
high_quality_only = False #Filter for reads that are high-quality only
high_med_quality_only = True #filter for reads that are medium and high quality
save_fname = None #export file name, Also doubles as True for save data, data is not saved automatically
MR_proteins_Only = True #only AMR protein sequences, overrides number_of_AMR_seqs and Number_of_control_seqs

data_obj.create_data_file(number_of_AMR_seqs=None,AMR_proteins_Only=True,save_fname = 'TemPhD_All') #Main function for creating a dataset

#data_obj.export_csv(file_name='TemPhD_All_data', data=data_obj.data_df) #export to csv

# %%
