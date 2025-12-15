import numpy as np
import pandas as pd
import pickle
from Bio import SeqIO
import ipdb
import random
random.seed(10)
import re



class DataFormat():
    '''Class for merging all of the data from the phagescope database
    
    ***Put all files into one folder with no changes in the file names***
    
    ***file Names:
    
    -TemPhD # folder with all of the fasta files
    -TemPhD_antimicrobial_resistance_gene_data.tsv
    -temphd_phage_annotated_protein_meta_data.tsv
    -temphd_phage_meta_data.tsv
    -TemPhD.gff3
    
    ******'''
    
    def __init__(self,folderpath, DBName):
        '''Intitialization takes the name of the database data and the pth to the folder containing all of the data'''
        
        self.path = folderpath #Path to data folder
        self.DB = DBName #Name of database 
        
        
        self.amr_tsv = '%s_antimicrobial_resistance_gene_data' % self.DB
        self.protein_tsv = '%s_phage_annotated_protein_meta_data' % self.DB.lower()
        self.gff3_file = '%s' % self.DB
        self.meta_tsv = '%s_phage_meta_data'% self.DB.lower()
        
        
        self.data_df = None #Data frame with all of the merged data
        self.fasta_path = '%s/%s' % (self.path,self.DB)#fasta folder path
        
    #Functions for loading and writing files 
    def __load_tsv__(self,file_name):
        '''Function for loading .tsv files'''
        
        path = '%s/%s.tsv' % (self.path,file_name)
        df = pd.read_csv(path,sep="\t",dtype=str)
        return df
        
    def __load_gff3__(self,file_name):
        '''Function for loading .gff3 file'''
        
        path = '%s/%s.gff3' % (self.path,file_name)
        df = pd.read_csv(path,sep="\t",header=None,comment="#",names=["seqid", "source", "type", "start", "end",
                        "score", "strand", "phase", "attributes"])
        return df

    
    def __load_fasta__(self,file_name):
        '''Function for loading .fasta file'''
        
        path = '%s/%s.fasta' % (self.fasta_path,file_name)
        fast_list = list(SeqIO.parse(path, "fasta"))
        
        df = pd.DataFrame([{"id": rec.id,"description": rec.description,"sequence": str(rec.seq),
                            "length": len(rec.seq)} for rec in fast_list])
        return df
        
    def load_data_pickle(self, file_name):
        '''Load Pickle File'''
        
        path = '%s/%s.pickle' % (self.path,file_name)
        f = open(path, 'rb')
        data = pickle.load(f)
        f.close()
        
        return data
    
    def savePickle(self,file_name,data):
        '''Save Data as Pickle'''
        
        path = '%s/%s.pickle' % (self.path,file_name)
        f = open(path, 'wb')
        pickle.dump(data, f)
        f.close()
        
        return
        
    
    def export_csv(self,file_name,data):
        '''export data as .csv'''
        
        path = '%s/%s.csv' % (self.path,file_name)
        data.to_csv(path)
        
        return
        
    #Tools
    
    def reverse_complement(self,seq):
        """Return the reverse complement of a DNA sequence."""
        complement = str.maketrans("ACGTacgt", "TGCAtgca")
        return seq.translate(complement)[::-1]
    
    def __get_Phage_seqs__(self,phage_ids,protein_df):
        '''returns all phage sequences and protein sequences from list of phage_ids'''
        

        df_seqs = self.__load_fasta__(phage_ids[0])
        for x in range(1, len(phage_ids)):
            new_row = self.__load_fasta__(phage_ids[x])
            df_seqs = pd.concat([df_seqs, new_row], ignore_index=True)

        # build a lookup dict
        phage_to_seq = dict(zip(df_seqs["id"], df_seqs["sequence"]))
        phage_to_desc  = dict(zip(df_seqs["id"], df_seqs["description"]))
        phage_to_len   = dict(zip(df_seqs["id"], df_seqs["length"]))    

        rows = []

        for x, pro_row in protein_df.iterrows():
            phage_id = pro_row["Phage_ID"]
            start = int(pro_row["Start"])
            stop = int(pro_row["Stop"])
            strand = pro_row["Strand"]
            prot_id = pro_row.get("Protein_ID")

            genome_seq = phage_to_seq.get(phage_id)
            if genome_seq is None:
                dna_seq = None
            else:
                # Prodigal/GFF: [start, stop] is 1-based inclusive
                start_i = start - 1   # 0-based
                stop_i = stop         # exclusive for Python slicing
                dna_seq = genome_seq[start_i:stop_i]
                if strand == "-":
                    dna_seq = self.reverse_complement(dna_seq)

            rows.append(
                {    
                    "Protein_ID": prot_id,
                    "Protein Sequence": dna_seq,
                    #"Genome Sequence": genome_seq, 
                    "description": phage_to_desc.get(phage_id),
                    "length": phage_to_len.get(phage_id), 
                }
            )

        df_out = pd.DataFrame(rows)
        return df_out
     
    def __get_phages_with_no_AMR__(self,phage_AMR_ids,meta_data, n):
        '''Generates a list of non-AMR phage ids'''
        non_AMR_ids = []
        
        meta_data_AMR_removed = meta_data[~meta_data["Phage_ID"].isin(phage_AMR_ids)].reset_index()
        meta_data_AMR_removed_ids = meta_data_AMR_removed["Phage_ID"].tolist()
        
        non_AMR_ids = random.sample(meta_data_AMR_removed_ids, n)
    
        return non_AMR_ids
            
            
    def __get_phages_with_AMR__(self,phage_AMR_ids,meta_data, n, get_all_AMR = False):
        '''generate a list of AMR ids '''
        
        AMR_ids = []
        
        meta_data_AMR = meta_data[meta_data["Phage_ID"].isin(phage_AMR_ids)].reset_index()
        meta_data_AMR_ids = meta_data_AMR["Phage_ID"].tolist()
        
        if get_all_AMR is not True:
            AMR_ids = random.sample(meta_data_AMR_ids, n)
        else:
            AMR_ids = meta_data_AMR_ids
    
        return AMR_ids
        
    def __remove_repeated_cols_df__(self,df):
        ''' Drop duplicate columns from data frame merger'''
        
        unique_cols = []
        seen = set()

        for col in df.columns:
            # Remove typical merge suffixes: _x, _y, _left, _right
            base = re.sub(r'(_x$|_y$|_left$|_right$)', '', col, flags=re.IGNORECASE)
            key = base.lower()
            
            if key not in seen:
                seen.add(key)
                unique_cols.append(col)

        return df[unique_cols]  
    
    def __drop_index_columns__(self,df):
    # drop things like 'index', 'index_x', 'index_y', 'level_0', 'Unnamed: 0', etc.
        bad = [c for c in df.columns
            if c.lower().startswith("index")
            or c.lower().startswith("level_0")
            or c.lower().startswith("unnamed")]
        return df.drop(columns=bad, errors="ignore")
        
    #Functions for creating datafile 
    def create_data_file(self, number_of_AMR_seqs = None, Number_of_control_seqs = None, 
                         high_quality_only = False, high_med_quality_only = True, save_fname = None,
                         AMR_proteins_Only = False):
        
        '''Main function for creating datasets:
        
        Inputs:
        
        number_of_AMR_seqs = None # number of phage AMR sequences in the dataset, default ia All
        Number_of_control_seqs = None # number of phage non-AMR sequences, default is 0
        high_quality_only = False #Filter for reads that are high-quality only
        high_med_quality_only = True #filter for reads that are medium and high quality
        save_fname = None #export file name, Also doubles as True for save data, data is not saved automatically
        MR_proteins_Only = True #only AMR protein sequences, overrides number_of_AMR_seqs and Number_of_control_seqs
        
        outputs:
        saves a .pickle file if save_fname is passed
        initilizes object with data in self.data_df'''
        
        
        #Load in all data files
        phage_AMR_df = self.__load_tsv__(self.amr_tsv)
        gff3_df = self.__load_gff3__(self.gff3_file)
        protein_df = self.__load_tsv__(self.protein_tsv)
        meta_df = self.__load_tsv__(self.meta_tsv)
        
        
        #get All AMR phage data
        phage_AMR_ids = phage_AMR_df['Phage_id']
        
        #filter based on completness, Can also filter based on score in gff3 but this seems fine
        if high_quality_only is True:
            meta_df = meta_df[meta_df["Completeness"] == "High-quality"].reset_index()
        elif high_med_quality_only is True:
            meta_df = meta_df[(meta_df["Completeness"] == "High-quality") | (meta_df["Completeness"] == "Medium-quality")].reset_index()

        #get a number of AMR phage ids
        if number_of_AMR_seqs is None:
            number_of_AMR_seqs = len(phage_AMR_ids)
            get_all_AMR = True
        phage_AMR_ids_pos = self.__get_phages_with_AMR__(phage_AMR_ids,meta_df, number_of_AMR_seqs,get_all_AMR = get_all_AMR)
        
        #get a number of non AMR phage ids
        if Number_of_control_seqs is not None:
            phage_control_ids = self.__get_phages_with_no_AMR__(phage_AMR_ids,meta_df, Number_of_control_seqs)
        else:
            phage_control_ids = []
            
        #Merge control and positive list
        full_list = phage_AMR_ids_pos+phage_control_ids
        
        
        #filter all data
        gff3_df = gff3_df.rename(columns={"seqid":"Phage_ID", "start":"Start","end":"Stop","strand":"Strand"})
        gff3_df = gff3_df[gff3_df["Phage_ID"].isin(full_list)].reset_index()
        protein_df = protein_df[protein_df["Phage_ID"].isin(full_list)].reset_index()
        meta_df = meta_df[meta_df["Phage_ID"].isin(full_list)].reset_index()
        
        #Loada fasta files
        seq_df = self.__get_Phage_seqs__(full_list,protein_df=protein_df)
        seq_df = seq_df.rename(columns={"id":"Phage_ID"})

        #drop index cols
        
        gff3_df=self.__drop_index_columns__(gff3_df)
        protein_df=self.__drop_index_columns__(protein_df)
        meta_df=self.__drop_index_columns__(meta_df)
        seq_df= self.__drop_index_columns__(seq_df)
        #merge dfs
        df_one = meta_df.merge(protein_df, on="Phage_ID", how="left")
        
        
        df_two = seq_df.merge(df_one,on="Protein_ID", how="left")
    
        
        # numeric coordinates
        for col in ["Start", "Stop"]:
            gff3_df[col] = gff3_df[col].astype("int64")
            df_two[col]  = pd.to_numeric(df_two[col], errors="coerce").astype("Int64")  # or .astype("int64") if no NaNs

        # string IDs / strand
        for col in ["Phage_ID", "Strand"]:
            gff3_df[col] = gff3_df[col].astype(str).str.strip()
            df_two[col]  = df_two[col].astype(str).str.strip()
        
        
        merge_keys = ["Phage_ID", "Start", "Stop", "Strand"]

        df_all = gff3_df.merge(df_two,on=merge_keys,how="left",validate="one_to_one")
        
        #Remove duplicate coloumns
        self.data_df = self.__remove_repeated_cols_df__(df_all)
        
        #drop garbage cols for reduced file size
        self.data_df = self.data_df.drop(columns=['Function_prediction_source','Protein_source','Product','Taxonomy',
                                                  'Lifestyle','Cluster','Subcluster','source','type']).reset_index()
        #Filter for only AMR proteins
        if AMR_proteins_Only is True:
             self.data_df = self.data_df[self.data_df["Protein_ID"].isin(phage_AMR_df['Protein_id'])].reset_index()
            
        if save_fname is not None:
            self.savePickle(save_fname,self.data_df)
            

        return