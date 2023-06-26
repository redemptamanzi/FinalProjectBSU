import csv
import string
import pickle
import textstat
import numpy as np
import pandas as pd

from tqdm import tqdm
from langdetect import detect
from lingfeat import extractor #from lingfeatBASE.lingfeat import extractor
from spellchecker import SpellChecker


# Load data
all_queries = pd.read_pickle('../data/AllQueries4746.p')

# The following block of code extract linguistic features with lingFeat package

def lingFeatExtract(queries):
    
    all_queries = queries
    
    # Pass the text into an extractor
    all_queries['extract'] = ''
    for i in range(len(all_queries)):
        all_queries['extract'][i] = extractor.pass_text(str(all_queries['query'][i]))
        
    # -- Discourse Feat ---
    
    # Extract Entity Density Features (EnDF_)
    all_queries['EnDF_'] = ''
    for i in range(len(all_queries)):
        all_queries['EnDF_'][i] = all_queries['extract'][i].preprocess()
        
    #  Extract entity Grid Features (EnGF_)
    EnGF_Def = {'ra_SSToT_C':0,'ra_SOToT_C':0,'ra_SXToT_C':0,'ra_SNToT_C':0,'ra_OSToT_C':0,'ra_OOToT_C':0,'ra_OXToT_C':0,'ra_ONToT_C':0,'ra_XSToT_C':0,'ra_XOToT_C':0,'ra_XXToT_C':0,'ra_XNToT_C':0,'ra_NSToT_C':0,'ra_NOToT_C':0,'ra_NXToT_C':0,'ra_NNToT_C':0,'LoCohPA_S':0,'LoCohPW_S':0,'LoCohPU_S':0,'LoCoDPA_S':0,'LoCoDPW_S':0,'LoCoDPU_S':0}
    all_queries['EnGF_'] = ''
    for i in range(len(all_queries)):
        try:
            all_queries['EnGF_'][i] = all_queries['extract'][i].EnGF_()
        except:
            all_queries['EnGF_'][i] = EnGF_Def
            
    # ----- Syntactic -----
            
    # Extract Phrasal Features  (PhrF_)      
    all_queries['PhrF_'] = ''
    for i in range(len(all_queries)):
        all_queries['PhrF_'][i] = all_queries['extract'][i].PhrF_()
        
    # Extract Tree Structure Features (TrSF_)
    all_queries['TrSF_'] = ''
    for i in range(len(all_queries)):
        all_queries['TrSF_'][i] = all_queries['extract'][i].TrSF_()
    
    # Extract Part-of-Speech Features (POSF_)
    all_queries['POSF_'] = ''
    for i in range(len(all_queries)):
        all_queries['POSF_'][i] = all_queries['extract'][i].POSF_()
    
    # ----- Lexico Semantic ------
    
    # Extract Variation Ratio Features (VarF_)
    all_queries['VarF_'] = ''
    for i in range(len(all_queries)):
        all_queries['VarF_'][i] = all_queries['extract'][i].VarF_()
    
    # Extract Type Token Ratio Features (TTRF_)
    TTRF_def = {'SimpTTR_S':1, 'CorrTTR_S':1, 'BiLoTTR_S':0, 'UberTTR_S':0, 'MTLDTTR_S':0.72}
    all_queries['TTRF_'] = ''
    for i in range(len(all_queries)):
        try:
            all_queries['TTRF_'][i] = all_queries['extract'][i].TTRF_()
        except:
            all_queries['TTRF_'][i] = TTRF_def

    # Extract Psycholinguistic Features (PsyF_)
    all_queries['PsyF_'] = ''
    for i in range(len(all_queries)):
        all_queries['PsyF_'][i] = all_queries['extract'][i].PsyF_()
    
    # Extract Word Familiarity (WorF_)
    all_queries['WorF_'] = ''
    for i in range(len(all_queries)):
        all_queries['WorF_'][i] = all_queries['extract'][i].WorF_()
    
    # ----- Shallow Traditional -----
    
    # Extract Shallow Features (ShaF_)
    all_queries['ShaF_'] = ''
    for i in range(len(all_queries)):
        all_queries['ShaF_'][i] = all_queries['extract'][i].ShaF_()
    
    # Extract Traditional Formulas (TraF_)
    all_queries['TraF_'] = ''
    for i in range(len(all_queries)):
        all_queries['TraF_'][i] = all_queries['extract'][i].TraF_()

    #-- queries and extracted lingFeat  
    lingFeat_data = all_queries

# The output of the above block is dataframe that contain column's entry 
# in form dictionary. The block of code below transform the results above 
# into single entries 

    # The following function extracts transform the key of dictonaries into the columns of a 
    # dataframe and their corresponding values into their corresponding entries

    def feat_extract(dataName, featName):
        """ 
        Steps:
        1. Get the keys of a dict to be used as columns in this dataframe.
        2. Initially the dictionally are stings. eval() and .replace() are used to convert the str into a dict

        """
        cols = eval(dataName[featName][0].replace("'", "\""))
        df_tot = pd.DataFrame(columns = list(cols.keys()))
        for i in range(len(dataName)):
            f = eval(dataName[featName][i].replace("'", "\""))
            val = np.array(list(f.values())).reshape(-1,1).T # reshape the dict value to become the column entries
            df = pd.DataFrame(data = val, columns = list(f.keys()))
            df_tot = pd.concat([df_tot, df])

        return df_tot

    # preprocess
    preprocess = feat_extract(lingFeat_data,'preprocess')
    # Discourse (Disco)
    EntityDensityF = feat_extract(lingFeat_data,'EnDF_')
    EntityGridF = feat_extract(lingFeat_data,'EnGF_')
    # Syntactic (Synta)
    PhrasalF = feat_extract(lingFeat_data,'PhrF_')
    TreeStructureF = feat_extract(lingFeat_data,'TrSF_')
    PartOfSpeechF = feat_extract(lingFeat_data,'POSF_')
    # Lexico Semantic (LxSem)
    TypeTokenRatioF = feat_extract(lingFeat_data,'TTRF_')
    VariationRatioF = feat_extract(lingFeat_data,'VarF_')
    PsycholinguisticF = feat_extract(lingFeat_data,'PsyF_')
    WordFamiliarityF = feat_extract(lingFeat_data,'WorF_')
    # Shallow Traditional (ShTra)
    ShallowF = feat_extract(lingFeat_data,'ShaF_')
    TraditionalFormulas = feat_extract(lingFeat_data,'TraF_')

    allFeatures = pd.concat([preprocess, 
                        EntityDensityF, 
                        EntityGridF, 
                        PhrasalF, 
                        TreeStructureF, 
                        PartOfSpeechF, 
                        TypeTokenRatioF, 
                        VariationRatioF, 
                        PsycholinguisticF, 
                        WordFamiliarityF, 
                        ShallowF, 
                        TraditionalFormulas], axis=1) 


    return allFeatures
    
