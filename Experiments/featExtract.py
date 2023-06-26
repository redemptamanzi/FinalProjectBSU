import re
import os
import csv
import time
import nltk
import pickle
import string
import warnings
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from spellchecker import SpellChecker

# from langdetect import detect
# from lingfeat import extractor # from lingfeat directory

# The 'queries' parameter in each function can be
#       - Single query
#       - List
#       - Series
#       - Dataframe that contain a column named 'query'

#************************SPELLING*************************************************

def Spelling(queries):
    import pandas
    
    if type(queries) == str:
        
        new_df=pd.DataFrame({
            'query': queries
        }, index=[0])
        
        queries=[queries]
    
    elif type(queries) == list:
        
        new_df=pd.DataFrame({
            'query': queries
        })
    
    elif type(queries) == pandas.core.frame.DataFrame:
        new_df=queries.copy()
        queries= queries.iloc[:,0].tolist()
        
    elif type(queries) == pandas.core.series.Series:
        new_df=pd.DataFrame(queries, columns=['query'])
        queries=queries.tolist()
        
    
    #q = queries.copy()
    #queries=queries[query_col].tolist()
    
    wsle=pd.read_csv('../data/KidSpell/Web_Search_Lab_Errors.csv',
                usecols=['spelling']).spelling.tolist()
    
    wsie=pd.read_csv('../data/KidSpell/Web_Search_Informal_Errors.csv',
                usecols=['spelling']).spelling.tolist()
    
    ewe=pd.read_csv('../data/KidSpell/Essay_Writing_Errors.csv',
                usecols=['Spelling']).Spelling.tolist()
    
    kidsMispelled= set(wsle+wsie+ewe)
    
    spell = SpellChecker()
    kidsError=[]
    oneOffError = []
    misspelledCol=[]
    for i, query in enumerate(queries):

        query=query.translate(str.maketrans('', '', string.punctuation)) # -- remove all panctuations

        misspelled=spell.unknown(query.split()) 

        missed_k=misspelled.intersection(kidsMispelled)
        kidsError.append(len(missed_k))
        misspelledCol.append(len(misspelled))

        oneOff = 0
        try:
            for word in misspelled:
                mis_one=spell.edit_distance_1(word).\
                intersection(spell.candidates(word))

                if len(mis_one) > 0:
                    oneOff+=1
            oneOffError.append(oneOff)

        except:
            oneOffError.append(-1)

    #     if len(missed_k) != 0:
    #         print(i, len(missed_k), missed_k)
        #print(len(misspelled.intersection(kidsMispelled)))
        #print(misspelled)
        
    allFeatures=(
        new_df
        .assign(kidsError=kidsError)
        .assign(misspelledCol=misspelledCol)
        .assign(oneOffError=oneOffError)
    )
#     print(allFeatures)
    
    return allFeatures


#***********************PUNCTUATION and CASING***********************************

def Punct_Casing(queries):
    
    import pandas
    
    if type(queries) == str:
        
        new_df=pd.DataFrame({
            'query': queries
        }, index=[0])
        
        queries=[queries]
    
    elif type(queries) == list:
        
        new_df=pd.DataFrame({
            'query': queries
        })
    
    elif type(queries) == pandas.core.frame.DataFrame:
        new_df=queries.copy()
        queries= queries.iloc[:,0].tolist()
        
    elif type(queries) == pandas.core.series.Series:
        new_df=pd.DataFrame(queries, columns=['query'])
        queries=queries.tolist()
    
    invalidcharacters= set(['!', ',', '.', '?'])
    punct = []
    casing = []
    with tqdm(total = len(queries)) as pbar:
        for query in queries:

            if any(char in invalidcharacters for char in query):
                punct.append(1)
            else: 
                punct.append(0)

            if query.islower():
                casing.append(0)
            else:
                casing.append(1)
            pbar.update()

    allFeatures=(
        new_df
        .assign(punct=punct)
        .assign(casing=casing)
    )
    
    return allFeatures

#************************CONCRETENESS*********************************************

def absConcFeat(queries):
    
    import pandas
    
    if type(queries) == str:
        
        new_df=pd.DataFrame({
            'query': queries
        }, index=[0])
        
        queries=[queries]
    
    elif type(queries) == list:
        
        new_df=pd.DataFrame({
            'query': queries
        })
    
    elif type(queries) == pandas.core.frame.DataFrame:
        new_df=queries.copy()
        queries= queries.iloc[:,0].tolist()
        
    elif type(queries) == pandas.core.series.Series:
        new_df=pd.DataFrame(queries, columns=['query'])
        queries=queries.tolist()
    
    word_concreteness = pickle.load( open( "../data/concreteness/word_concreteness.p", "rb" ) )
    word_concreteness['word']=word_concreteness['word'].str.lower() 

    aw = word_concreteness[word_concreteness['label']=='abstract']
    cw = word_concreteness[word_concreteness['label']=='concrete']

    abW = []
    coW = []
    for w_a in aw['word']:
        abW.append(w_a)
    for w_c in cw['word']:
        coW.append(w_c)

    absrtCount = []
    concCount = []
    with tqdm(total=len(queries)) as pbar:
        for query in queries:

            a_countWord = 0
            c_countWord = 0
            wordCount = 0
            for word in query.split(' '):
                word.lower()
                wordCount +=1
                if word in abW:
                    a_countWord +=1
                if word in coW:
                    c_countWord +=1

            absrtCount.append(a_countWord/wordCount)
            concCount.append(c_countWord/wordCount)
            pbar.update()

    allFeatures=(
        new_df
        .assign(ratioAbs=absrtCount)
        .assign(ratioConc=concCount)
    )

    return allFeatures
