import os,sys
import numpy as np
import pandas as pd

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname) 
pd.set_option('future.no_silent_downcasting', True)
candidates_raw = pd.read_html("https://en.wikipedia.org/wiki/Candidates_of_the_2025_Australian_federal_election")

def remove_square_brackets(astr):
    if "[" in astr and "]" in astr:
        return remove_square_brackets(astr[:astr.find("[")] + astr[astr.find("]")+1:])
    return astr

def remove_round_brackets(astr):
    if "(" in astr and ")" in astr:
        return remove_square_brackets(astr[:astr.find("(")-1] + astr[astr.find(")")+1:]) #-1 is to get rid of space
    return astr

def remove_brackets_df(table):
    outtable = table.copy().replace("nan",np.nan)
    outtable.columns = [remove_square_brackets(col) for col in outtable.columns]
    if 'Liberal' in outtable.columns:
        outtable['Coalition'] = outtable['Liberal'] + " (L)"
        outtable = outtable.drop(['Liberal'], axis=1)
    if 'LNP' in outtable.columns:
        outtable['Coalition'] = outtable['LNP'] + " (LNP)"
        outtable = outtable.drop(['LNP'], axis=1)
    if "Others" in outtable.columns:
        outtable['Other'] = outtable['Others']
        outtable = outtable.drop(['Others'], axis=1)
    # return outtable
    return outtable.astype(str).map(remove_square_brackets)

candidates_df_raw = pd.concat([remove_brackets_df(tb.astype(str)) for tb in candidates_raw if 'Electorate' in tb.columns]).replace({"nan":np.nan}).dropna(how='all',axis=0)
candidates_df_raw['Other'] = candidates_df_raw['Other'].fillna('').apply(lambda x: remove_round_brackets(x))
candidates_df_raw['Coalition'] = candidates_df_raw['Coalition'].fillna('').apply(lambda x: remove_round_brackets(x))

candidates_df_raw.rename(columns={"Electorate":"DivisionNm"},inplace=True)
candidates_df_raw.set_index("DivisionNm",inplace=True)
candidates_df_raw.drop("Held by",axis=1,inplace=True)

teals2025 = pd.read_csv("processed/teals2025.csv",index_col="DivisionNm")
candidates_df_raw['TEAL'] = teals2025.apply(lambda x: x["GivenNm"] + " " + x["Surname"].capitalize(), axis=1)

def clean_ind(row):
    ind_str = row['Independent']
    teal_str = row['TEAL']

    if teal_str in ind_str:
        new_ind = ind_str.replace(teal_str, '').strip()
        new_ind = ' '.join(new_ind.split())  # Removes extra spaces
        return new_ind
    return ind_str

candidates_df_raw['Independent'] = candidates_df_raw.fillna('').apply(clean_ind, axis=1)
candidate_counts = candidates_df_raw.fillna('').map(lambda x: x.count(' '))
candidate_counts['Labor'] = 1 # Assume Labor contests every seat
candidate_counts['Coalition'] = 1 # Assume Coalition contests every seat
candidate_counts['Greens'] = 1 # Assume Greens contests every seat
candidate_counts.loc['Kennedy', 'Independent'] = 1 # Ensure Katter is running in Kennedy

mapper = {"Labor":"ALP", "Coalition":"LNP", "Greens":"GRN", "One Nation":"ONP", "TEAL":"TEAL", "Independent":"IND"}
for col in candidate_counts.columns:
    if col not in mapper:
        mapper[col] = "OTH"

candidates = candidate_counts.rename(mapper,axis=1).T.groupby(level=0).sum().T
candidates.to_csv("processed/candidates.csv")