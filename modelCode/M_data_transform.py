'''
A file that will transfrom the raw data a processed form that will make the model simpler to run.
Generally this is stripping the raw AEC data down to only required rows and then combining them into a single matrix
'''

# Globals
MODELED_PARTES = ['UAP', 'ONP', 'ALP', 'GRN', 'LNP', 'IND', 'TEAL', 'OTH']


# Imports
import os, sys
import numpy as np
import pandas as pd


# Functions
def load_distribution_data(columns:list[str],path="raw/distributions", *args, **kwargs):
    '''
    This file loops over the folders in the path and gets all the raw .csv. converts all the files into a single dataframe with columns specified by the columns parameter
    '''
    folders = [i for i in os.listdir(path) if i!=".gitignore"]
    files = []
    for folder in folders:
        files.extend(
            [f"{path}/{folder}/{filename}" for filename in os.listdir(f"{path}/{folder}")] # Searches through every subfolder and puts them all into a big list
    )

    data = pd.concat([pd.read_csv(file,*args,**kwargs) for file in files],axis=0)

    return data[columns]

def keeprows_distribution(vote_df:pd.DataFrame):
    vote_dfout = vote_df[(vote_df["CalculationType"] == "Transfer Percent") & (vote_df["CountNum"] > 0) & (vote_df["PartyAb"] != "Informal")].copy()
    return vote_dfout

def handle_teals(vote_df:pd.DataFrame, teals_path="teals.csv"):
    '''Loop over each row and where that is a teal candidate, adjust their partyAb accordingly'''
    teal_candidates = pd.read_csv(teals_path)
    outvote_df = vote_df.copy()
    for idx,teal in teal_candidates.iterrows():
        givenNm = teal['GivenNm']
        surname = teal['Surname']
        division = teal['DivisionNm']
        outvote_df.loc[(outvote_df[['GivenNm','Surname','DivisionNm']]==[givenNm,surname,division]).all(axis=1),'PartyAb']="TEAL"

    
    return outvote_df

def handle_party_names(vote_df:pd.DataFrame):
    party_mappers = {
        'UAPP':'UAP', 
        'ON':'ONP',
        'ALP':'ALP', 
        'IND':'IND', 
        'GRN':'GRN',
        'GVIC':'GRN',
        'LP':'LNP',
        'NP':'LNP',
        'CLP':'LNP',
        'LNP':'LNP',
        'XEN':'IND',
        'KAP':'IND',
    }
    def partymap(row):
        party = row['PartyAb']
        if party in MODELED_PARTES:
            party = party
        elif party in party_mappers.keys():
            party = party_mappers[party]
        elif pd.isna(party):
            if pd.isna(row['Surname']):
                party = "Informal"
            else:
                party = "IND"
        else:
            party = "OTH"
        row['PartyAb'] = party
        return row

    vote_df = vote_df.apply(partymap,axis=1)
    return vote_df

def transform_to_Ydist(vote_df):
    def fixrow(row):
        posvalues = row >0
        row[posvalues] = row[posvalues] / row[posvalues].sum()
        row[row<0] = -1
        return row

    Ydist = vote_df.pivot_table(index=['DivisionNm', 'CountNum', "PPId"], columns='PartyAb', values='CalculationValue', aggfunc='sum') # Get the columns that we want
    Ydist.columns.name = None # Tidy name of the columns
    Ydist =Ydist.apply(fixrow, axis=1).copy() # Get rows into the expected format
    Ydist = Ydist[Ydist.eq(-1).any(axis=1)] # Remove rows that don't have the right form. Typically when minor canidate with only 2 votes gets transfered to other minor party
    return Ydist

def load_primaries_data(columns:list[str],path="raw/primaries", *args, **kwargs):
    '''
    This file loops over the folders in the path and gets all the raw .csv. converts all the files into a single dataframe with columns specified by the columns parameter
    '''
    files = [f"{path}/{file}" for file in os.listdir(path)]

    data = pd.concat([pd.read_csv(file,*args,**kwargs) for file in files],axis=0)

    return data[columns]

def keeprows_primaries(vote_df:pd.DataFrame):
    vote_dfout = vote_df[(vote_df["Surname"] != "Informal")].copy()
    return vote_dfout

def transform_to_pref(vote_df):
    def fixrow(row):
        return row / np.nansum(row)

    Ydist = vote_df.pivot_table(index=['DivisionNm', 'PollingPlaceID'], columns='PartyAb', values='OrdinaryVotes', aggfunc='sum') # Get the columns that we want
    Ydist.columns.name = None # Tidy name of the columns
    Ydist =Ydist.apply(fixrow, axis=1).copy() # Get rows into the expected format
    # Ydist = Ydist[Ydist.eq(-1).any(axis=1)] # Remove rows that don't have the right form. Typically when minor canidate with only 2 votes gets transfered to other minor party
    return Ydist    

def transform_to_tcp(vote_df):
    def fixrow(row):
        return row / np.nansum(row)

    Ydist = vote_df.pivot_table(index=['DivisionNm', 'PollingPlaceID'], columns='PartyAb', values='OrdinaryVotes', aggfunc='sum') # Get the columns that we want
    Ydist.columns.name = None # Tidy name of the columns
    Ydist =Ydist.apply(fixrow, axis=1).copy() # Get rows into the expected format
    for party in MODELED_PARTES:
        if party not in Ydist.columns:
            Ydist[party] = np.nan

    Ydist.columns = [f"{col}_TCP" for col in Ydist.columns]

    return Ydist    




if __name__=="__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    distribution_columns = ["DivisionNm","PPId","CountNum","Surname",'GivenNm',"PartyAb","CalculationType","CalculationValue"]
    distribution_data = load_distribution_data(distribution_columns,header=1)
    distribution_data = keeprows_distribution(distribution_data)
    distribution_data = handle_teals(distribution_data)
    distribution_data = handle_party_names(distribution_data)
    distribution_data = transform_to_Ydist(distribution_data)
    distribution_data.to_csv("processed/distributions.csv")

    primary_columns = ["DivisionNm","PollingPlaceID","Surname",'GivenNm',"PartyAb","OrdinaryVotes"]
    primary_data = load_primaries_data(primary_columns,path="raw/primaries",header=1)
    primary_data = handle_teals(primary_data)
    primary_data = handle_party_names(primary_data)
    primary_data = keeprows_primaries(primary_data)
    primary_data = transform_to_pref(primary_data)

    tcp_columns = ["DivisionNm","PollingPlaceID","Surname",'GivenNm',"PartyAb","OrdinaryVotes"]
    tcp_data = pd.read_csv("raw/HouseTcpByCandidateByPollingPlaceDownload-27966.csv", header=1)[tcp_columns]
    tcp_data = handle_teals(tcp_data)
    tcp_data = handle_party_names(tcp_data)
    tcp_data = transform_to_tcp(tcp_data)

    merged_df = pd.concat([primary_data,tcp_data],axis=1).reset_index()
    
    merged_df["PPId"] = merged_df["PollingPlaceID"].astype(int)
    merged_df = merged_df.drop("PollingPlaceID",axis=1)
    merged_df.to_csv("processed/tcp_train.csv",index=False)
