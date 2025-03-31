import pandas as pd
import geopandas as gpd
import numpy as np
import os,sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


electorate_shapes = gpd.read_file("raw/CED_2024_AUST_GDA2020/CED_2024_AUST_GDA2020.shp")[["CED_CODE24", "CED_NAME24", "geometry"]].rename({"CED_NAME24":"2025Division","CED_CODE24":"2025DivisionCode"},axis=1)
polling_places = pd.read_csv("raw/GeneralPollingPlacesDownload-27966.csv",header=1,index_col="PollingPlaceID")[["PollingPlaceNm","Latitude", "Longitude","DivisionNm"]].rename({"DivisionNm":"2022Division"},axis=1)
gdf_points = gpd.GeoDataFrame(polling_places[["PollingPlaceNm","2022Division"]], geometry=gpd.points_from_xy(polling_places.Longitude, polling_places.Latitude), crs="EPSG:7844")
gdf_joined = gpd.sjoin(gdf_points, electorate_shapes, how="left", predicate="within")
PPMAPPER = gdf_joined.dropna()['2025Division']

files = [
    'HouseStateFirstPrefsByPollingPlaceDownload-27966-NT.csv',
    'HouseStateFirstPrefsByPollingPlaceDownload-27966-QLD.csv',
    'HouseStateFirstPrefsByPollingPlaceDownload-27966-ACT.csv',
    'HouseStateFirstPrefsByPollingPlaceDownload-27966-TAS.csv',
    'HouseStateFirstPrefsByPollingPlaceDownload-27966-WA.csv',
    'HouseStateFirstPrefsByPollingPlaceDownload-27966-VIC.csv',
    'HouseStateFirstPrefsByPollingPlaceDownload-27966-SA.csv',
    'HouseStateFirstPrefsByPollingPlaceDownload-27966-NSW.csv'
]

# votes2022 = pd.read_csv('raw/primaries/HouseStateFirstPrefsByPollingPlaceDownload-27966-NSW.csv',header=1)
votes2022 = pd.read_csv(f"raw/primaries/{files[0]}",header=1)
for file in files[1:]:
    votes2022 = pd.concat([votes2022,pd.read_csv(f"raw/primaries/{file}",header=1)])
votes2022allVoteTypes = pd.read_csv("raw/primaries/HouseFirstPrefsByCandidateByVoteTypeDownload-27966.csv",header=1)
votes2022allVoteTypes = votes2022allVoteTypes[["DivisionNm", "PartyAb", "Surname", "GivenNm","CandidateID","OrdinaryVotes", "AbsentVotes", "ProvisionalVotes", "PrePollVotes", "PostalVotes", "TotalVotes"]]

def handle_no_location_PPIds(votes2022):
    PPIDs_no_loc = []
    for index, row in votes2022.iterrows():
        if row['PollingPlaceID'] in PPMAPPER.index:
            continue
        else:
            PPIDs_no_loc.append(row['PollingPlaceID'])
    return np.unique(PPIDs_no_loc)

def handle_row(row):
    if row['PollingPlaceID'] in PPMAPPER.index:
        newDIV = PPMAPPER.loc[row['PollingPlaceID']]
    else:
        print(row)
        newDIV = row['DivisionNm']

    if pd.isna(row['PartyAb']) and row['Surname']=="Informal":
        partyab = "INF"
    elif pd.isna(row['PartyAb']):
        partyab = "IND"
    else:
        partyab = row['PartyAb']

    og_numPPIDs = len(votes2022[votes2022["DivisionNm"]==row['DivisionNm']]["PollingPlaceID"].unique())
    candidate_idx = votes2022allVoteTypes[(votes2022allVoteTypes[["DivisionNm","Surname", "GivenNm"]] == [row['DivisionNm'],row['Surname'], row['GivenNm']]).all(axis=1)].index
    if len(candidate_idx)==1:
        candidate_idx=candidate_idx[0]
    elif len(candidate_idx)==0:
        candidate_idx = votes2022allVoteTypes[(votes2022allVoteTypes[["DivisionNm","Surname"]] == [row['DivisionNm'],row['Surname']]).all(axis=1)].index[0]

    else:
        print([row['DivisionNm'],row['Surname'], row['GivenNm'],row['PartyAb']])
        print(candidate_idx)
        raise Exception("Error multiple candidates detected. Should only be one.")

    candidate_nonloc_ordinary = votes2022allVoteTypes.loc[candidate_idx]["OrdinaryNoLocVotes"]
    candidate_absentee = votes2022allVoteTypes.loc[candidate_idx]["AbsentVotes"]
    candidate_provisional = votes2022allVoteTypes.loc[candidate_idx]["ProvisionalVotes"]
    candidate_prepoll = votes2022allVoteTypes.loc[candidate_idx]["PrePollVotes"]
    candidate_postal = votes2022allVoteTypes.loc[candidate_idx]["PostalVotes"]



    return pd.Series(
        {"PollingPlaceID":row['PollingPlaceID'],
         "DivisionNm":newDIV,
         "PartyAb":partyab,
         "Surname":row['Surname'],
         "GivenNm":row["GivenNm"],
         "OrdinaryVotes":row["OrdinaryVotes"] + candidate_nonloc_ordinary/og_numPPIDs,
         "AbsentVotes":candidate_absentee/og_numPPIDs,
         "ProvisionalVotes":candidate_provisional/og_numPPIDs, 
         "PrePollVotes":candidate_prepoll/og_numPPIDs,
         "PostalVotes": candidate_postal/og_numPPIDs,
         "TotalVotes": row["OrdinaryVotes"] + (candidate_nonloc_ordinary + candidate_absentee + candidate_provisional + candidate_prepoll + candidate_postal)/og_numPPIDs
         }
    )
        
votes2022allVoteTypes['OrdinaryNoLocVotes'] = 0
for index, row in votes2022[votes2022["PollingPlaceID"].isin(handle_no_location_PPIds(votes2022))].iterrows():
    idx = votes2022allVoteTypes[(votes2022allVoteTypes[["DivisionNm","Surname", "GivenNm"]] == [row['DivisionNm'],row['Surname'], row['GivenNm']]).all(axis=1)].index
    if len(idx)>1:
        print([row['DivisionNm'],row['Surname'], row['GivenNm'],row['PartyAb']])
        print(idx)
        print(row)
        raise Exception("Error multiple candidates detected. Should only be one.")
    elif len(idx)==0:
        idx = votes2022allVoteTypes[(votes2022allVoteTypes[["DivisionNm","Surname"]] == [row['DivisionNm'],row['Surname']]).all(axis=1)].index

    votes2022allVoteTypes.loc[idx,"OrdinaryNoLocVotes"] += row["OrdinaryVotes"]
    votes2022allVoteTypes.loc[idx,"OrdinaryVotes"] = votes2022allVoteTypes.loc[idx,"OrdinaryVotes"] - votes2022allVoteTypes.loc[idx,"OrdinaryNoLocVotes"]
votes2022 = votes2022[~votes2022["PollingPlaceID"].isin(handle_no_location_PPIds(votes2022))]

votes2025 = votes2022.apply(handle_row, axis=1)
teals = pd.read_csv("processed/teals2022.csv")
votes2025['PartyAb_adjusted'] = votes2025['PartyAb']
for index, row in teals.iterrows():
    votes2025.loc[(votes2025[["DivisionNm","Surname","GivenNm"]] == [row['DivisionNm'],row['Surname'], row['GivenNm']]).all(axis=1),"PartyAb_adjusted"] = "TEAL"


for_ind_handling = votes2025.groupby(["DivisionNm","PartyAb_adjusted","Surname","GivenNm"], as_index=False)[['OrdinaryVotes', 'AbsentVotes', 'ProvisionalVotes', 'PrePollVotes',
       'PostalVotes', 'TotalVotes']].sum()


for div in votes2025["DivisionNm"].unique():
    slice_= for_ind_handling[for_ind_handling["DivisionNm"]==div]
    slice_ = slice_[slice_["PartyAb_adjusted"]=="IND"]
    if slice_.shape[0]<=1: # Election with no inds or 1 ind can be ignored. We handle multiple inds here. Only largest ind is considered and smaller inds are assigned as OTH
        continue
        
    for index, row in slice_.sort_values("OrdinaryVotes").iterrows():
        if row['TotalVotes'] == np.max(slice_["TotalVotes"]):
            continue
        surname, given = row["Surname"], row["GivenNm"]
        votes2025.loc[(votes2025[["DivisionNm","Surname","GivenNm"]] == [div,surname,given]).all(axis=1),"PartyAb_"] = "OTH_IND"
raw_post_redistribution = votes2025.pivot_table(index='DivisionNm', columns=['PartyAb_adjusted'], values='OrdinaryVotes', aggfunc='sum')
pct_post_redistribution = raw_post_redistribution.drop("INF", axis=1).div(raw_post_redistribution.drop("INF", axis=1).sum(axis=1), axis=0)
# pct_post_redistribution = raw_post_redistribution.drop('INF',axis=1)
# pct_post_redistribution = pct_post_redistribution.apply(lambda x: x/x.sum())
# pct_post_redistribution['INF'] = raw_post_redistribution['INF']/raw_post_redistribution.sum()

raw_post_redistribution.to_csv("processed/raw_post_redistribution.csv")
pct_post_redistribution.to_csv("processed/pct_post_redistribution.csv")
