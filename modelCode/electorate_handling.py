'''
A file that handles manipulating the 2022 election results into a form that is useable for the model.

For every seat, we have a record of the actual 2022 result and up to date data on the candidates contesting each seat at this election.

We then process this combined with our estimate of the preference matrix, M (see M_data_transform.py, M_generate.py).

In essennce, we implement the following process:

For a given seat in the 2025 election (i.e excluding Higgins, North Sydney and including Bullwinkel):
    - We estimate the election result in 2022 after redistribution.
        - If the seat is outside of WA, NSW, VIC, NT - then the result is the same.
        - If the seat is in WA, NSW, VIC, NT:
            - We approximate the election result by including the results by polling booth if that polling booth is in the new electorate.
            - We then assume that the non-Ordinary votes are distributed equally across each polling booth. That is
                - If polling place p was in electorate i and is now in electorate j, we assume that electorate j receives non-ordianry votes
                "from polling place p" which is non-ordinary votes in electorate i divided by the total number of polling places in electorate i.
                - This has the effect that if i is unchanged then our predicition of i is also unchanged.
                - Note we do not attempt to model changes in the primary vote due to parties competing differently.
                    - In particular, see Antony Green's https://antonygreen.com.au/fed25-a-plethora-of-estimated-margins-and-problems-with-the-aec-version/ 
                    article on predicting teal independent primary votes after redististributions for areas
                    where a teal did not compete in 2022.

    - This serves as our post-redistribution estimate of the election result in 2022. We now turn to the challenge of changing candidates.
        - If a candidate competed in 2022 and is not competing in 2025, we simply distributed their preferences to the other candidates according to our estimate of M.
        - If a candidate is competing in 2025 who did not compete in 2022 we estimate their support from polling in the 2022 election.
            - We then create a "pseudo_vote_2022" which is all the other parties primary votes who actually competed plus the new candidate.
            - We then distribute the preferences of the new candidate. We calculate the difference between the pseudo_vote_2022_eliminated and the actual vote.
            - Then sum of this diff vector is equal to the support we predict for the candidate. We then subtract that from "pseudo_vote_2022" which gives our estimate of the 2022 vote given the new candidate.

        - Repeat both of the above processes (replacing the actual_vote_2022 with the most up to date prediction) until all 2022 non-competing candidates are excluded and all 2025 candidates are included.
        - This yields our initial estimate for electorate result.
        i.e if nothing changed amongst voters since 2022 and the only changes were in boundaries and competing candidates, what would our prediction be.

        - Our model then attempts to model the swing from this prediction.`.lo
'''

import pandas as pd
import geopandas as gpd
import numpy as np
import os,sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_actual_primary_votes(path="./raw/primaries"):
    primaries_df = pd.DataFrame()
    for file in os.listdir(path):
        if file.endswith(".csv"):
            primaries_df = pd.concat([primaries_df,pd.read_csv(f"{path}/{file}",header=1)],axis=0)
    return primaries_df

print(get_actual_primary_votes())