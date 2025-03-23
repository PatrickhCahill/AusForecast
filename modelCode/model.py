# Take polling estimates of each variable. This is handled in polling.py
# These estimates come in the form of a mean and a confidence interval/std
# The estimates are the swing from the election outcome in 2022.
# We all have the correlation matrix of each of these variables.
# Finally in each seat, we have the demographic breakdown according to the ABS.

# Then we sample from a distribution given by polling means and a covariance matrix created from the regression matrix and uncertainty estimates in the polling.
# This then generates a set of primary votes swings in each seat and a set of TPP swings (Think about doing some raw vote handling but ignorable for now. Use population data from ABS)
# Sum the previous election primary vote and the swing to get the estimate of outcome.
# Apply M on the primary votes and record the final pair.
# If outcome is tpp use tpp swing estimates and previous election flows to estimate TCP.
# If outcome is non-TCP use previous election flows and M to predict TCP.
import pandas as pd
import os,sys
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

seat_demographics = pd.read_csv("processed/pct_demographic_facts.csv",index_col="Unnamed: 0")
seat_demographics.index.name = "DivisionNm"

polling_estimates = pd.read_csv("processed/polling_estimates2025.csv",index_col="Unnamed: 0")
polling_estimates.index.name = "Party"
polling_swings = pd.read_csv("processed/polling_estimates2025swings.csv",index_col="Unnamed: 0")
polling_estimates.index.name = "Party"


polling_mean = polling_estimates.loc[["ALP", "LNC", "GRN", "PHON", "OTH", "ALP 2pp", "L-NP 2pp"]].copy()
polling_std = polling_estimates.loc[["ALP_std", "LNC_std", "GRN_std", "PHON_std", "OTH_std", "ALP 2pp_std", "L-NP 2pp_std"]].copy()
polling_std.index = [idx[:-4] for idx in polling_std.index]



state_df = pd.read_csv("processed/division_by_state.csv",index_col="DivisionNm")
state_df.index = [idx.lower() for idx in state_df.index]
demographic_groups = {
    'income_cols' : ['upto50k', '50to99k', '100to150k', '150kplus'],
    'age_cols' : ['18to34', '35to49','50to64', '65plus'],
    'sex_cols' : ['female', 'male'],
    'language_cols' : ['englishonly', 'nonenglish'],
    'religion_cols' : ['christian', 'noreligion'],
    'education_cols' : ['notertiary', 'tafe', 'university']
}

non_grouped_values = ["national","nsw","qld","sa","tas","victoria","wa"]



def get_seat_swing(division, polling_mean, polling_std):
    indicative_swings = pd.DataFrame(index=polling_mean.index)
    weights = pd.DataFrame(index=polling_mean.index)
    for key in demographic_groups:
        indicative_swings[key] = (polling_mean[demographic_groups[key]].multiply(division[demographic_groups[key]],axis=1).sum(axis=1))
        weights[key] = 1 / ((polling_std[demographic_groups[key]]**2).multiply(division[demographic_groups[key]],axis=1).sum(axis=1))

    indicative_swings['national']  = (polling_mean['national'])
    # if state_df.loc[division.name.lower(),"StateAb"].lower() in non_grouped_values:
    #     indicative_swings['state'] = polling_mean[state_df.loc[division.name.lower(),"StateAb"].lower()]
    # weights['national']  = 1 / ((polling_std['national']*100)**2)
    # if state_df.loc[division.name.lower(),"StateAb"].lower() in non_grouped_values:
    #     weights['state'] = 1 / ((polling_std[state_df.loc[division.name.lower(),"StateAb"].lower()]*100)**2 )

    out_value = pd.Series(index=polling_mean.index)
    out_std = pd.Series(index=polling_mean.index)
    for idx, row in indicative_swings.iterrows():
        out_value[idx] = (row * weights.loc[idx]).sum() / weights.loc[idx].sum()
        out_std[idx] = 1 / np.sqrt(weights.loc[idx].sum() )

    return out_value, out_std


seat_swings = pd.DataFrame(index=seat_demographics.index,columns=polling_mean.index)
seat_swings_std = pd.DataFrame(index=seat_demographics.index,columns=polling_mean.index)

for division in seat_demographics.index:
    swing, std = get_seat_swing(seat_demographics.loc[division],polling_mean,polling_std)
    seat_swings.loc[division] = swing
    seat_swings_std.loc[division] = std

seat_swings.to_csv("processed/seat_swings.csv")
seat_swings_std.to_csv("processed/seat_swings_std.csv")
