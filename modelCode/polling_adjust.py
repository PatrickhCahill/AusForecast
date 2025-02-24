import pandas as pd
import os,sys
import numpy as np
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


polling_estimates = pd.read_csv("processed/polling_estimates2025.csv",index_col="demographic").T
election_estimates= pd.read_csv("processed/election2022estimates.csv",index_col="demographic").fillna(0).T
national_facts = pd.read_csv("processed/pct_national_facts.csv",index_col="demographic").T


demographic_groups = {
    'income_cols' : ['upto50k', '50to99k', '100to150k', '150kplus'],
    'age_cols' : ['18to34', '35to49','50to64', '65plus'],
    'sex_cols' : ['female', 'male'],
    'language_cols' : ['englishonly', 'nonenglish'],
    'religion_cols' : ['christian', 'noreligion'],
    'education_cols' : ['notertiary', 'tafe', 'university']
}
# national_polling = polling_estimates['national'].copy()
# adjusted_polls = polling_estimates.copy()
# adjusted_2022 = election_estimates.copy()

def adjust_to_national(national_polling:pd.Series,polling_estimates:pd.DataFrame, demographic_groups:dict,national_facts:pd.DataFrame)->pd.DataFrame:
    '''Adjust polling averages so that each individual demographic group has the same indicative swing as the national swing.
        i.e if the national swing is -2% and the income group polling suggests a swing of 3% after weighting all income groups equally, shift to uniformly by -5% to make the indicative national swing of the income group polling -2%.

        Keeps the standard deviations terms alone.
    '''
    adjusted_polls = polling_estimates.copy()

    for demographic_group in demographic_groups.keys():
        national_grouped_facts = national_facts[demographic_groups[demographic_group]].iloc[0]
        polling_grouped_facts = polling_estimates[demographic_groups[demographic_group]]
        diff = national_polling - polling_grouped_facts.multiply(national_grouped_facts).sum(axis=1) / national_grouped_facts.sum()
        
        for idx, val in diff.items():
            if "_std" not in idx:
                adjusted_polls.loc[idx,demographic_groups[demographic_group]] = val + adjusted_polls.loc[idx,demographic_groups[demographic_group]]

    return adjusted_polls

adjusted_2022 = adjust_to_national(election_estimates['national'],election_estimates,demographic_groups,national_facts)
adjusted_2025 = adjust_to_national(polling_estimates['national'],polling_estimates,demographic_groups,national_facts)
for group in demographic_groups.keys():
    print(adjusted_2022[demographic_groups[group]])
    print("\n")

print("----- #### -----")

for group in demographic_groups.keys():
    print(adjusted_2025[demographic_groups[group]])
    print("\n")


adjusted_2022.to_csv("processed/adjusted2022.csv")
adjusted_2025.to_csv("processed/adjusted2025.csv")