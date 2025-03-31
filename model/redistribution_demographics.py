import os,sys
import pandas as pd
def sa1_11_to_7_code(code):
    code = str(code)
    assert len(code)==11
    st = code[0]
    sa2=code[5:9]
    sa1=code[9:11]

    out = f"{st}{sa2}{sa1}"
    return int(out)
def aec_sa1_remove_appendix(code):
    code = str(code)
    try:
        if code[-1].isalpha():
            return aec_sa1_remove_appendix(code[:-1])
    except Exception as e:
        print(code)
        raise e
    return int(code) # Throws an error if there is a letter anywhere other than the end.


abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

raw_df_main = pd.DataFrame()
pct_df_main = pd.DataFrame()
raw_df_2022 = pd.DataFrame()
pct_df_2022 = pd.DataFrame()

# Get the demogrpahics for the states with the new maps which are NSW, VIC and WA. We get the data by SA1 and then combine them to form electorate level information
file_tuples = [
    ("NSW","raw/2021_GCP_SA1_for_NSW_short-header/2021 Census GCP Statistical Area 1 for NSW", "raw/NSW-by-SA2-and-SA1.xlsx", "NEW SOUTH WALES"),
    ("VIC","raw/2021_GCP_SA1_for_VIC_short-header/2021 Census GCP Statistical Area 1 for VIC", "raw/Vic-2024-electoral-divisions-SA1-and-SA2.xlsx", "VICTORIA"),
    ("WA","raw/2021_GCP_SA1_for_WA_short-header/2021 Census GCP Statistical Area 1 for WA", "raw/Western Australia - electoral divisions - SA1 and SA2.xlsx", "WESTERN AUSTRALIA"),
    ("NT","raw/2021_GCP_SA1_for_NT_short-header/2021 Census GCP Statistical Area 1 for NT", "raw/Northern-Territory-electoral-divisions-SA1-and-SA2.xlsx", "NORTHERN TERRITORY"),

]


for state_code, demographics_folder, ced_code_name, state_name in file_tuples:
    demographic_db = pd.read_csv(f"{demographics_folder}/2021Census_G01_{state_code}_SA1.csv",index_col="SA1_CODE_2021")

    raw_df = pd.DataFrame(columns=['upto50k','50to99k','100to150k', '150kplus', '18to34', '35to49', '50to64',  '65plus', 'female', 'male','englishonly',  'nonenglish', 'christian', 'noreligion', 'notertiary', 'tafe', 'university'], index=demographic_db.index)
    raw_df['male'] = demographic_db['Tot_P_M']
    raw_df['female'] = demographic_db['Tot_P_F']
    age_db = pd.read_csv(f"{demographics_folder}/2021Census_G04A_{state_code}_SA1.csv",index_col="SA1_CODE_2021")
    adults = demographic_db[['Age_20_24_yr_P',
    'Age_25_34_yr_P',
    'Age_35_44_yr_P',
    'Age_45_54_yr_P',
    'Age_55_64_yr_P',
    'Age_65_74_yr_P',
    'Age_75_84_yr_P',
    'Age_85ov_P']].sum(axis=1) + age_db[['Age_yr_18_P',
    'Age_yr_19_P']].sum(axis=1)
    raw_df['18to34'] = (demographic_db[['Age_20_24_yr_P','Age_25_34_yr_P',]].sum(axis=1) + age_db[['Age_yr_18_P', 'Age_yr_19_P']].sum(axis=1))
    raw_df['35to49'] = (demographic_db[['Age_35_44_yr_P']].sum(axis=1) + age_db[['Age_yr_45_P','Age_yr_46_P','Age_yr_47_P','Age_yr_48_P','Age_yr_49_P']].sum(axis=1))
    raw_df['50to64'] = (demographic_db[['Age_55_64_yr_P']].sum(axis=1) + age_db[['Age_yr_50_P','Age_yr_51_P','Age_yr_52_P','Age_yr_53_P','Age_yr_54_P']].sum(axis=1))
    raw_df['65plus'] = (demographic_db[[ 'Age_65_74_yr_P', 'Age_75_84_yr_P', 'Age_85ov_P']].sum(axis=1))


    income_db = pd.read_csv(f"{demographics_folder}/2021Census_G17A_{state_code}_SA1.csv",index_col="SA1_CODE_2021")
    income_db = income_db.combine_first(pd.read_csv(f"{demographics_folder}/2021Census_G17B_{state_code}_SA1.csv",index_col="SA1_CODE_2021"))
    income_upt50k_cols = ['M_Neg_Nil_income_Tot', 'M_1_149_Tot', 'M_150_299_Tot', 'M_300_399_Tot', 'M_400_499_Tot', 'M_500_649_Tot', 'M_650_799_Tot', 'M_800_999_Tot', 'F_Neg_Nil_income_Tot', 'F_1_149_Tot', 'F_150_299_Tot', 'F_300_399_Tot', 'F_400_499_Tot', 'F_500_649_Tot', 'F_650_799_Tot', 'F_800_999_Tot']
    income_50to99k = ['M_1000_1249_Tot', 'M_1250_1499_Tot', 'M_1500_1749_Tot', 'M_1750_1999_Tot', 'F_1000_1249_Tot', 'F_1250_1499_Tot', 'F_1500_1749_Tot', 'F_1750_1999_Tot',]
    income100to150k = ["M_2000_2999_Tot","F_2000_2999_Tot"]
    income150kplus = [ 'M_3000_3499_Tot', 'M_3500_more_Tot','F_3000_3499_Tot', 'F_3500_more_Tot',]
    raw_df['upto50k'] = income_db[income150kplus].sum(axis=1)
    raw_df['50to99k'] = income_db[income_50to99k].sum(axis=1)
    raw_df['100to150k'] = income_db[income100to150k].sum(axis=1)
    raw_df['150kplus'] = income_db[income150kplus].sum(axis=1)

        
    language_db = pd.read_csv(f"{demographics_folder}/2021Census_G13C_{state_code}_SA1.csv",index_col="SA1_CODE_2021")
    language_db = language_db.combine_first(pd.read_csv(f"{demographics_folder}/2021Census_G13E_{state_code}_SA1.csv",index_col="SA1_CODE_2021"))
    raw_df['englishonly'] = language_db["PSEO_Tot"]
    raw_df['nonenglish'] = language_db["POL_Tot_Tot"]


    religion_db = pd.read_csv(f"{demographics_folder}/2021Census_G14_{state_code}_SA1.csv",index_col="SA1_CODE_2021")
    raw_df['christian'] = religion_db["Christianity_Tot_P"]
    raw_df['noreligion'] = religion_db["SB_OSB_NRA_Tot_P"]


    education_db = pd.read_csv(f"{demographics_folder}/2021Census_G49A_{state_code}_SA1.csv",index_col="SA1_CODE_2021")
    education_db = education_db.combine_first(pd.read_csv(f"{demographics_folder}/2021Census_G49B_{state_code}_SA1.csv",index_col="SA1_CODE_2021"))
    raw_df['university'] = education_db[["P_BachDeg_Total", "P_PGrad_Deg_Total", "P_GradDip_and_GradCert_Total"]].sum(axis=1)
    raw_df['tafe'] = education_db[["P_AdvDip_and_Dip_Total", "P_Cert_Lev_Tot_Total"]].sum(axis=1)
    raw_df['notertiary'] = education_db["P_Tot_Total"] - raw_df['university'] - raw_df['tafe']


    income_cols = ['upto50k', '50to99k', '100to150k', '150kplus']
    age_cols = ['18to34', '35to49','50to64', '65plus',]
    sex_cols = ['female', 'male']
    language_cols = ['englishonly', 'nonenglish',]
    religion_cols = ['christian', 'noreligion']
    education_cols = ['notertiary', 'tafe', 'university']


    raw_df= raw_df.reset_index()
    raw_df["SA1_CODE_2021"] = raw_df["SA1_CODE_2021"].apply(sa1_11_to_7_code)
    raw_df = raw_df.set_index("SA1_CODE_2021")


    electorates_by_sa1 = pd.read_excel(ced_code_name)

    sa1codename = [col for col in electorates_by_sa1.columns if ("sa1" in col.lower())][0]
    divisionNmname = [col for col in electorates_by_sa1.columns if ("new electoral division" in col.lower())][0]
    electorates_by_sa1 = electorates_by_sa1[[sa1codename,divisionNmname]]
    electorates_by_sa1 = electorates_by_sa1.rename({sa1codename:"SA1_CODE_2021",divisionNmname:"DivisionNm"},axis=1)
    electorates_by_sa1 = electorates_by_sa1.drop(electorates_by_sa1[electorates_by_sa1["SA1_CODE_2021"]==state_name].index,axis=0)
    electorates_by_sa1 = electorates_by_sa1.drop(electorates_by_sa1[pd.isna(electorates_by_sa1["SA1_CODE_2021"])].index,axis=0)

    electorates_by_sa1['SA1_CODE_2021'] = electorates_by_sa1['SA1_CODE_2021'].apply(aec_sa1_remove_appendix)

    out_df = pd.DataFrame(columns=raw_df.columns,index=electorates_by_sa1['DivisionNm'].unique())
    for divisionNm in electorates_by_sa1['DivisionNm'].unique():
        sa1s_to_add = electorates_by_sa1[electorates_by_sa1['DivisionNm']==divisionNm]['SA1_CODE_2021']
        sa1s_to_add_proper = sa1s_to_add[sa1s_to_add.isin(raw_df.index)]
        if len(sa1s_to_add_proper) != len(sa1s_to_add):
            print(f"WARNING: {divisionNm} has {len(sa1s_to_add)-len(sa1s_to_add_proper)} SA1s that are not in the raw file")
        out_df.loc[divisionNm]=raw_df.loc[sa1s_to_add_proper].sum(axis=0)

    raw_df = out_df.copy()
    out_df = None

    pct_df = raw_df.copy()
    for col_set in [income_cols,age_cols,sex_cols,language_cols,religion_cols,education_cols]:
        pct_df[col_set] = raw_df[col_set].div(raw_df[col_set].sum(axis=1),axis=0)

 
    raw_df_main = pd.concat([raw_df_main,raw_df],axis=0)
    pct_df_main = pd.concat([pct_df_main,pct_df],axis=0)


# Get the demogrpahics for all the 2021 electoral divisions using the division level data. This is is valid in the states that didn't have a redistibution
demographics_folder = "raw/2021_GCP_CED_for_AUS_short-header/2021 Census GCP Commonwealth Electroral Division for AUS"
demographic_db = pd.read_csv(f"{demographics_folder}/2021Census_G01_AUST_CED.csv",index_col="CED_CODE_2021")

raw_df = pd.DataFrame(columns=['upto50k','50to99k','100to150k', '150kplus', '18to34', '35to49', '50to64',  '65plus', 'female', 'male','englishonly',  'nonenglish', 'christian', 'noreligion', 'notertiary', 'tafe', 'university'], index=demographic_db.index)
raw_df['male'] = demographic_db['Tot_P_M']
raw_df['female'] = demographic_db['Tot_P_F']
age_db = pd.read_csv(f"{demographics_folder}/2021Census_G04A_AUST_CED.csv",index_col="CED_CODE_2021")
adults = demographic_db[['Age_20_24_yr_P',
'Age_25_34_yr_P',
'Age_35_44_yr_P',
'Age_45_54_yr_P',
'Age_55_64_yr_P',
'Age_65_74_yr_P',
'Age_75_84_yr_P',
'Age_85ov_P']].sum(axis=1) + age_db[['Age_yr_18_P',
'Age_yr_19_P']].sum(axis=1)
raw_df['18to34'] = (demographic_db[['Age_20_24_yr_P','Age_25_34_yr_P',]].sum(axis=1) + age_db[['Age_yr_18_P', 'Age_yr_19_P']].sum(axis=1))
raw_df['35to49'] = (demographic_db[['Age_35_44_yr_P']].sum(axis=1) + age_db[['Age_yr_45_P','Age_yr_46_P','Age_yr_47_P','Age_yr_48_P','Age_yr_49_P']].sum(axis=1))
raw_df['50to64'] = (demographic_db[['Age_55_64_yr_P']].sum(axis=1) + age_db[['Age_yr_50_P','Age_yr_51_P','Age_yr_52_P','Age_yr_53_P','Age_yr_54_P']].sum(axis=1))
raw_df['65plus'] = (demographic_db[[ 'Age_65_74_yr_P', 'Age_75_84_yr_P', 'Age_85ov_P']].sum(axis=1))


income_db = pd.read_csv(f"{demographics_folder}/2021Census_G17A_AUST_CED.csv",index_col="CED_CODE_2021")
income_db = income_db.combine_first(pd.read_csv(f"{demographics_folder}/2021Census_G17B_AUST_CED.csv",index_col="CED_CODE_2021"))
income_upt50k_cols = ['M_Neg_Nil_income_Tot', 'M_1_149_Tot', 'M_150_299_Tot', 'M_300_399_Tot', 'M_400_499_Tot', 'M_500_649_Tot', 'M_650_799_Tot', 'M_800_999_Tot', 'F_Neg_Nil_income_Tot', 'F_1_149_Tot', 'F_150_299_Tot', 'F_300_399_Tot', 'F_400_499_Tot', 'F_500_649_Tot', 'F_650_799_Tot', 'F_800_999_Tot']
income_50to99k = ['M_1000_1249_Tot', 'M_1250_1499_Tot', 'M_1500_1749_Tot', 'M_1750_1999_Tot', 'F_1000_1249_Tot', 'F_1250_1499_Tot', 'F_1500_1749_Tot', 'F_1750_1999_Tot',]
income100to150k = ["M_2000_2999_Tot","F_2000_2999_Tot"]
income150kplus = [ 'M_3000_3499_Tot', 'M_3500_more_Tot','F_3000_3499_Tot', 'F_3500_more_Tot',]
raw_df['upto50k'] = income_db[income150kplus].sum(axis=1)
raw_df['50to99k'] = income_db[income_50to99k].sum(axis=1)
raw_df['100to150k'] = income_db[income100to150k].sum(axis=1)
raw_df['150kplus'] = income_db[income150kplus].sum(axis=1)

    
language_db = pd.read_csv(f"{demographics_folder}/2021Census_G13C_AUST_CED.csv",index_col="CED_CODE_2021")
language_db = language_db.combine_first(pd.read_csv(f"{demographics_folder}/2021Census_G13E_AUST_CED.csv",index_col="CED_CODE_2021"))
raw_df['englishonly'] = language_db["PSEO_Tot"]
raw_df['nonenglish'] = language_db["POL_Tot_Tot"]


religion_db = pd.read_csv(f"{demographics_folder}/2021Census_G14_AUST_CED.csv",index_col="CED_CODE_2021")
raw_df['christian'] = religion_db["Christianity_Tot_P"]
raw_df['noreligion'] = religion_db["SB_OSB_NRA_Tot_P"]


education_db = pd.read_csv(f"{demographics_folder}/2021Census_G49A_AUST_CED.csv",index_col="CED_CODE_2021")
education_db = education_db.combine_first(pd.read_csv(f"{demographics_folder}/2021Census_G49B_AUST_CED.csv",index_col="CED_CODE_2021"))
raw_df['university'] = education_db[["P_BachDeg_Total", "P_PGrad_Deg_Total", "P_GradDip_and_GradCert_Total"]].sum(axis=1)
raw_df['tafe'] = education_db[["P_AdvDip_and_Dip_Total", "P_Cert_Lev_Tot_Total"]].sum(axis=1)
raw_df['notertiary'] = education_db["P_Tot_Total"] - raw_df['university'] - raw_df['tafe']


income_cols = ['upto50k', '50to99k', '100to150k', '150kplus']
age_cols = ['18to34', '35to49','50to64', '65plus',]
sex_cols = ['female', 'male']
language_cols = ['englishonly', 'nonenglish',]
religion_cols = ['christian', 'noreligion']
education_cols = ['notertiary', 'tafe', 'university']

raw_df['DivisionNm'] = pd.read_csv("raw/CEDcodemapper.csv",index_col = "CED_CODE_2021")
raw_df = raw_df.drop(raw_df[pd.isna(raw_df['DivisionNm'])].index)

raw_df_2022 = raw_df.copy().set_index("DivisionNm")

# Now need to remove NSW, VIC, WA electorates from the CED level data as they have already been determined from the SA1 process.
idx_to_keep = []
for idx, DivisionNm in raw_df['DivisionNm'].items():
    if DivisionNm in ["Higgins", "North Sydney"]:
        continue
    elif DivisionNm.lower() in [mystr.lower() for mystr in raw_df_main.index]:
        continue
    else:
        idx_to_keep.append(idx)

raw_df = raw_df.loc[idx_to_keep].set_index('DivisionNm')
pct_df = raw_df.copy()
for col_set in [income_cols,age_cols,sex_cols,language_cols,religion_cols,education_cols]:
    pct_df[col_set] = raw_df[col_set].div(raw_df[col_set].sum(axis=1),axis=0)
    pct_df_2022[col_set] = raw_df_2022[col_set].div(raw_df_2022[col_set].sum(axis=1),axis=0)

# Combine to final dfs
raw_df_main = pd.concat([raw_df_main,raw_df],axis=0).sort_index()
raw_df_main.index.name = 'DivisionNm'
pct_df_main = pd.concat([pct_df_main,pct_df],axis=0).sort_index()
pct_df_main.index.name = 'DivisionNm'


raw_df_main.to_csv("processed/raw_2025_demographic_facts.csv")
pct_df_main.to_csv("processed/pct_2025_demographic_facts.csv")
raw_df_2022.to_csv("processed/raw_2022_demographic_facts.csv")
pct_df_2022.to_csv("processed/pct_2022_demographic_facts.csv")



######
demographics_folder = "raw/2021_GCP_CED_for_AUS_short-header/2021 Census GCP Commonwealth Electroral Division for AUS"
demographic_db = pd.read_csv(f"{demographics_folder}/2021Census_G01_AUST_CED.csv",index_col="CED_CODE_2021")
cedcodemapper = pd.read_csv("raw/CEDcodemapper.csv",index_col="CED_CODE_2021")
demographic_db.head()
pct_df = pd.DataFrame(columns=['upto50k','50to99k','100to150k', '150kplus', '18to34', '35to49', '50to64',  '65plus', 'female', 'male','englishonly',  'nonenglish', 'christian', 'noreligion', 'notertiary', 'tafe', 'university'], index=demographic_db.index)
raw_df = pd.DataFrame(columns=['upto50k','50to99k','100to150k', '150kplus', '18to34', '35to49', '50to64',  '65plus', 'female', 'male','englishonly',  'nonenglish', 'christian', 'noreligion', 'notertiary', 'tafe', 'university'], index=demographic_db.index)
pct_df['male'] = demographic_db['Tot_P_M']/demographic_db['Tot_P_P']
pct_df['female'] = demographic_db['Tot_P_F']/demographic_db['Tot_P_P']
raw_df['male'] = demographic_db['Tot_P_M']
raw_df['female'] = demographic_db['Tot_P_F']
age_db = pd.read_csv(f"{demographics_folder}/2021Census_G04A_AUST_CED.csv",index_col="CED_CODE_2021")
age_db.head()
adults = demographic_db[['Age_20_24_yr_P',
 'Age_25_34_yr_P',
 'Age_35_44_yr_P',
 'Age_45_54_yr_P',
 'Age_55_64_yr_P',
 'Age_65_74_yr_P',
 'Age_75_84_yr_P',
 'Age_85ov_P']].sum(axis=1) + age_db[['Age_yr_18_P',
 'Age_yr_19_P']].sum(axis=1)


pct_df['18to34'] = (demographic_db[['Age_20_24_yr_P','Age_25_34_yr_P',]].sum(axis=1) + age_db[['Age_yr_18_P', 'Age_yr_19_P']].sum(axis=1))/adults
pct_df['35to49'] = (demographic_db[['Age_35_44_yr_P']].sum(axis=1) + age_db[['Age_yr_45_P','Age_yr_46_P','Age_yr_47_P','Age_yr_48_P','Age_yr_49_P']].sum(axis=1))/adults
pct_df['50to64'] = (demographic_db[['Age_55_64_yr_P']].sum(axis=1) + age_db[['Age_yr_50_P','Age_yr_51_P','Age_yr_52_P','Age_yr_53_P','Age_yr_54_P']].sum(axis=1))/adults
pct_df['65plus'] = (demographic_db[[ 'Age_65_74_yr_P', 'Age_75_84_yr_P', 'Age_85ov_P']].sum(axis=1))/adults


raw_df['18to34'] = (demographic_db[['Age_20_24_yr_P','Age_25_34_yr_P',]].sum(axis=1) + age_db[['Age_yr_18_P', 'Age_yr_19_P']].sum(axis=1))
raw_df['35to49'] = (demographic_db[['Age_35_44_yr_P']].sum(axis=1) + age_db[['Age_yr_45_P','Age_yr_46_P','Age_yr_47_P','Age_yr_48_P','Age_yr_49_P']].sum(axis=1))
raw_df['50to64'] = (demographic_db[['Age_55_64_yr_P']].sum(axis=1) + age_db[['Age_yr_50_P','Age_yr_51_P','Age_yr_52_P','Age_yr_53_P','Age_yr_54_P']].sum(axis=1))
raw_df['65plus'] = (demographic_db[[ 'Age_65_74_yr_P', 'Age_75_84_yr_P', 'Age_85ov_P']].sum(axis=1))

pct_df
income_db = pd.read_csv(f"{demographics_folder}/2021Census_G17A_AUST_CED.csv",index_col="CED_CODE_2021")
income_db = income_db.combine_first(pd.read_csv(f"{demographics_folder}/2021Census_G17B_AUST_CED.csv",index_col="CED_CODE_2021"))
income_db.head()
'100to150k', '150kplus','50to99k', 'upto50k'
income_upt50k_cols = ['M_Neg_Nil_income_Tot', 'M_1_149_Tot', 'M_150_299_Tot', 'M_300_399_Tot', 'M_400_499_Tot', 'M_500_649_Tot', 'M_650_799_Tot', 'M_800_999_Tot', 'F_Neg_Nil_income_Tot', 'F_1_149_Tot', 'F_150_299_Tot', 'F_300_399_Tot', 'F_400_499_Tot', 'F_500_649_Tot', 'F_650_799_Tot', 'F_800_999_Tot']
income_50to99k = ['M_1000_1249_Tot', 'M_1250_1499_Tot', 'M_1500_1749_Tot', 'M_1750_1999_Tot', 'F_1000_1249_Tot', 'F_1250_1499_Tot', 'F_1500_1749_Tot', 'F_1750_1999_Tot',]
income100to150k = ["M_2000_2999_Tot","F_2000_2999_Tot"]
income150kplus = [ 'M_3000_3499_Tot', 'M_3500_more_Tot','F_3000_3499_Tot', 'F_3500_more_Tot',]

raw_df['upto50k'] = income_db[income150kplus].sum(axis=1)
raw_df['50to99k'] = income_db[income_50to99k].sum(axis=1)
raw_df['100to150k'] = income_db[income100to150k].sum(axis=1)
raw_df['150kplus'] = income_db[income150kplus].sum(axis=1)

pct_df['upto50k'] = income_db[income150kplus].sum(axis=1) / raw_df[['100to150k', '150kplus','50to99k', 'upto50k']].sum(axis=1)
pct_df['50to99k'] = income_db[income_50to99k].sum(axis=1) / raw_df[['100to150k', '150kplus','50to99k', 'upto50k']].sum(axis=1)
pct_df['100to150k'] = income_db[income100to150k].sum(axis=1) / raw_df[['100to150k', '150kplus','50to99k', 'upto50k']].sum(axis=1)
pct_df['150kplus'] = income_db[income150kplus].sum(axis=1) / raw_df[['100to150k', '150kplus','50to99k', 'upto50k']].sum(axis=1)

    


language_db = pd.read_csv(f"{demographics_folder}/2021Census_G13C_AUST_CED.csv",index_col="CED_CODE_2021")
language_db = language_db.combine_first(pd.read_csv(f"{demographics_folder}/2021Census_G13E_AUST_CED.csv",index_col="CED_CODE_2021"))
language_db.head()

raw_df['englishonly'] = language_db["PSEO_Tot"]
raw_df['nonenglish'] = language_db["POL_Tot_Tot"]

pct_df['englishonly'] = language_db["PSEO_Tot"] / raw_df[['englishonly','nonenglish']].sum(axis=1)
pct_df['nonenglish'] = language_db["POL_Tot_Tot"] / raw_df[['englishonly','nonenglish']].sum(axis=1)
religion_db = pd.read_csv(f"{demographics_folder}/2021Census_G14_AUST_CED.csv",index_col="CED_CODE_2021")
religion_db.head()
raw_df['christian'] = religion_db["Christianity_Tot_P"]
raw_df['noreligion'] = religion_db["SB_OSB_NRA_Tot_P"]

pct_df['christian'] = religion_db["Christianity_Tot_P"] / (raw_df[['christian','noreligion']].sum(axis=1))
pct_df['noreligion'] = religion_db["SB_OSB_NRA_Tot_P"] / (raw_df[['christian','noreligion']].sum(axis=1))
education_db = pd.read_csv(f"{demographics_folder}/2021Census_G49A_AUST_CED.csv",index_col="CED_CODE_2021")
education_db = education_db.combine_first(pd.read_csv(f"{demographics_folder}/2021Census_G49B_AUST_CED.csv",index_col="CED_CODE_2021"))
education_db.head()
sa1s = pd.read_csv("raw/2021_GCP_SA1_for_NSW_short-header/2021 Census GCP Statistical Area 1 for NSW/2021Census_G60B_NSW_SA1.csv")["SA1_CODE_2021"]
raw_df['university'] = education_db[["P_BachDeg_Total", "P_PGrad_Deg_Total", "P_GradDip_and_GradCert_Total"]].sum(axis=1)
raw_df['tafe'] = education_db[["P_AdvDip_and_Dip_Total", "P_Cert_Lev_Tot_Total"]].sum(axis=1)
raw_df['notertiary'] = education_db["P_Tot_Total"] - raw_df['university'] - raw_df['tafe']


pct_df['university'] =  raw_df['university'] / raw_df[['university','tafe','notertiary']].sum(axis=1)
pct_df['tafe'] =  raw_df['tafe'] / raw_df[['university','tafe','notertiary']].sum(axis=1)
pct_df['notertiary'] =  raw_df['notertiary'] / raw_df[['university','tafe','notertiary']].sum(axis=1)

income_cols = ['upto50k', '50to99k', '100to150k', '150kplus']
age_cols = ['18to34', '35to49','50to64', '65plus',]
sex_cols = ['female', 'male']
language_cols = ['englishonly', 'nonenglish',]
religion_cols = ['christian', 'noreligion']
education_cols = ['notertiary', 'tafe', 'university']
raw_df['DivisionNm'] = cedcodemapper['DivisionNm']
national_facts = raw_df.set_index('DivisionNm').sum(axis=0)
national_facts.name="raw"
national_facts.index.name="demographic"
national_facts.to_csv("processed/raw_national_facts.csv")
national_facts = national_facts.astype(float)
for col_set in [income_cols,age_cols,sex_cols,language_cols,religion_cols,education_cols]:
    national_facts[col_set] =national_facts[col_set] / national_facts[col_set].sum()


national_facts.name="pct"
national_facts.index.name="demographic"
national_facts.to_csv("processed/pct_national_facts.csv")


