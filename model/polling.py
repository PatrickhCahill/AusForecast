import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
from tqdm import tqdm

## SHIFT DEMOGRAPHIC DATA SWINGS TO MATCH NATIONAL SWING.

## ALSO NEED TO DEVELOP A CORRELATION MATRIX BETWEEN THE SWINGS IN THE VARIABLES -- COULD BE HIGH?
        ## THIS IS MOSTLY TO AVOID WASHOUTS IN THE SWING ESTIMATES.


def fixtpp(row): 
    outrow = row.copy()
    if not np.isnan(row["UND r/a"]):
        outrow["ALP r/a"] = row["ALP r/a"]/(row["ALP r/a"] + row["L-NP r/a"])*100
        outrow["L-NP r/a"] = row["L-NP r/a"]/(row["ALP r/a"] + row["L-NP r/a"])*100


    if np.isnan(row["ALP 2pp"]) and not np.isnan(outrow["ALP r/a"]):
        outrow["ALP 2pp"] = outrow["ALP r/a"]
    elif not np.isnan(row["ALP 2pp"]):
        outrow["ALP 2pp"] = outrow["ALP 2pp"]
    if np.isnan(row["L-NP 2pp"]) and not np.isnan(outrow["L-NP r/a"]):
        outrow["L-NP 2pp"] = outrow["L-NP r/a"]
    elif not np.isnan(row["L-NP 2pp"]):
        outrow["L-NP 2pp"] = outrow["L-NP 2pp"]

    outrow[parties] = outrow[parties]/100
    return outrow

def fixtpp2022(row): 
    outrow = row.copy()
    # if not np.isnan(row["UND r/a"]):
    #     outrow["ALP r/a"] = row["ALP r/a"]/(row["ALP r/a"] + row["L-NP r/a"])*100
    #     outrow["L-NP r/a"] = row["L-NP r/a"]/(row["ALP r/a"] + row["L-NP r/a"])*100


    if np.isnan(row["ALP 2pp"]) and not np.isnan(outrow["ALP r/a"]):
        outrow["ALP 2pp"] = row["ALP r/a"]/(row["ALP r/a"] + row["L-NP r/a"])*100
    elif not np.isnan(row["ALP 2pp"]):
        outrow["ALP 2pp"] = outrow["ALP 2pp"]
    if np.isnan(row["L-NP 2pp"]) and not np.isnan(outrow["L-NP r/a"]):
        outrow["L-NP 2pp"] = row["L-NP 2pp"]/(row["ALP r/a"] + row["L-NP r/a"])*100
    elif not np.isnan(row["L-NP 2pp"]):
        outrow["L-NP 2pp"] = outrow["L-NP 2pp"]

    outrow[parties] = outrow[parties]/100
    return outrow

def kalman_drive(xold,Pold,F,Q):
    xnew = F @ xold
    Pnew = F @ Pold @ F.T + Q
    return xnew, Pnew

def kalman_update(y,xprior:np.array, Pprior:np.array,H:np.array,R):
    yhat_prior = H @ xprior
    residual = y - yhat_prior
    innovation = H @ Pprior @ (H.T) + R
    K = Pprior @ H.T @ np.linalg.inv(innovation)

    xpost = xprior + K @ residual
    kh = K @ H

    Ppost = (np.eye(kh.shape[0]) - kh) @ Pprior
    
    yhat_post = H @ xpost
    if np.abs(yhat_prior-yhat_post)>1:
        print(y)
        print(xprior)
        print(innovation)
        print(P @ H.T)
        print(R)
        raise Exception("AGH")
    return xpost,Ppost,yhat_post

def kalman_run(timestamps,ys,Rs,x0,P0,F,Q,H):
    if np.any(timestamps<np.datetime64("2022-05-22")):
        start_date = np.datetime64("2019-05-18")
    else:
        start_date=np.datetime64('2022-05-22')
    x = x0.copy()
    P = P0.copy()

    Yhat_history = [(H @ x).item()]
    Xhistory = [x]
    Phistory = [ P]

    for date in pd.date_range(start_date,np.datetime64('2025-05-22'), freq='D')[1:]:
        x, P = kalman_drive(x,P,F,Q)
        
        yhat_post = None
        obs_indices= np.where(np.datetime64(date.date()) == timestamps.astype('datetime64[D]'))[0] # np.where(timestamps == date.date())[0]
        if obs_indices.shape[0]>0:
            for obs_idx in obs_indices:
                y = ys[obs_idx]
                R = Rs[obs_idx]

                x,P,yhat_post = kalman_update(y,x,P,H,R)
        if yhat_post is None:
            yhat_post = H @ x
        
        Xhistory.append(x.copy())
        Yhat_history.append(yhat_post.item())
        Phistory.append(P)

    ts = pd.date_range(start_date,np.datetime64('2025-05-22'))
    return ts, Yhat_history,Xhistory,Phistory

def estimate_Q(X:list,F:np.array):
    Xforward = [F @ x for x in X]
    residuals = [X[i+1] - Xplus for i,Xplus in enumerate(Xforward[:-1])]
    Qest = sum(residual @ residual.T for residual in residuals)/len(residuals)
    return Qest

def compute_y_std(row):
    x_values = np.array([row[x] for x in parties_modelled])
    y_values = np.array([row[f"{x}_std"] for x in parties_modelled])

    return np.mean(y_values) / np.mean(np.abs(x_values)) * np.abs(row['OTH'])

def normalise(row):
    outrow = row.copy()
    primary_parties = ["ALP","LNC","GRN","PHON","OTH"]
    outrow[primary_parties] = outrow[primary_parties] / outrow[primary_parties].sum()
    outrow[["ALP 2pp","L-NP 2pp"]] = outrow[["ALP 2pp","L-NP 2pp"]] / outrow[["ALP 2pp","L-NP 2pp"]].sum()
    return outrow

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    max_iter_EM = 200

    polling_data_states = ['national.csv', 'nsw.csv', 'qld.csv', 'sa.csv', 'tas.csv', 'victoria.csv', 'wa.csv']
    polling_files_demographics = ['100to150k.csv', '150kplus.csv', '18to34.csv', '35to49.csv', '50to64.csv', '50to99k.csv', '65plus.csv', 'christian.csv', 'englishonly.csv', 'female.csv', 'male.csv', 'nonenglish.csv', 'noreligion.csv', 'notertiary.csv', 'tafe.csv', 'university.csv', 'upto50k.csv', ]
    

    parties = ['ALP', 'LNC', 'GRN', 'PHON', 'UND', 'ALP 2pp', 'L-NP 2pp']
    parties_modelled = ['ALP', 'LNC', 'GRN', 'PHON']
    swings2025_df = pd.DataFrame(index=
                        [os.path.splitext(filename)[0] for filename in (polling_data_states+polling_files_demographics)])
    election2025_df = pd.DataFrame(index=
                        [os.path.splitext(filename)[0] for filename in (polling_data_states+polling_files_demographics)])
    

    election2022_df = pd.DataFrame(index=
                            [os.path.splitext(filename)[0] for filename in (polling_data_states+polling_files_demographics)],columns=parties)

    

    for file in tqdm(polling_data_states):
        # if file != "female.csv":
        #     continue
        polling_data = pd.read_csv(f"polling_data/{file}").apply(fixtpp,axis=1)
        election_day = polling_data[polling_data["Pollster"]=="Election"].iloc[0].copy()
        election2022_df.loc[os.path.splitext(file)[0]] = election_day
        polling_data = polling_data.drop(0,axis=0) # Drop the election day data
        polling_data = polling_data.reset_index()



        for party in tqdm(parties,leave=False):
            # if party != "ALP 2pp":
                # continue
            timestamps = pd.to_datetime(polling_data.loc[~polling_data[party].isna(),"End Date"]).to_numpy()
            ys = (polling_data.loc[~polling_data[party].isna(),party].to_numpy() - election_day[party])
            yno_swing = polling_data.loc[~polling_data[party].isna(),party].to_numpy()
            sample_size = polling_data.loc[~polling_data[party].isna(),"Sample"].replace({',': ''}, regex=True).astype(int).to_numpy()
            Rs = yno_swing * (1-yno_swing) / sample_size
        
            x0 = np.array([0,0]).reshape(-1,1) # [local value, local trend]
            F = np.array([[1,1], # u_t+1 = u_t + delta_t (+ noise)
                        [0,1]]) # delta_t+1 = delta_t (+ noise)
            # F = np.array([[1,1,0, 0, 0, 0],
            #               [0,1,0,-1,-1,-1],
            #               [0,0,-1,0,-1,-1],
            #               [0,0,-1,-1,0,-1],
            #               [0,0,-1,-1,-1,0]]) ## --- This is for a seasonal model ##
            
            H = np.array([1,0]).reshape(1,-1)

            Q = np.diag([0.0005**2,0.00005**2])
            for _ in tqdm(range(max_iter_EM),leave=False):

                ts, Yhat_history,Xhistory,Phistory = kalman_run(timestamps,ys,Rs,x0,Q,F,Q,H)
                Qprev = Q.copy()
                Q = estimate_Q(Xhistory,F)
                if np.linalg.norm(Q-Qprev)<1e-8:
                    break
            
                
            Xadjusted = []
            Padjusted = []
            for idx,(X,P) in enumerate([i for i in zip(Xhistory,Phistory)][::-1]):
                if idx ==0:
                    Xadjusted.append(X)
                    Padjusted.append(P)
                else:
                    Xgiven_n = Xadjusted[-1]
                    Pgiven_n = Padjusted[-1]

                    xgiven_idx = F @ X
                    Pgiven_idx = F @ P @ F.T + Q
                    try:
                        Cidx = P @ F.T @ np.linalg.inv(Pgiven_idx)
                    except np.linalg.LinAlgError as e:
                        Cidx = P @ F.T @ np.linalg.pinv(Pgiven_idx)

                    Xadjusted.append(X + Cidx @ (Xgiven_n - xgiven_idx))
                    Padjusted.append(P + Cidx @ (Pgiven_n - Pgiven_idx) @ Cidx.T)

            Yadjusted = [(H @ Xadj).item() for Xadj in Xadjusted][::-1]
            Sadjusted = [(H @ Padj @ H.T).item() for Padj in Padjusted][::-1]


            swings2025_df.loc[os.path.splitext(file)[0],party]=Yadjusted[-1]
            swings2025_df.loc[os.path.splitext(file)[0],f"{party}_std"]=np.sqrt(Sadjusted[-1])
    
            election2025_df.loc[os.path.splitext(file)[0],party]=Yadjusted[-1]+ election_day[party]
            election2025_df.loc[os.path.splitext(file)[0],f"{party}_std"]=np.sqrt(Sadjusted[-1])
    for file in tqdm(polling_files_demographics):
        # if file != "noreligion.csv":
        #     continue
        polling_data = pd.read_csv(f"polling_data/{file}").apply(fixtpp,axis=1)
        polling_data_2022 = pd.read_csv(f"polling_data2022/{file}").apply(fixtpp2022,axis=1)
        polling_data = pd.concat([polling_data_2022,polling_data],axis=0)
        polling_data = polling_data.reset_index()

        for party in tqdm(parties,leave=False):
            # if party != "ALP 2pp":
            #     continue
            timestamps = pd.to_datetime(polling_data.loc[~polling_data[party].isna(),"End Date"],format="%b %d, %Y").to_numpy()
            ys = polling_data.loc[~polling_data[party].isna(),party].to_numpy()
            sample_size = polling_data.loc[~polling_data[party].isna(),"Sample"].replace({',': ''}, regex=True).astype(int).to_numpy()
            Rs = ys * (1-ys) / sample_size
        
            x0 = np.array([0,0]).reshape(-1,1) # [local value, local trend]
            F = np.array([[1,1], # u_t+1 = u_t + delta_t (+ noise)
                        [0,1]]) # delta_t+1 = delta_t (+ noise)
            # F = np.array([[1,1,0, 0, 0, 0],
            #               [0,1,0,-1,-1,-1],
            #               [0,0,-1,0,-1,-1],
            #               [0,0,-1,-1,0,-1],
            #               [0,0,-1,-1,-1,0]]) ## --- This is for a seasonal model ##
            
            H = np.array([1,0]).reshape(1,-1)

            Q = np.diag([0.0005**2,0.00005**2])
            for _ in tqdm(range(max_iter_EM),leave=False):

                ts, Yhat_history,Xhistory,Phistory = kalman_run(timestamps,ys,Rs,x0,Q,F,Q,H)
                Qprev = Q.copy()
                Q = estimate_Q(Xhistory,F)
                if np.linalg.norm(Q-Qprev)<1e-8:
                    break
            
                
            Xadjusted = []
            Padjusted = []
            for idx,(X,P) in enumerate([i for i in zip(Xhistory,Phistory)][::-1]):
                if idx ==0:
                    Xadjusted.append(X)
                    Padjusted.append(P)
                else:
                    Xgiven_n = Xadjusted[-1]
                    Pgiven_n = Padjusted[-1]

                    xgiven_idx = F @ X
                    Pgiven_idx = F @ P @ F.T + Q
                    try:
                        Cidx = P @ F.T @ np.linalg.inv(Pgiven_idx)
                    except np.linalg.LinAlgError as e:
                        Cidx = P @ F.T @ np.linalg.pinv(Pgiven_idx)

                    Xadjusted.append(X + Cidx @ (Xgiven_n - xgiven_idx))
                    Padjusted.append(P + Cidx @ (Pgiven_n - Pgiven_idx) @ Cidx.T)

            Yadjusted = [(H @ Xadj).item() for Xadj in Xadjusted][::-1]
            Sadjusted = [(H @ Padj @ H.T).item() for Padj in Padjusted][::-1]

            election_date = ts.get_loc(pd.Timestamp("2022-05-22"))





            swings2025_df.loc[os.path.splitext(file)[0],party]=(Yadjusted[-1]-Yadjusted[election_date])
            swings2025_df.loc[os.path.splitext(file)[0],f"{party}_std"]=np.sqrt(Sadjusted[election_date]+Sadjusted[-1])

            election2025_df.loc[os.path.splitext(file)[0],party]=(Yadjusted[-1])
            election2025_df.loc[os.path.splitext(file)[0],f"{party}_std"]=np.sqrt(Sadjusted[-1])

            election2022_df.loc[os.path.splitext(file)[0],party]=(Yadjusted[election_date])
            election2022_df.loc[os.path.splitext(file)[0],f"{party}_std"]=np.sqrt(Sadjusted[election_date])
    
    swings2025_df.index.name = "demographic"
    swings2025_df = swings2025_df.drop(['UND','UND_std'],axis=1)
    swings2025_df['OTH'] = - swings2025_df[parties_modelled].sum(axis=1)
    swings2025_df['OTH_std'] = swings2025_df.apply(compute_y_std,axis=1)
    swings2025_df = swings2025_df.reindex(['ALP','ALP_std','LNC','LNC_std','GRN','GRN_std','PHON','PHON_std','OTH','OTH_std','ALP 2pp','ALP 2pp_std', 'L-NP 2pp', 'L-NP 2pp_std'],axis=1)
    swings2025_df.to_csv("processed/polling_estimates2025swings.csv")

    election2025_df.index.name = "demographic"
    election2025_df = election2025_df.drop(['UND','UND_std'],axis=1)
    election2025_df['OTH'] = 1 - election2025_df[parties_modelled].sum(axis=1)
    election2025_df['OTH_std'] = election2025_df.apply(compute_y_std,axis=1)
    election2025_df = election2025_df.reindex(['ALP','ALP_std','LNC','LNC_std','GRN','GRN_std','PHON','PHON_std','OTH','OTH_std','ALP 2pp','ALP 2pp_std', 'L-NP 2pp', 'L-NP 2pp_std'],axis=1)
    election2025_df.to_csv("processed/polling_estimates2025.csv")


    election2022_df.index.name = "demographic"
    election2022_df = election2022_df.drop(['UND','UND_std'],axis=1)
    election2022_df['OTH'] =  1 - election2022_df[parties_modelled].sum(axis=1)
    election2022_df['OTH_std'] = election2022_df.apply(compute_y_std,axis=1)
    election2022_df = election2022_df.apply(normalise,axis=1)
    election2022_df = election2022_df.reindex(['ALP','ALP_std','LNC','LNC_std','GRN','GRN_std','PHON','PHON_std','OTH','OTH_std','ALP 2pp','ALP 2pp_std', 'L-NP 2pp', 'L-NP 2pp_std'],axis=1)
    election2022_df.to_csv("processed/election2022estimates.csv")

    print(election2025_df)

    # plot_ts = np.where(ts>=np.datetime64("2022-05-22"))[0]

    # plot_timestamps = np.where(timestamps>=np.datetime64("2022-05-22"))[0]
    # plt.plot(ts[plot_ts],np.array(Yadjusted)[plot_ts])

    # ylower = np.array(Yadjusted) - np.sqrt(np.array(Sadjusted))*1.96
    # yupper = np.array(Yadjusted) + np.sqrt(np.array(Sadjusted))*1.96

    # plt.fill_between(ts[plot_ts],ylower[plot_ts],yupper[plot_ts],color=(0.1, 0.2, 0.5, 0.3))
    # plt.scatter(timestamps[plot_timestamps],ys[plot_timestamps],sample_size[plot_timestamps]/max(sample_size[plot_timestamps])*100)
    # plt.show()

    # Xhistory = np.array(Xhistory)
    # plt.plot(ts[plot_ts],Xhistory[plot_ts][:,1])
    # plt.show()



