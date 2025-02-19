import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
from tqdm import tqdm

## TO DO HANDLE ELECTION DAY SUPPORT ESTIMATE.

## FOR ONES WHERE ELECTION DATA ALREADY EXISTS RENAME UND to OTH. WITH POLLS CONVERT UND TO OTH.

## THIS REQUIRES INTERMEDIATE DATA HANDLING STAGE BEFORE THE KALMAN FILTER

## FOR OTHER VALUES WE WILL EXTRACT FROM THE 2022 AES ESTIMATES FOR EACH OF THE VALUES.
## THIS WILL GIVES US PRIMARY VOTES. FROM THERE WE ESTIMATE THE TPP WITH TRENDS FROM 2022.
## THIS WILL GIVES US THE ROWS FOR EACH OF THE POLLING DATA VARIABLES.
## THIS CAN THEN BE FED INTO THE POLLING MODEL

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

    outrow[columns_of_interest] = outrow[columns_of_interest]/100
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
    x = x0.copy()
    P = P0.copy()

    Yhat_history = [(H @ x).item()]
    Xhistory = [x]
    Phistory = [ P]

    for date in pd.date_range(np.datetime64('2022-05-22'),np.datetime64('2025-05-22'), freq='D')[1:]:
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

    ts = pd.date_range(np.datetime64('2022-05-22'),np.datetime64('2025-05-22'))
    return ts, Yhat_history,Xhistory,Phistory

def estimate_Q(X:list,F:np.array):
    Xforward = [F @ x for x in X]
    residuals = [X[i+1] - Xplus for i,Xplus in enumerate(Xforward[:-1])]
    Qest = sum(residual @ residual.T for residual in residuals)/len(residuals)
    return Qest

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    polling_data_dir = os.listdir("polling_data")
    columns_of_interest = ['ALP', 'LNC', 'GRN', 'PHON', 'UND', 'ALP 2pp', 'L-NP 2pp']
    out_df = pd.DataFrame(index=[os.path.splitext(filename)[0] for filename in polling_data_dir])

    for file in tqdm(polling_data_dir):
        # if file != "female.csv":
        #     continue
        polling_data = pd.read_csv(f"polling_data/{file}").apply(fixtpp,axis=1)
        election_day = polling_data.loc[0].copy()
        polling_data = polling_data.drop(0,axis=0) # Drop the election day data
        polling_data = polling_data.reset_index()



        for party in tqdm(columns_of_interest,leave=False):
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
            for _ in tqdm(range(1),leave=False):

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


            out_df.loc[os.path.splitext(file)[0],party]=Yadjusted[-1]
            out_df.loc[os.path.splitext(file)[0],f"{party}_std"]=Sadjusted[-1]
    print(out_df)

    # plt.plot(ts,Yadjusted)

    # ylower = np.array(Yadjusted) - np.sqrt(np.array(Sadjusted))*1.96
    # yupper = np.array(Yadjusted) + np.sqrt(np.array(Sadjusted))*1.96

    # plt.fill_between(ts,ylower,yupper,color=(0.1, 0.2, 0.5, 0.3))
    # plt.scatter(timestamps,ys,sample_size/max(sample_size)*100)
    # plt.show()

    # Xhistory = np.array(Xhistory)
    # plt.plot(ts,Xhistory[:,1])
    # plt.show()



