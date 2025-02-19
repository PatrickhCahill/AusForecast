'''
This file generates M. It defines the metropolis algorithm for simulate annealing. It then implements this using the processed data.
'''

# Globals
MODELED_PARTES = ['UAP', 'ONP', 'ALP', 'GRN', 'LNP', 'IND', 'TEAL', 'OTH']


# Imports
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm


# Functions
def TCP(y:np.array,M:np.array, tol=1e-4, max_iter_sub=1000):
    # eliminated = np.array([False]*len(y)).reshape(-1,1)
    # eliminated[np.isnan(y)] = True # Parties not running are eliminated
    yhats = []
    yhat = y.copy()
    yhats.append(yhat.copy())
    yhat[np.isnan(yhat)] = 0
    for _ in range(max_iter_sub):
        argssorted = np.argsort(yhat.reshape(-1))
        argssorted = argssorted[yhat.reshape(-1)[argssorted]>0] # Remove parties that are eliminated
        smallest_remaining_party = argssorted[0] # Index of smallest remaining party
        flow = np.zeros_like(yhat)
        flow[smallest_remaining_party] = 1
        flow =  M @ flow

        yhat = yhat + flow*yhat[smallest_remaining_party]
        yhat[smallest_remaining_party,0] = 0

        yhat[yhat<tol] = 0
        yhat = yhat / np.nansum(yhat)
        yhats.append(yhat.copy())

        if sum(yhat!=0)[0] == 2:
            return yhat #,yhats#, eliminated
    return yhat
def distribute(y,M, tol=1e-4, max_iter_sub=1000):
    '''
    Inputs:
    y: Observation (vector of length modeled_parties)x
    M: Transition matrix (num_parties x num_parties)
    tol: Tolerance for convergence. Default is 1e-6
    max_iter: Maximum number of iterations. Default is 100
    Outputs:
    yhat: Predicted observation

    Description:
    Position of nan in y indicates already eliminated parties.

    Position of negative value in y indicates the party to be eliminated.

    Distribute a vector of 1 for eliminated party according to M.as_integer_ratio

    If nan parties have value less than tolerance then set to 0.

    Check if already eliminated parties received votes. If did then distribute those.

    Repeat
    '''
    def check_all_eliminated(yhat, valid_parties):
        eliminated = ~valid_parties
        return np.all(yhat[eliminated]<tol)
    def set_eliminated(yhat, valid_parties, party_i):
        eliminated = ~valid_parties
        yhat[eliminated] = np.nan
        yhat = yhat / np.nansum(yhat)
        yhat[party_i] = -1
        return yhat
    def pref_flow(yhat, idx, M):
        ''' Eliminate idx and distribute preferences according to M. Ignores any elimination'''


        flow = np.zeros_like(yhat)
        flow[idx] = yhat[idx]
        flow = M @ flow
        yhat[idx] = 0
        return yhat[idx] + flow
    party_i = np.where(y==-1)[0][0] #np.argmin(y) # Index of party to be eliminated
    valid_parties = ~np.isnan(y) # Boolean array of uneliminated parties
    valid_parties[party_i] = False # Exclude party to be eliminated

    yhats = []

    yhat =  np.zeros_like(y)
    yhat[party_i] = 1
    yhats.append(yhat.copy())
    yhat = M @ yhat
    yhats.append(yhat.copy())
    if check_all_eliminated(yhat, valid_parties):
        return set_eliminated(yhat, valid_parties,party_i)
    

    for _ in range(max_iter_sub):
        for eliminated in np.where(~valid_parties)[0]:
            if yhat[eliminated]>tol:
                yhat = pref_flow(yhat,eliminated, M)
                yhats.append(yhat.copy())

            if check_all_eliminated(yhat, valid_parties):
                return set_eliminated(yhat, valid_parties, party_i)

    print("WARNING: Did not converge")
    return set_eliminated(yhat,valid_parties, party_i)

def simulate_anneal(M0:np.array, Ydist:pd.DataFrame, Ytcp:pd.DataFrame, T=0.9, alpha=0.9, w=0.5, max_epoch=100):
    def accept(current_err, prev_err,T):
        if current_err < prev_err:
            return True
        else:
            with np.errstate(over='call'):
                def handle_overflow(err, flag):
                    return 0

                np.seterrcall(handle_overflow)  # Set custom handler
                result = np.divide(-(current_err-prev_err), T)

                return np.random.rand()<result # Returns true or false

    def perturb(M):
        Msuggest = M.copy()
        i = np.random.randint(0,len(Msuggest))
        j = np.random.choice([val for val in range(len(Msuggest)) if val != i])    
        Msuggest[i,j] = np.clip(Msuggest[i,j]*np.random.uniform(0.95,1.05),0,1)
        Msuggest = Msuggest / Msuggest.sum(axis=1, keepdims=True)

        return Msuggest
    
    def diff_ignore_nan(tcp1:np.array,tcp2:np.array)->np.array:
        return np.nan_to_num(tcp1).reshape(-1,1) - np.nan_to_num(tcp2).reshape(-1,1)

            
    def iteration(M:np.array,Ydist_slice:pd.DataFrame,Ytcp_slice:pd.DataFrame, w)->float:
        '''
        Defines one iteration of the simulate anneal process.
        Is given 1 polling place from both the Ydist and Ytcp dataframes in the form of a slice ddf.

        Then transforms this into the correct format for the observable H.

        Then predicts y from H_y(M).

        Returns error abs(y-H_y).

        CHANGES TO M and T handled outside of this. THIS IS ABSTRACTING AWAY THE ERROR CALCULATION.
        '''

        
        primary_vote_slice = Ytcp_slice[MODELED_PARTES].iloc[0].to_numpy().reshape(-1,1)
        tcp_actual = Ytcp_slice[Ytcp_slice.columns.difference(MODELED_PARTES)].to_numpy().reshape(-1,1)
        tcp_prediction = TCP(primary_vote_slice, M)

        tcp_err = np.sum(diff_ignore_nan(tcp_prediction,tcp_actual)**2)
        dist_err = 0
        for CountNum in Ydist_slice['CountNum'].unique():
            pref_flow = Ydist_slice[Ydist_slice['CountNum']==CountNum].drop(["DivisionNm","CountNum","PPId"],axis=1).to_numpy().reshape(-1,1)
            pref_flow_prediction = distribute(pref_flow,M)
            dist_err += np.sum(diff_ignore_nan(pref_flow_prediction,pref_flow)**2) / len(Ydist_slice['CountNum'].unique())
        
        error = dist_err*w + (1-w)*tcp_err    
        return error

    PPIds = Ytcp['PPId'].unique()
    np.random.shuffle(PPIds)

    M = M0
    PPId = PPIds[0]
    Ytcp_slice = Ytcp[Ytcp['PPId']==PPId].drop(["DivisionNm","PPId"],axis=1)
    Ydist_slice = Ydist[Ydist['PPId']==PPId]
    curr_error = iteration(M,Ydist_slice,Ytcp_slice,w)

    errors = [curr_error]
    for _ in tqdm(range(max_epoch)):
        for PPId in tqdm(PPIds,leave=False):
            Mproposed = perturb(M)
            Ytcp_slice = Ytcp[Ytcp['PPId']==PPId].drop(["DivisionNm","PPId"],axis=1)
            Ydist_slice = Ydist[Ydist['PPId']==PPId]
            proposed_error = iteration(Mproposed,Ydist_slice,Ytcp_slice,w)

            if accept(proposed_error,curr_error,T):
                M = Mproposed.copy()
                curr_error = proposed_error
                errors.append(curr_error)
        T = T * alpha

    return M,np.asarray(errors)




if __name__=="__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    M0 = pd.read_csv("M0.csv",index_col='PartyAb')
    M0 = M0.to_numpy().T

    Ydist = pd.read_csv("processed/distributions.csv")
    Ytcp = pd.read_csv("processed/tcp_train.csv")
    Ytcp = Ytcp[~(Ytcp[MODELED_PARTES].isna()).all(axis=1)]
    M, errs = simulate_anneal(M0,Ydist,Ytcp)
    pd.DataFrame(M,columns=MODELED_PARTES,index=MODELED_PARTES).to_csv("M.csv")
    pd.Series(errs).to_csv("errs.csv")