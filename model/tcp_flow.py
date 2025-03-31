# We use M to guess the tcp finalists. But that is pretty conservative about the flow of preferences to the non-lead candidate because of the way that it calculates flows
# Hence, we use this actually calculate the expected flow from each of the parties to the final two candidates using direct tcp flow data from the election.


import pandas as pd
import numpy as np

tpp_flows = pd.read_csv("raw/HouseTppFlowByStateByPartyDownload-27966.csv",header=1)
tcp_flows = pd.read_csv("raw/HouseTcpFlowByStateByPartyDownload-27966.csv",header=1)