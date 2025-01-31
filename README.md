# The Australian Election Forecast
This is my project investigating the psephology of the 2025 Australian Election. I aim to create a forecast of the Australian election that incorporates polling, historical data, models of preference flows and a website to interface with this. In general this will break down into two things:

## 1. The model
The aim is to forecast (with uncertainty) the TCP in each electorate.

To achieve that we will the following:
1. Create a class representing a seat. Every seat should have the following fields:
   * Current member
   * Current party
   * Contesting members
   * Contesting parties
   * **TBD: Seat specific data required for the forecast**

   * Mean vote and vote matrix method which can be sampled with MCMC
    Seats will also require the following methods:
   * A generate actual vote matrix method
   * Observe vote matrix

## 2. The website (and data visualisation)