<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
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

### Model output
We want to generate an output matrix $\Omega$ where each column $\Omega_i$ is a different election outcome. These election outcomes encode **every** piece of interest that I might have for visualisation or communication:

$$
\Omega_i =
\begin{array}{|c|c|c|c|c|c|c|}
\hline
{\rm{ALP}}_{\rm{seats}} & {\rm{LNP}}_{\rm{seats}} & \dots & {\rm{ALP}}_{\rm{voteshare}}^{\rm{Greenway}} & {\rm{TCP}}_{\rm{party-name-1}}^{\rm{Greenway}} & {\rm{TCP}}_{\rm{party-voteshare-1}}^{\rm{Greenway}} & \dots  \\
\hline
\end{array}
$$

Some of these values will be aggregate values like the number of seats for each party, the primary vote nationally for each party, and the two party-preferred etc... While the majority will be seat specific. At the outset I plan to use `CSV` to store these large output matrix.

### Seat specifics
In shorth, $\Omega$ is the output of a Monte-Carlo simulation. The aggregate values can be calculated after the simulation is completed. Hence, we now turn to the question of drawing samples for each seat.

We will follow a Bayesian approach to generate a probability distribution for each seat - for example $\rho(\rm{Greenway})$, which we then sample.

#### Available data:
1. Candidates. (We ignore candidate specific effects for now but we will keep track of which part(ies) are running)
2. Historical election results (eventually we should correct for redistributions but this is a task for another time).
3. National and (possibly state) polling with breakdowns for gender, ethnicity, income and other relevant variables.
4. Geographic region
   
#### First draft of model
1. Generate a national vote matrix. e.g. A large matrix that represents for each party the typical vote with preferences. **This can be done by looking at the last few elections and fitting a model to the preference flows**
2. Assume that every party is running in every seat.
3. Get the demographics of each seat. From demographic data predict the swing of the primary votes. **Account for the uncertainty in the demographic data**.
4. From polling predict the swing on a uniform basis nationally. **Account for the uncertainty in the polling data**.
5. Combine this in a Bayesian way to have a distribution of primary votes in the seat. The primary votes in each seat should have the covariance.
6. Sample from the covariate distribution to generate the election outcome. 
7. Disqualify and distribute the preferences of parties --  according the national vote matrix -- that are not running in that specific seat. The remainder forms the primary vote prediction.
8. Apply the national vote matrix to calculate the TCP. **This is where we can explore different scenarios of the election outcome**.
9. Record the two party preferred outcome for each seat.

#### Second draft
To model an election outcome, we follow a structured probabilistic approach that integrates historical data, demographic shifts, polling, and Bayesian inference to predict primary votes and two-party preferred (2PP) outcomes. Below, we detail each step in the process.

0. Choose the parties that we want to model:
   * ALP
   * LIB
   * NAT
   * LNP
   * GRN
   * ONP
   * TEAL
   * IND
   * UAP
   * CLP
   * Generic Other

1. Generating a National Vote Matrix

We first construct a national vote matrix, denoted as \(M\), where each row represents a political party, and each column represents the distribution of voter preferences, including first preferences and preference flows. This matrix is generated by analyzing historical election data over the past few cycles and fitting a statistical model to the preference flows. Let \(M_{ij}\) represent the probability that a voter who initially prefers party \(i\) will direct their vote to party \(j\) upon elimination. This forms a Markov transition matrix:

\[
M = \begin{bmatrix}
M_{11} & M_{12} & \cdots & M_{1n} \\
M_{21} & M_{22} & \cdots & M_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
M_{n1} & M_{n2} & \cdots & M_{nn}
\end{bmatrix}
\]

**To make this:**
*Let the preference distrubtion in each seat be given by $\rm{PD}_{\rm{seat}}$ which is just a matrix  where the first column is the primary votes, the second column is the second preferences of the first eliminated party distributed etc. This will take some work to create from the raw AEC files. From this we can observe $M^{\rm{seat}}$ which is the partial observation of $M$. It is not full because not every seat will have every candidate we want to model and not every candidate will be eliminated in the right order.*

*Suppose that party-1 is eliminated and the preferences are distributed to parties-$\alpha = (\alpha_i)$. Then the first column of $M^{\rm{seat}}$ is a noisy observation of the following:*
$$M^{\rm{seat}}_1 = \begin{bmatrix}
0 \\
   M_{\alpha_1} / Z^{\rm{seat}}_1 \\
   \vdots \\
   M_{\alpha_i}/ Z^{\rm{seat}}_1\\
   0 \\
\end{bmatrix}$$
*where $Z^{\rm{seat}}_1 = \sum_i M_{\alpha_i} $. From this we form a matrix across every elimination of party-1. We then multiply every column by a factor to guarantee that the most common non-zero entry is 1. This normalises across the vectors, at the risk of some error being lost. Some entries might not have been scaled. We look at the most common value in the remaining unscaled values and scale them with respect to the average already scaled values. Do this iteratively until everything is scaled. Now every value will be proprotional to the true probability vector. We average column wise and the divide each entry by the sum to normalise. This gives us our party distribution best estimate of $M_1$. Do this for each party that we wish to measure to yield $M$.*


2. Assuming Every Party Runs in Every Seat

For the purpose of a national predictive model, we initially assume that all political parties contest every electoral seat. This simplifies the estimation process by standardizing preference flows and allows for a more uniform baseline before adjusting for real-world variations.

3. Estimating Demographic Influence on Primary Vote Swing

For each electoral seat \(s\), we obtain demographic data, denoted as \(D_s\). Using demographic factors such as income, education, age distribution, and past voting behavior, we estimate the swing in primary votes \(\Delta V_s\). This is modeled as:

\[
\Delta V_s = f(D_s) + \epsilon_s,
\]

where \(f(D_s)\) is a function mapping demographic variables to vote swings, and \(\epsilon_s\) captures uncertainty.

4. Incorporating National Polling-Based Swing

We incorporate national polling data, which provides a measure of uniform swing \(U\). Polling uncertainty is accounted for by treating \(U\) as a probability distribution rather than a single value, typically modeled as:

\[
U_s \sim \mathcal{N}(\mu_U, \sigma_U^2),
\]

where \(\mu_U\) is the mean swing from polling and \(\sigma_U\) represents its standard error.

5. Combining Demographic and Polling Data Using Bayesian Inference

For each seat \(s\), we combine the seat-level demographic swing \(\Delta V_s\) and the national swing \(U_s\) using Bayesian methods to generate a posterior distribution for primary votes:

\[
P(V_s | D_s, U_s) \propto P(D_s | V_s) P(U_s | V_s) P(V_s).
\]

This accounts for both local demographic effects and national trends, ensuring that the primary vote distributions maintain appropriate covariances across seats.

6. Sampling from the Covariate Distribution

To generate election outcomes, we sample from the posterior distribution obtained in the previous step. Let \(V_s^*\) represent a sampled realization of the primary vote distribution for seat \(s\). Repeating this process multiple times allows us to estimate election result probabilities.

7. Adjusting for Non-Running Parties

Since not all parties contest every seat, we adjust the primary votes by redistributing votes of non-running parties according to the national vote matrix \(M\). If party \(i\) is not contesting seat \(s\), we reallocate its votes using the conditional probabilities \(M_{ij}\) from \(M\), ensuring the total vote share remains normalized.

8. Calculating Two-Party Preferred (TCP) Outcomes

Using the preference matrix \(M\), we conduct the full preference distribution process to determine the two-party preferred (2PP) outcome. By iterating through preference distributions, we obtain the final vote shares for the top two parties \(A\) and \(B\) in each seat:

\[
T_s = g(V_s^*, M),
\]

where \(g\) is the function applying preferences to primary votes to compute the final TCP outcome.

9. Recording Two-Party Preferred Outcomes

Finally, for each sampled election scenario, we record the two-party preferred vote for each seat \(s\). Repeating this process across multiple iterations generates a distribution of likely election results, allowing for the exploration of different electoral scenarios under varying assumptions of polling error, demographic shifts, and candidate presence.

This methodology provides a robust framework for simulating and analyzing election outcomes in a statistically sound manner.

## 2. The website (and data visualisation)

   * Create interavtive bar chart graphics
   * Create interactive parliament graphiscs
      1. Requires parliament.svg in inkscape with id's
      2. Convert to no_inkscape_parliament.svg
      3. Fetch seats and modify colours by using d3 in js.
      4. Embed in broader html.

   * Unified design feel