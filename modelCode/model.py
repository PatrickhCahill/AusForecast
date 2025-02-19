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