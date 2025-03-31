# Part 1 (Pre-Class Assignment)
# Setup Your Digital Workspace
# As a reminder, this includes the following:
#
# Import All Relevant Packages
# We will need one new function today (summary_col), which will enable us to make tables
# Set your Working Directory
# Load Up Your Data (ARC.csv)
# #=====================================#
# # Lab 6: Limited Dependent Variables  #
# # In Class Lab                        #
# # Written by: Zach Tragakis           #
# # Written on: 3/5/25                  #
# #=====================================#
# #import all relevant packages
import os
import pandas as pd                                 ## dataframe framework
import numpy as np                                  ## provides library of basic mathematical functions
import matplotlib.pyplot as plt                     ## necessary for plotting
import statsmodels.formula.api as smf               ## necessary for regressions
from statsmodels.iolib.summary2 import summary_col  ## enables you to save regression results (NEW!!)

# Pre-Class Work
# #Load In Your ARC Promote Data
ARC_data = pd.read_stata('ARC_promote.dta', convert_categoricals=True)
# #look at the first ten rows of your dataset to re-familiarize yourself with what is there.

print(ARC_data.head(10))

# Setup Necessary Variables For Lab
# Create a dummy variable to identify female employees
# #Create the female dummy variable
#
ARC_data['female']=(ARC_data['sex']=='female').astype(int)

# Complex Variable Generation: Using Summary Statistics
# Next, we want to learn how to generate some additional variables that contain summary statistics based on your dataset
# These types of variables are useful covariances in regressions.
#
# Let's try something relatively easy:
# generating a variable that contains average pay across all stores and in all time periods
#
# #First, what is average pay in the dataset?
print(ARC_data['pay'].describe())
#
# #ANSWER HERE : 18.25
#
# #Create a variable that contains average pay
ARC_data['mean_pay'] = ARC_data['pay'].mean()
# #Use the print command to check on the values that your new variable takes on. What do you notice?
#
# #ANSWER HERE
# print(ARC_data['mean_pay'].describe())

# Setting Up Variables Continued
# Let's make things a little more sophisticated by creating a variable that contains average pay by gender.
# First, calculate average pay by gender by hand.
#
# #Summarize average pay by gender
print(ARC_data.groupby('female')['pay'].mean())
#
# #ANSWER HERE
# Next, create a variable that contains gender-specific average pay
#
# #Create the variable using the "groupby" option
ARC_data['mean_pay_by_gender'] = ARC_data.groupby('female')['pay'].transform('mean')
# To understand what we've done:
#
# ARC_data.groupby('female')['pay'] : This groups the dataframe by the 'female' column, then within each group,
# it selects the 'pay' column
# .transform('mean'): This calculates the mean pay within each gender group and assigns
# the mean value to each row in the dataset
# ARC_data['mean_pay_by_gender']: The newly computed mean pay values (specific to gender)
# are stored in the new column in the ARC_data dataframe
# Check to be sure you've done things correctly!
#
# #Print top ten rows for women
print(ARC_data[ARC_data['female']==1][['pay','mean_pay_by_gender']].head())
#
# #Print top ten rows for men
print(ARC_data[ARC_data['female']==0][['pay','mean_pay_by_gender']].head())
#
# #ANSWER HERE
# Setting Up Variables Continued
# Next, recall that these data include multiple observations for each employee over time.
# Summarize how productive each employee is by generating a variable that captures employee-level
# average productivity across all stores and in all time periods using what you've learned.
#
# #Create the variable using the "groupby" option
# #USE WHAT YOU'VE LEARNED
# #Sort your data by emplid and confirm that things look okay
#
# #ANSWER HERE
ARC_data['empl_avg_prod'] = ARC_data.groupby('emplid')['prod'].transform('mean')
print(ARC_data['empl_avg_prod'])

# Setting Up Variables Continued
# Finally, generate a dummy variable called promote that equal to 1 for employees
# with the title "sr associate" and 0 otherwise.
#
# #Create the promote variable
ARC_data['promote'] = np.where((ARC_data['title'] == 'sr associate'),1, 0)


# #Check your work using a cross tabulate
print(pd.crosstab(ARC_data['title'], ARC_data['promote']))

# In-Class (Part 1)
# The first part of the in-class lab is designed to familiarize yourself with Linear Probability Models.
#
# First, limit your data to quarter 4 of 2021
#q
ARC_q42021 = ARC_data[(ARC_data['year']==2021) & (ARC_data['quarter']==4.0)]

print(ARC_q42021.head())

# #INSERT CODE TO CREATE NEW DATAFRAME
# What share of men and women were promoted as of Q4 2021?
#
# #Report the share of men and women being promoted
print(ARC_q42021.groupby('female')['promote'].mean())
#
# #ANSWER HERE
# 22% of men and 8% of women

# Estimate a Linear Probability Model
# Use an OLS regression to compare the probability of being promoted for men and women.
# # Setup the LPM
ols1 = smf.ols(formula='promote ~ female', data=ARC_q42021)
#
# # Estimate the regression
results_ols1 = ols1.fit(cov_type='HC2')
#
# #Store coefficients if needed for later
coef1 = results_ols1.params
#
# # Report OLS regression results
print(results_ols1.summary())
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.2217      0.029      7.585      0.000       0.164       0.279
# female        -0.1352      0.036     -3.773      0.000      -0.205      -0.065
# ==============================================================================
# #Answer the following questions
#
# #Interpret B0
# The probability of being promoted being a man
# #Interpret B1
# The effect of being female on the probability of being promoted

# #How do the estimates compare to the summary stats?
# They are the same values.

# Estimate Another LPM
# Now, estimate a Linear Probability Model (LPM) to estimate the relationship between employee
# level average productivity and the likelihood of being promoted.
# # Set up the LPM
ols2 = smf.ols(formula='promote ~ empl_avg_prod', data=ARC_q42021)

# #Estimate the regression
results_ols2 = ols2.fit(cov_type='HC2')
# #Store Regression Coefficients
coef2 = results_ols2.params
# # Report OLS regression results
print(results_ols2.summary())
# =================================================================================
#                     coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# Intercept         0.2999      0.091      3.305      0.001       0.122       0.478
# empl_avg_prod    -0.0006      0.000     -1.658      0.097      -0.001       0.000
# =================================================================================
# #Answer the following questions
#
# #Interpret B1
# The effect on promotion likelihood of a one unit increase in mean productivity (1%)

# #Calculate the predicted likelihood of being promoted for an employee with an average productivity of 100, 50, and 25
#
print(coef2['Intercept'] + 25 * coef2['empl_avg_prod'])
# 100: 23.6%
# 50: 26.83%
# 25: 28.41%

# Predict Promotion Using This Model
# Create predicted promotion probability using this model.
#
# # Fit the model
ARC_q42021['phat'] = results_ols2.fittedvalues
# Create a Scatterplot
plt.scatter(ARC_q42021['empl_avg_prod'], ARC_q42021['promote'], label='Data', color='blue', alpha=0.1)
plt.plot(ARC_q42021['empl_avg_prod'], ARC_q42021['phat'], label='LPS', color='red')

plt.title("Average Employee Productivity vs Probability of Promotion")
plt.xlabel("Average Productivity")
plt.ylabel("Promotion Probability")
plt.legend()

# Export the figure as a PNG
plt.savefig("avg_prod_histogram.png", format="png", dpi=300)

# Show the figure

plt.show()
# #What do you notice?
#
# As mean productivity goes up, likelihood of promotion goes down
#
# #ANSWER HERE

# Estimate a Linear Probability Model (Again!)
# Estimate a Linear Probability Model (LPM) to estimate the relationship between employee
# level average productivity and the likelihood of being promoted, controlling for gender and tenure.
#
# # Setup the LPM
#
# #Estimate the regression
#
ols3 = smf.ols(formula='promote ~ empl_avg_prod + female + tenure', data=ARC_q42021)
#
# # Estimate the regression
results_ols3 = ols3.fit(cov_type='HC2')
#
# #Store coefficients if needed for later
coef4 = results_ols3.params
#
# # Report OLS regression results
print(results_ols3.summary())

# =================================================================================
#                     coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# Intercept         0.4444      0.098      4.519      0.000       0.252       0.637
# empl_avg_prod    -0.0005      0.000     -1.348      0.178      -0.001       0.000
# female           -0.1269      0.036     -3.507      0.000      -0.198      -0.056
# tenure           -0.0017      0.001     -3.146      0.002      -0.003      -0.001
# ==============================================================================

# #Answer the following questions
#
# #Interpret the coefficient on empl_avg_prod
# The effect of a 1% increase in productivity on promotion likelihood, ignoring gender and tenure

# #Interpret the coefficient on female
# The effect of being female on promotion likelihood when average productivity is 0.

# #Interpret the coefficient on tenure
# The effect of tenure on promotion likelihood when average productivity is 0.
#
# Generate a few predicted outcomes
# What is the expected likelihood of promotion for a male employee with an average
# productivity of 50 and a tenure of 60 days?
# 31.74%

# What is the expected likelihood of promotion for a female employee with an average
# productivity of 50 and a tenure of 60 days?
# 19%

# #PREDCTIONS HERE
# Generate predictions for all observations in your dataframe
# # Fit the model
#

# Plot a histogram of predicted promotion likelihood
# #Set up the Figure
#
plt.close()
ARC_q42021['phat2'] = results_ols3.fittedvalues
# Create a Scatterplot
plt.scatter(ARC_q42021['empl_avg_prod']+ARC_q42021['female']+ARC_q42021['tenure'], ARC_q42021['promote'], label='Data', color='blue', alpha=0.1)
plt.plot(ARC_q42021['empl_avg_prod'], ARC_q42021['phat2'], label='LPS', color='red')

plt.title("Average Employee Productivity vs Probability of Promotion (with controls)")
plt.xlabel("Average Productivity")
plt.ylabel("Promotion Probability")
plt.legend()

# Export the figure as a PNG
plt.savefig("avg_prod_histogram2.png", format="png", dpi=300)

# Show the figure

plt.show()
#
# #What do you notice?
# ...

# In Class (Part 2): Probit Models
# In this part of the lab, you will use the probit model to estimate how the likelihood
# of promotion depends on gender, skill, and tenure.
# Estimate a Probit Model
# Estimate a probit model of promotion on gender
#
# # Set up the probit
probit1 = smf.probit(formula='promote ~ female', data=ARC_q42021)
#
# # Estimate the regression
results_probit1 = probit1.fit(cov_type='HC2')
#
# # Report OLS regression results
print(results_probit1.summary())
# ==============================================================================
#                  coef    std err          z      P>|z|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     -0.7665      0.098     -7.819      0.000      -0.959      -0.574
# female        -0.5962      0.164     -3.642      0.000      -0.917      -0.275
# ==============================================================================
# #Answer the following questions
#
# #What is the interpretation of the female coefficient?
# Effect on promotion probability of being female

# Making Predictions with Probit
# To make a prediction in a probit model, you need to use the standard normal CDF!
#
# To do this, we will need to update our available packages.
#

from scipy.stats import norm

# Answer the following:
#
# What is the predicted probability of promotion for a female?
# -0.7665 - -0.5962
# What is the predicted probability of promotion for a male?
# -0.7665
# #Store Regression Coefficients
pr_coef1 = results_probit1.params
#
print(norm.cdf(pr_coef1['Intercept'] + 1 * pr_coef1['female']))
# #The likelihood of being promoted is XX% for women
# 8.6%
# #Predicted probability for men?
print(norm.cdf(pr_coef1['Intercept'] + 1 * pr_coef1['female']==0))
# 50%

# #How does the likelihood of promotion change for men and women?
# 50% vs 8.6%

# #How does this estimate compare to your LPM?
#
# Estimate a Probit (Again!)
# Estimate a probit model of promotion on employee average productivity
# # Setup the probit
probit2 = smf.probit(formula='promote ~ empl_avg_prod', data=ARC_q42021)
#
# # Estimate the regression
results_probit2 = probit2.fit(cov_type='HC2')
#
# # Report OLS regression results
print(results_probit2.summary())
# =================================================================================
#                     coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# Intercept        -0.4110      0.367     -1.119      0.263      -1.131       0.309
# empl_avg_prod    -0.0027      0.002     -1.643      0.100      -0.006       0.001
# =================================================================================
# #
# #Store Regression Coefficients
pr_coef2 = results_probit2.params
# Create a histogram of predictions
# #Fit the data to create predictions
ARC_q42021['phat_probit'] = results_probit2.predict(ARC_q42021)
#
# #Setup the Figure
#
#
# #Export the figure as a png
#
#
# #Show the figure
plt.close()
plt.scatter(ARC_q42021['empl_avg_prod'],ARC_q42021['promote'], label='Data', color='blue', alpha=0.1)
plt.plot(ARC_q42021['empl_avg_prod'], ARC_q42021['phat_probit'], label='LPS', color='red')

plt.title("Average Employee Productivity vs Probability of Promotion (Probit Estimator)")
plt.xlabel("Average Productivity")
plt.ylabel("Promotion Probability")
plt.legend()

# Export the figure as a PNG
plt.savefig("avg_prod_histogram3.png", format="png", dpi=300)

# Show the figure

plt.show()

# #What do you notice?
# It's a curved line as opposed to linear

# Transform Probit Estimates to Marginal Effects
# Recall that the standard output of the probit must be transformed to report marginal effects.
# To do this, you will use .get_margeff method.
#
# Let's try using it by calculating the marginal effect of employee productivity,
# evaluated at the mean value of productivity
#
# #Calculate the marginal effect from results_probit2, evaluated at the mean for empl_avg_prod
print(results_probit2.get_margeff(at = 'mean').summary())
# =================================================================================
#                    dy/dx    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# empl_avg_prod    -0.0006      0.000     -1.652      0.099      -0.001       0.000
# =================================================================================
#
# #Interpret the marginal effect of employee productivity
# #one additional point of productivity increases the likelihood of promotion by .99 pp, evaluated at the mean
# Interpret Marginal Effects of Probit at Specified values
# Use the margins command to calculate the marginal effect of productivity on promotion
# for an employee with an average productivity of 50.
#
# #Calculate the marginal effect from results_probit2, evaluated at the mean for empl_avg_prod
#
print(results_probit2.get_margeff(atexog={1: 50}).summary())
# =================================================================================
#                    dy/dx    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# empl_avg_prod    -0.0009      0.001     -1.314      0.189      -0.002       0.000
# =================================================================================
#
# #Note that here, we define a dictionary using "atexog" that includes a value for each covariate.
# #In this case we have one covariate, avg prodc, and we set this to 50
#
# #Interpret the marginal effect of employee productivty
# #one additional point of productivity increases the likelihood of promotion by .46 pp
# for someone with an average productivity of 50

# Estimate a Probit (Again!)
# Run a probit model of promotion on employee average productivity, controlling for gender and tenure.
# # Setup the probit
probit3 = smf.probit(formula='promote ~ empl_avg_prod + female + tenure', data=ARC_q42021)
#
# # Estimate the regression
results_probit3 = probit3.fit(cov_type='HC2')
#
# # Report OLS regression results
print(results_probit3.summary())
# =================================================================================
#                     coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# Intercept         0.4812      0.453      1.062      0.288      -0.407       1.369
# empl_avg_prod    -0.0026      0.002     -1.555      0.120      -0.006       0.001
# female           -0.6001      0.165     -3.640      0.000      -0.923      -0.277
# tenure           -0.0109      0.004     -2.465      0.014      -0.020      -0.002
# =================================================================================
#
# #Store Regression Coefficients
pr_coef3 = results_probit3.params
# Interpret Marginal Effects of Probit
# Use the margins command to calculate the marginal effect of productivity on promotion,
# evaluated at the mean value of employee average productivity and tenure, for women.

# #Predict values
#
predict_values = {1: ARC_q42021['empl_avg_prod'].mean(),
                   2: 1,
                   3: ARC_q42021['tenure'].mean()
                  }
#
print(results_probit3.get_margeff(at = 'mean',dummy=True).summary())
# =================================================================================
#                    dy/dx    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# empl_avg_prod    -0.0006      0.000     -1.537      0.124      -0.001       0.000
# female           -0.1290      0.035     -3.681      0.000      -0.198      -0.060
# tenure           -0.0024      0.001     -2.554      0.011      -0.004      -0.001
# =================================================================================

print(results_probit3.get_margeff(atexog = predict_values,dummy=True).summary())
# =================================================================================
#                    dy/dx    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# empl_avg_prod    -0.0004      0.000     -1.436      0.151      -0.001       0.000
# female           -0.1290      0.035     -3.681      0.000      -0.198      -0.060
# tenure           -0.0016      0.001     -2.385      0.017      -0.003      -0.000
# =================================================================================

#
# #Interpret the marginal effect of employee productivty
# one additional point of productivity increases the likelihood of promotion by .68 pp, evaluated at the mean

# You have finished Lab 6!
