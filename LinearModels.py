import pandas as pd
import statsmodels.api as sm

# Solve this problem using the pandas and statsmodel (this includes the OLS method) libraries


# Read the data from the 'data\World Indicators.csv' into a variable named data

# Your code starts here
data = pd.read_csv("/Users/agataporwit/PycharmProjects/LinearModels/World Indicators.csv")

# Your code ends here


# Filter the population so that only data from Africa remains

# Your code starts here
region = data.loc[data['Region'] == 'Africa']
print(region)
# Your code ends here


# Subset the data so that only the column Year and Population Total remain

# Your code starts here

regionFiltered = region[["Year", "Population Total"]]
print(regionFiltered)
# Your code ends here


# Group the data so that the population is aggregated by year into a new data set name 'byyear'

# If using the pandas.groupby function, set the parameter as_index to False

# Your code starts here
byyear = regionFiltered.groupby(["Year"], as_index=False).sum()

# Your code ends here


print(byyear)

# Create a linear model of Population Total by year (don't forget the constant)

# Name the result of calling the fit() function fit

# Your code starts here

X = sm.add_constant(byyear['Year'])
Y = byyear['Population Total']
fit = sm.OLS(Y, X).fit()
# Your code ends here


print(fit.params)

# Predict the population in Africa in the Year 2015, name the result 'prediction'

# Your code starts here


# Your code ends here
prediction = fit.predict([1, 2015])

print(prediction)
