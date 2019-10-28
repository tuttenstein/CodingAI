# =============================================================================
# HOMEWORK 1 - Supervised learning
# LINEAR REGRESSION ALGORITHM TEMPLATE
# =============================================================================


from sklearn import datasets, metrics, linear_model, model_selection
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np



# Load diabetes data from 'datasets' class
# =============================================================================

diabetes = datasets.load_diabetes()

# print(diabetes.data[0])
# print(diabetes.target[0])
# print(diabetes.DESCR)

# =============================================================================



# Get samples from the data, and keep only the features that you wish.
# =============================================================================

# Load just 1 feature for simplicity and visualization purposes...
# X: features
# Y: target value (prediction target)

X = diabetes.data[:, np.newaxis, 2]
# print(X.shape)

y = diabetes.target
# print(y.shape)

# =============================================================================

# Create linear regression model.
# =============================================================================


linearRegressionModel = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)

# =============================================================================


# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y)

# Let's train our model.
# =============================================================================

linearRegressionModel.fit(x_train, y_train)

# =============================================================================




# Ok, now let's predict the output for the test input set
# =============================================================================

y_predicted = linearRegressionModel.predict(x_test)

# # =============================================================================



# Time to measure scores. We will compare predicted output (resulting from input x_test)
# with the true output (i.e. y_test).
# You can call 'pearsonr()' or 'spearmanr()' methods for computing correlation,
# 'mean_squared_error()' for computing MSE,
# 'r2_score()' for computing r^2 coefficient.
# =============================================================================

# ADD COMMANDS TO EVALUATE YOUR MODEL HERE (AND PRINT ON CONSOLE)
r1, p1 = stats.spearmanr(X, y)
r2, p2 = stats.pearsonr(X, y)

print('Spearman = {}'.format(r1))
print('Pearson correlation = {}'.format(r2))
print('Mean Square error - MSE = {}'.format(metrics.mean_squared_error(y_test, y_predicted)))
print('R^2 coefficient = {}'.format(metrics.r2_score(y_test, y_predicted, sample_weight=None, multioutput='uniform_average')))
# print('R^2 coefficient = {}'.format(linearRegressionModel.score(x_test, y_test)))

# =============================================================================




# Plot results in a 2D plot (scatter() plot, line plot())
# =============================================================================

plt.scatter(x_test, y_test)
plt.plot(x_test, y_predicted, color='red')

plt.xlabel('Mass body index')
plt.ylabel('y')

# Display 'ticks' in x-axis and y-axis
plt.xticks()
plt.yticks()

# Show plot
plt.show()

# =============================================================================
