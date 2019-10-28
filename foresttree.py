# =============================================================================
# HOMEWORK 2 - DECISION TREES
# RANDOM FOREST ALGORITHM TEMPLATE
# =============================================================================


from sklearn import datasets, metrics, ensemble, model_selection


# =============================================================================



# Load breastCancer data
# =============================================================================

breastCancer = datasets.load_breast_cancer()



# =============================================================================



# Get samples from the data, and keep only the features that you wish.
# Decision trees overfit easily from with a large number of features! Don't be greedy.
numberOfFeatures = 10
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target
print(breastCancer.target_names)


# RandomForestClassifier() is the core of this script. You can call it from the 'ensemble' class.
# You can customize its functionality in various ways, but for now simply play with the 'criterion' and 'maxDepth' parameters.
# 'criterion': Can be either 'gini' (for the Gini impurity) and 'entropy' for the Information Gain.
# 'n_estimators': The number of trees in the forest. The larger the better, but it will take longer to compute. Also,
#                 there is a critical number after which there is no significant improvement in the results
# 'max_depth': The maximum depth of the tree. A large depth can lead to overfitting, so start with a maxDepth of
#              e.g. 3, and increase it slowly by evaluating the results each time.
# =============================================================================

model = ensemble.RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=3, n_jobs=-1, random_state=0
                                        , max_features=None)


# =============================================================================



# The function below will split the dataset that we have into two subsets. We will use
# the first subset for the training (fitting) phase, and the second for the evaluation phase.
# By default, the train set is 75% of the whole dataset, while the test set makes up for the rest 25%.
x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)



# Let's train our model.
# =============================================================================


model.fit(x_train, y_train)

# =============================================================================

# Ok, now let's predict the output for the test input set
# =============================================================================

y_predicted = model.predict(x_test)


# =============================================================================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)


# Time to measure scores. We will compare predicted output (from input of x_test)
# with the true output (i.e. y_test).
# You can call 'recall_score()', 'precision_score()', 'accuracy_score()', 'f1_score()' or any other available metric
# from the 'metrics' library.
# The 'average' parameter is used while measuring metric scores to perform 
# a type of averaging on the data. DON'T WORRY ABOUT THAT JUST YET. USE EITHER 'MICRO' OR 'MACRO'.
# =============================================================================

print('Accuracy = {}'.format(metrics.accuracy_score(y_test, y_predicted)))
print('Precision = {}'.format(metrics.precision_score(y_test, y_predicted)))
print('Recall = {}'.format(metrics.recall_score(y_test, y_predicted)))
print('F1_score = {}'.format(metrics.f1_score(y_test, y_predicted)))

# =============================================================================
