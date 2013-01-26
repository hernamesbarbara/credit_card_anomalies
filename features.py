import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn import ensemble
from credit_card_data import read_data
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

df = read_data()

state_names = df.state1.unique()
state_names.sort()
state_names = pd.Series(state_names)

state_factor = pd.Factor(labels=state_names.index, levels=state_names.values, name="State")
df["state_factor"] = df.state1.apply(lambda x: state_factor.levels.get_loc(x))
df = df.drop("state1", axis=1)

domain_names = df.domain1.unique()
domain_names.sort()
domain_names = pd.Series(domain_names)

domain_factor = pd.Factor(labels=domain_names.index, levels=domain_names.values, name="Domain")
df["domain_factor"] = df.domain1.apply(lambda x: domain_factor.levels.get_loc(x))
df = df.drop("domain1", axis=1)

classes = pd.factorize(df.anomaly)
classes = pd.Factor(labels=classes[0], levels=classes[1], name="Anomaly")
df.anomaly = classes.labels

target_name = 'anomaly'
feature_names = np.array([col for col in df.columns if col != target_name])


X, y = shuffle(df[feature_names], df[target_name], random_state=13)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

params = {'n_estimators': 500,
         'max_depth': 4,
         'min_samples_split': 1,
         'learn_rate': 0.01,
         'loss': 'ls'}

########################################################################
clf = ensemble.GradientBoostingClassifier().fit(X_train, y_train)
predicted = clf.predict(X_test)

print "Mean Squared Error"
mse = mean_squared_error(y_test, predicted)
print("MSE: %.4f" % mse)
print 

params = clf.get_params()

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
        label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
        label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

cm  = confusion_matrix(y_test, predicted)
print "Confusion Matrix"
print
for i, row in enumerate(cm):
    print str(classes.levels[i]).ljust(5),str(row[0]).ljust(5), 
    print str(row[1])
print

cr  = classification_report(y_test, predicted)
print cr

feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

