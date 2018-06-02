# importing modules
import sys
sys.path.append("../")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from src import ridge_regression

# reading spam dataset
spam_data = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data', header=None, delim_whitespace=True)

# pre-processing
X = spam_data.drop(57, axis=1)
y = spam_data[57]

# divide the data into training and test sets. By default, 25% goes into the test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Standardize the data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# append a column of 1's to X_train and X_test for the intercept
X_train = preprocessing.add_dummy_feature(X_train)
X_test = preprocessing.add_dummy_feature(X_test)

# making y_values {-1,1}
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1

# launch the method
clf = ridge_regression()
clf.fit(X_train, y_train)

# evaluating the performance of the method
res_test = clf.predict(X_test)

# calculating the misclassification error
test_df = pd.DataFrame(y_test)
test_df.columns = ['labels']
test_df['pred'] = res_test
test_df['error'] = test_df['labels'] - test_df['pred']
test_df['error'] = test_df['error'] != 0
test_df['error'] = test_df['error']*1

# calculating the total error
err = sum(test_df['error'])/float(test_df.shape[0])
print "The error on the spam dataset is", err
