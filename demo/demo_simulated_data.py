#importing modules
import numpy as np
import random
import pandas as pd
from class_ridge import ridge_regression

# creating a simple simulated dataset
num_rows = 1000
num_col = 10

# creating features from a standard normal distribution
sim_data_features = np.random.normal(0, 1, (num_rows, num_col))
sim_data_features = pd.DataFrame(sim_data_features)

# adding an intercept 
sim_data_features[sim_data_features.shape[1]] = 1

# a random selection of -1s and 1s for the class labels
sim_data_labels = []
for i in range(0, num_rows):
    sim_data_labels.append(random.choice([-1,1]))
sim_data_labels = pd.DataFrame(sim_data_labels)
sim_data_labels.columns = ['y']

# creating the final dataset
sim_data = pd.concat([sim_data_features, sim_data_labels], axis=1)

# creating seperate train and test sets : 70% training
msk = np.random.rand(len(sim_data)) < 0.7
train = sim_data[msk]
test = sim_data[~msk]

# separating features and labels
train_features = train.drop('y', 1)
train_labels = train['y']
test_features = test.drop('y', 1)
test_labels = test['y']

# converting all into numpy arrays
train_features = train_features.as_matrix()
test_features = test_features.as_matrix()
train_labels = train_labels.values
test_labels = test_labels.values

# launch the method
clf = ridge_regression(lambda_val = 0.001)
clf.fit(train_features, train_labels)

# evaluating the performance of the method
res_test = clf.predict(test_features)

# calculating the misclassification error
test_df = pd.DataFrame(test_labels)
test_df.columns = ['labels']
test_df['pred'] = res_test
test_df['error'] = test_df['labels'] - test_df['pred']
test_df['error'] = test_df['error'] != 0
test_df['error'] = test_df['error']*1

#calculating the total error
err = sum(test_df['error'])/float(test_df.shape[0])
print "The error on the simulated dataset is", err