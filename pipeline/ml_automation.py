from sklearn import svm
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, StratifiedKFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

import numpy as np
import pandas as pd

import generate_classified_image as gen
import image_slice as slicer
import attr_extraction as processor


''' train model with upsampled data
    @return the trained model '''

def train_model(train):

    train['Slice'].replace(to_replace="[a-zA-Z_.]", value='', regex=True, inplace=True)
    train['Slice'].astype('float64')
    attr = ['Slice', 'Number of Blob', 'Average Size', 'Circ', 'AR', 'Greenness']
    X = train[attr]
    y = train['Class']
    #sme = SMOTEENN() # upsample
    sme = SMOTEENN(random_state=0)
    X_train, y_train = sme.fit_sample(X, y)
    X_train = pd.DataFrame(X_train[:,1:], index=X_train[:,0])
    linearly_separable = (X_train, y_train)
    print('Training dataset shape {}'.format(Counter(y_train)))

    knn = KNeighborsClassifier(2)
    predicted = cross_val_predict(knn, X_train, y_train, cv=10)
    cv_res = metrics.accuracy_score(y_train, predicted)
    knn.fit(X_train, y_train)

    print("cross validation accuracy:", cv_res)
    
    return knn



''' predict which image is in the mixed state
    @return a list containing the index of images '''

def make_prediction(model, test):
    
    test['Slice'].replace(to_replace="[a-zA-Z_.]", value='', regex=True, inplace=True)
    test['Slice'].astype('float64')
    attr = ['Slice', 'Number of Blob', 'Average Size', 'Circ', 'AR', 'Greenness']
    X_test = test[attr]
    y_test = test['Class']
    X_test = X_test.set_index('Slice')
    
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print("test data score:", score)

    df = pd.DataFrame(X_test)
    df["predicted"] = y_pred
    #incorrect = df[df["actual"] != df["predicted"]]

    dry = []
    wet = []
    mixed = []
    for index, row in df.iterrows():
        if row.iloc[5] == 0:
            dry.append(index)
        elif row.iloc[5] == 1:
            wet.append(index)
        elif row.iloc[5] == 2:
            mixed.append(index)
    return dry, wet, mixed

    
#print(incorrect)


##
##params_space = {'kernel': ['linear', 'poly', 'rbf']}
##gs = GridSearchCV(knn, params_space, n_jobs=-1, cv=5)
##gs.fit(X, y)
##gs.predict_proba(X)


##for train_index, test_index in kf.split(X):
##    print("TRAIN:", train_index, "TEST:", test_index)
##    X_train, X_test = X[train_index], X[test_index]
##    y_train, y_test = y[train_index], y[test_index]
##    knn.fit(X_train, y_train)
##    knn.predict(X_test)
##    
##  

#a_train, a_test, b_train, b_test = train_test_split(X, y, test_size=0.1)
#print(a_train)
##for train, test in skf.split(X, y): 
##    X_train, X_test = X.columns.get_indexer([train]), X.columns.get_indexer([test])
##    y_train, y_test = y[train], y[test]
##    knn.fit(X_train, y_train)
##    knn.predict(X_test)
##    

##training_features, test_features, \
##training_target, test_target, = train_test_split(data.drop(['Class'], axis=1),
##                                               data['Class'],
##                                               test_size = .1,
##                                               random_state=12)
#training_attr = data.drop(['Class'])
#training_target = data['Class']                          
#ros = RandomOverSampler(random_state=0)
#x_res, y_res = ros.fit_sample(training_features, training_target)


#print training_target.value_counts(), np.bincount(y_res)

#data_resampled = SMOTE().fit_sample(data, label)
#print(sorted(Counter(data_resampled).items()))

#print(data.target[:10])

def classify_image(iteration):
    training_data = pd.read_csv('/Users/violet/Documents/18WI/CSC499/evaluation/training.csv')
    test_data = pd.read_csv('/Users/violet/Documents/18WI/CSC499/evaluation/test.csv')

    model = train_model(training_data)
    dry, wet, mixed = make_prediction(model, test_data)

    gen.generate_image(dry, 'dry', iteration)
    gen.generate_image(wet, 'wet', iteration)
    new_dir = slicer.slice_image(mixed, iteration)
    processor.batch_processing(new_dir + '*', iteration)
    
    print("finish iteration: ", iteration)


classify_image(1)
