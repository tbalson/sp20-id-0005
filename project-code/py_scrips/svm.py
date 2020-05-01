from flask import Flask, request, send_file, make_response
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import confusion_matrix
from sklearn.externals.joblib import Memory
from sklearn.decomposition import PCA
from sklearn import datasets, svm
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from flask import jsonify
from time import time
import numpy as np
import logging
import pickle
import json
import os
import io
code_dir = os.path.dirname(__file__)

def get_data(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]

def info(filename):
    place = code_dir+"/../savedFiles/"+str(filename)
    file_of_faces_pkl = open(place, 'rb')
    lfw_people = pickle.load(file_of_faces_pkl)
    n_samples, h, w = lfw_people.images.shape

    X = lfw_people.data
    n_features = X.shape[1]

    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    
    return ["total dataset size: "+str(n_samples),"features:"+str(n_features),"classes:"+str(n_classes)]

def svm_gen(filename, trainingSize):
    place = code_dir+"/../savedFiles/"+str(filename)
    file_of_faces_pkl = open(place, 'rb')
    lfw_people = pickle.load(file_of_faces_pkl)
    n_samples, h, w = lfw_people.images.shape

    X = lfw_people.data
    n_features = X.shape[1]

    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    #train the data
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, train_size = float(trainingSize))
    # Compute PCA on the dataset
    n_components = 150
    pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    # train a SVM classification model
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'), param_grid
    )
    clf = clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cleaned_report = clean(report,n_classes)
    return [cleaned_report,"completed svm endpoint"]        

def clean(c,n_classes):
    newc = c.replace("\n\n", "\n")
    each = []
    total = n_classes + 4
    for i in range(total):
        each.append("")
    each = newc.split('\n',10)        
    for i in range(total):
        each[i] = each[i].strip("\n")
    return each

def gen_cof_mat(filename):
    place = code_dir+"/../savedFiles/"+str(filename)
    file_of_faces_pkl = open(place, 'rb')
    lfw_people = pickle.load(file_of_faces_pkl)
    #lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    
    n_samples, h, w = lfw_people.images.shape

    x = lfw_people.data
    y = lfw_people.target
    class_names = lfw_people.target_names
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    c_value = .1
    kernel_val = "linear"
    classifier = SVC(kernel=kernel_val, C = c_value)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    # Plot normalized confusion matrix
    bytes_obj = plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

    return send_file(bytes_obj,
                     attachment_filename='plot.png',
                     mimetype='image/png')

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    bytes_image = io.BytesIO()
    bytes_image
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image
