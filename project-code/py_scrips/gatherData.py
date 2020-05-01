import requests
import io
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from os import listdir
from flask import Flask, request, send_file, make_response

# This implementation allows the user to place a file in called input.txt in the dir called input
# The structure of this could be imporved

code_dir = os.path.dirname(__file__)

def get_url():
    input_path = code_dir+'/url.txt'
    input_file = open(input_path, "rt")
    contents = input_file.read()
    url = contents.rstrip()
    input_file.close()
    return str(url)

def new_download(filename):
    url = get_url()
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)

#def download_data(url, filename):
#    r = requests.get(url, allow_redirects=True)
#    open(filename, 'wb').write(r.content)
#    return

def download(output):
    data_dir = code_dir+'/../savedFiles/'
    output_file = data_dir+output
    new_download(filename=output_file)
    return  str(output) + " Downloaded" + " to " + str(data_dir) 
