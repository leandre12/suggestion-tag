import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from utils.functions import *
import pickle, json, requests
from flask import Flask, request, redirect, url_for, flash, jsonify

# st.write("""# Formulaire de recherche des tags :""")

# option = st.sidebar.selectbox('choisissez votre modèle de prédiction :',
#                               ('LogisticRegression', 'SGDClassifier',
#                                'Multinomi# alDB', 'LinearSVC', 'Perceptron',
#                                'PassiveAggressiveClassifier',
#                                'RandomF# orest', 'KNN'))
st.sidebar.write('Léandre ANDRIANIAINA :')

st.sidebar.write('Projet de suggestion de tag pour stackoverflow ')


#*******Recuperation des données pour la prédiction *************#
corpus = st.text_input('Saisissez votre question :')
X_data = pd.read_csv('Data/Backup/test_X_data.csv', sep='\t',engine='python')
X_data.drop(['Unnamed: 0'], axis=1, inplace=True)

y_data = pd.read_csv('Data/Backup/test_y_data.csv', sep='\t',engine='python')
y_data.drop(['Unnamed: 0'], axis=1, inplace=True)

top1000words_exp = pd.read_csv('Data/Backup/top1000_words.csv',header=None,
                               index_col=0,squeeze=True).to_dict()
top100tags_exp = pd.read_csv('Data/Backup/top100_tags.csv',header=None,
                             index_col=0,squeeze=True).to_dict()
# LogisticRegression est le modele le plus performant
# model = trainModel('LogisticRegression', X_data, y_data)
model = pickle.load(open('models/model.pkl','rb'))
            
#*******Information WEB-API *************#
url = 'http://0.0.0.0:5000/tag/'
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
j_data = json.dumps(corpus)
r = requests.post(url, data=j_data, headers=headers)

#**** Fin recuperation des données pour la prédiction ***********#

if (st.button('Recherche de tags') & (corpus != None ) & (model != None )):
    tagslist = predictTag(model, corpus, top1000words_exp, top100tags_exp)
    st.write('voici la liste des tags :' , str(tagslist))