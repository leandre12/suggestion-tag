import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from utils.functions import *

st.write("""# Formulaire de recherche des tags :""")

option = st.sidebar.selectbox('choisissez votre modèle de prédiction :',
                              ('LogisticRegression', 'SGDClassifier',
                               'MultinomialDB', 'LinearSVC', 'Perceptron',
                               'PassiveAggressiveClassifier',
                               'RandomForest', 'KNN'))
st.sidebar.write('votre choix:', option)

corpus = st.text_input('Saisissez votre question :')
#*******Recuperation des données pour la prédiction *************#
X_data = pd.read_csv('data/Backup/test_X_data.csv', sep='\t',engine='python')
X_data.drop(['Unnamed: 0'], axis=1, inplace=True)

y_data = pd.read_csv('data/Backup/test_y_data.csv', sep='\t',engine='python')
y_data.drop(['Unnamed: 0'], axis=1, inplace=True)

top1000words_exp = pd.read_csv('data/Backup/test_top1000_words.csv',header=None,
                               index_col=0,squeeze=True).to_dict()
top100tags_exp = pd.read_csv('data/Backup/test_top100_tags.csv',header=None,
                             index_col=0,squeeze=True).to_dict()

#**** Fin recuperation des données pour la prédiction ***********#

if (st.button('Prédire') & (corpus != None )):
    model = trainModel(option, X_data, y_data)
    tagslist = predictTag(model, corpus, top1000words_exp, top100tags_exp)
    st.write(str(tagslist))