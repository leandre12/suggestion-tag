# Import libraries
import numpy as np
import pandas as pd
from flask import Flask, redirect, url_for, request, jsonify, render_template, jsonify
from utils.functions import predictTag
import pickle
app = Flask(__name__)
# Load the model
model = pickle.load(open('models/model.pkl','rb'))

#*******Recuperation des données pour la prédiction *************#

top1000words_exp = pd.read_csv('Data/Backup/top1000_words.csv',header=None,
                               index_col=0,squeeze=True).to_dict()
top100tags_exp = pd.read_csv('Data/Backup/top100_tags.csv',header=None,
                             index_col=0,squeeze=True).to_dict()

@app.route('/')
def home():
    return render_template('formtag.html')

@app.route('/tag',methods=['POST','GET'])
def tag():
    if request.method == 'POST':
        question = np.asarray(request.form['requete'])
        return redirect(url_for('question', quest=question))
    else:
        return render_template('formtag.html')

@app.route('/<quest>')
def question(quest):
    tagslist = predictTag(model, quest, top1000words_exp, top100tags_exp)
    return jsonify(tagslist)


if __name__ == '__main__':
    app.run(debug=True)