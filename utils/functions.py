import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import numpy as np
from datetime import *
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def custom_tokenize(text):
    if not text:
        print('The text to be tokenized is a None type. Defaulting to blank string.')
        text = ''
    return nltk.word_tokenize(text)

def remove_stopword(tokens):
    stop_words = set(stopwords.words('english'))
    stop_words_lst = ["", "?", "!", ";", ".", "/", "//", ")",
                      "(", "|", "'", "''", "{", "}", "[", "]",
                      ">", "<", "=", ":", ",", "&", "'s", "'m",
                      "'d", "*", "\\n", "``", "...", "”", "â€",
                      "--"," - ", "#", "+", "$", "%","im"]
    
    filtered_sentence = [w for w in tokens if not w in stop_words]
    filtered_sentence = []

    for w in tokens:
        if w not in stop_words:
            if w not in stop_words_lst:
                filtered_sentence.append(w)
    return filtered_sentence

def remove_shortword(tokens):
    longword = [t for t in tokens if len(t) > 2]
    return longword

def apply_lemmatizer(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in text]

def cleanText(dataframe):
    dataframe_copy = dataframe.copy()
    # creating dictionary for trans table 
    trans_dict_num ={"0": "", "1": "", "2": "", "3": "","4": "", "5": "", "6": "","7": "", "8": "", "9": ""} 
    trans_table_num ="0123456789".maketrans(trans_dict_num) 
    
    
    trans_dict_char ={"'": "", "â": "", "€": "", "™": "", "/": " ","|": " ", "@": "","I": "","-": " "} 
    trans_table_char ="abc".maketrans(trans_dict_char)
    # creating translate table from dictionary 

    dataframe_copy["BodyTitle"]= dataframe_copy["BodyTitle"].str.translate(trans_table_num)
    dataframe_copy["BodyTitle"]= dataframe_copy["BodyTitle"].str.translate(trans_table_char)
    
    try:
        # tokenizer
        dataframe_copy['tokens'] = dataframe_copy.BodyTitle.apply(custom_tokenize)
        # stopword
        dataframe_copy['tokensStopWord'] = dataframe_copy.tokens.apply(remove_stopword)
        # lemmatizer
        dataframe_copy['words_tokens'] = dataframe_copy.tokensStopWord.apply(apply_lemmatizer)
        dataframe_copy['words_tokens'] = dataframe_copy.words_tokens.apply(remove_shortword)
        
        return dataframe_copy  
    except:
        return dataframe 

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print() 
    
def pattern_searcher(search_str: str, search_list: str):
    search_obj = re.search(search_list, search_str)
    if search_obj:
        return_str = 1
    else:
        return_str = 0 
    return return_str

def vectorizeInputTest(question, most_freq):
    sentence_vectorsInput = []

    data = {'BodyTitle':  question, 'Tags': ['']}
    dfInput = pd.DataFrame(data, columns=['BodyTitle', 'Tags'])
    dataInput = cleanText(dfInput)
    corpusInput = dataInput['words_tokens']

    for sentence in corpusInput:
        sent_vectest = []
        for token in most_freq:
            if str(token) in sentence:
                sent_vectest.append(1)
            else:
                sent_vectest.append(0)
        sentence_vectorsInput.append(sent_vectest)
    sentence_vectorsInput = np.asarray(sentence_vectorsInput)

    X_dataInput = pd.DataFrame(sentence_vectorsInput, columns=most_freq)
    return X_dataInput

def predictTag(model, text_input, most_freq_words, most_freq_tags):
    X_dataInput = vectorizeInputTest(text_input, most_freq_words)

    y_prediction = model.predict(X_dataInput)

    checked = y_prediction[0]
    tags = list(most_freq_tags.keys())
    d = {'tags': tags, 'checked': checked}
    dftest = pd.DataFrame(d)
    tagslist = (dftest[(dftest['checked'] == 1)]['tags']).tolist()
    return tagslist

def trainModel(choiceModel, X_data, y_data):   
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)
    
    if choiceModel == 'LogisticRegression' :
        clf = LogisticRegression(random_state=1)
    if choiceModel == 'SGDClassifier' :       
        clf = SGDClassifier(random_state=1, max_iter=5, tol=1e-3)
    if choiceModel == 'MultinomialDB' :       
        clf = MultinomialNB()
    if choiceModel == 'LinearSVC' :       
        clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5))
    if choiceModel == 'Perceptron' :       
        clf = Perceptron(tol=1e-3, random_state=1)
    if choiceModel == 'PassiveAggressiveClassifier' :       
        clf = PassiveAggressiveClassifier(max_iter=1000, random_state=1, tol=1e-3) 
    if choiceModel == 'RandomForest' :       
        clf = RandomForestClassifier(n_estimators=10, random_state=1)
    if choiceModel == 'KNN' :       
        clf = KNeighborsClassifier()        
    else :
        clf = LogisticRegression(random_state=1, max_iter=5, tol=1e-3)
        
    clf_fit = MultiOutputClassifier(clf).fit(X_train, y_train)
    return clf_fit

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model
    with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus=corpus, id2word=dictionary,
                             num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values