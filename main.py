import streamlit as st

st.write("Installing nltk...")
st.code("pip install nltk")
st.write("pip show nltk")
st.code("nltk installed!")


import nltk
import joblib
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.reset_default_graph()

import tflearn
import numpy as np
import random
import pickle
import json

with open("D:\chatbot\intents.json") as file:
    data = json.load(file)  # loading intents.json file

try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]: # (removing extra characters)
            wrds = nltk.word_tokenize(pattern) # bring all the words in a dict.
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words= [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words))) # remove all the duplicates.

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])]) # 1 layer/ input layer
net = tflearn.fully_connected(net, 8) # hidden layer 1
net = tflearn.fully_connected(net, 8) # hidden layer 2
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # output layer , softax gives the probability to each neuron for every particular word whenever we request an output
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True) # here it trains our data
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)

def chat():
    print("\n\nWelcome to Xananoids!! (type quit to stop)\n\n")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)]) # over here it will result probability of each neuron matches
        results_index = np.argmax(results)
        tag = labels[results_index] # this will return the specific tag
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        outpt=random.choice(responses)
        print(random.choice(responses))
        joblib.dump(outpt,'outpt.lb')

chat()
