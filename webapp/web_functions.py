import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import accuracy_score

def load_data():
    # load dataset
    dh = pd.read_csv('heart1.csv')
    x = dh[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
    y = dh[['target']]
    return dh, x, y

def proses_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return x_train, x_test, y_train, y_test

def train_model(x_train, y_train, n_neighbors=4):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)

    return model, model.score(x_train, y_train)

def predict(x_train, y_train, x_test, features):
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(x_train, y_train)

    features_processed = sc.transform(features.reshape(1, -1))
    prediction = model.predict(features_processed)
    score = model.score(x_test)

    return prediction, score

def calculate_error_rate(x_train, y_train, x_test, y_test, max_k):
    error_rate = []

    for k in range(1, max_k + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)

        pred = model.predict(x_test)
        error_rate.append(np.mean(pred != y_test.values.ravel()))

    return error_rate

def app():
    st.title("Prediksi Error Rate dengan KNN")
    dh, x, y = load_data()

    x_train, x_test, y_train, y_test = proses_data(x.values, y.values.ravel())

    if st.checkbox("Prediksi Error Rate"):
        max_k = 40
        error_rate = calculate_error_rate(x_train, y_train, x_test, y_test, max_k)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red',
                 markersize=10)
        plt.title('Error Rate vs K')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        st.pyplot()
