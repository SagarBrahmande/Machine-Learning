import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def MarvellousPlayPredictor(data_path):
    
    #Step 1 : Load Data
    data = pd.read_csv(data_path, idex_col =0)

    print("Size of Actual Dataset",len(data))

    #Step 2: Clean,  Prepare and Manipulate data
    feature_names =["whether","Temprature"]

    print("Names of Features",feature_names)

    whether = data.Whether
    Temprature = data.Temprature
    play = data.Play

    #creating labelEncoder
    le = preprocessing.LabelEncoder()

    #concerting strin lebel into numbers.
    weather_encoded = le.fit_transform(whether)
    print(weather_encoded)

    #converting string into labels
    temp_encoded = le.fit_transform(Temprature)
    label =le.fit_transform(play)

    print(temp_encoded)

    #combining weather and temp into single listof tuple
    features = list(zip(weather_encoded, temp_encoded))

    #step 3: Train Data
    model = KNeighborsClassifier(n_neighbors =3)

    #Train the model using the training sets
    model.fit(features,label)

    #step 4 : Test Data
    predicted = model.predict([[0,2]]) # 0: Overcast, 2:Mild
    print(predicted)

def main():
    print("Machine Learning Application")

    print("Play Predictor application using K Nearest Knighbor algorithm")

    MarvellousPlayPredictor("PlayPredictor.csv")

if __name__ == "__main__":
    main()