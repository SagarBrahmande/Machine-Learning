from sklearn import tree

def main():
    print("Ball Classification case study")

    #Load the data
    BallFeatures = [[35,"Rough"],[47,"Rough"],[90,"Smooth"],
                    [48,"Rough"],[90,"Smooth"],[35,"Rough"],
                    [92,"Smooth"],[35,"Rough"],[35,"Rough"],
                    [35,"Rough"],[96,"Smooth"],[43,"Rough"],
                    [110,"Rough"],[35,"Rough"],[95,"Smooth"]]

    Labels = ["Tennis","Tennis","Cricket","Tennis","Cricket"
              "Tennis","Cricket","Tennis","Tennis","Tennis"
              "Cricket","Tennis","Cricket","Tennis","Cricket"]
    
    obj = tree.DecisionTreeClassifier() #Decide the Algorithm

    obj = obj.fit(BallFeatures,Labels)  #Train Model

    print(obj.predict([[36,"Rough"], [91,"Smooth"]]))   #Test model


if __name__ == "__main__":
    main()