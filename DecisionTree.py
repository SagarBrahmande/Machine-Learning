from sklearn import tree

def MarvellousML(weight,surface):

    BallFeatures = [[35,1],[47,1],[90,0],
                    [48,1],[90,0],[35,1],
                    [92,0],[35,1],[35,1],
                    [35,1],[96,0],[43,1],
                    [110,1],[35,1],[95,0]]
    
    Names = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    clf = tree.DecisionTreeClassifier()

    clf = clf.fit(BallFeatures,Names)

    result = clf.predict([[weight,surface]])

    if result == 1:
        print("Your Object like Tennis ball")
    elif result == 2:
        print("Your object is like Cricket ball")


def main():
    print("Enter weight of the object")
    weight = input()

    print("What is the surface type of your object Rough or Smooth")
    surface = input()

    if surface.lower() == "rough":
        surface = 1
    elif surface.lower() == "smooth":
        surface = 0
    else:
        print("Error : Wrong Input")
        exit()

    MarvellousML(weight,surface)

if __name__ == "__main__":
    main()


    