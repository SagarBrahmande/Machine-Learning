from math import dist


def Fun():
    
    # Load data
    X = [1,2,3,4,5]
    Y = [3,4,2,4,5]

    # Least Square method
    mean_x = (X)
    mean_y = (Y)

    sum_x = 0
    sum_y = 0

          
    n = len(X)

    for i in range(n):
        sum_x += X[i]
        sum_y += Y[i]
    
    mean_x = sum_x / n
    mean_y = sum_y / n

    numerator = 0
    denomenator = 0
    
    # Equation of line is y = mx + c

    for i in range(n):
        numerator += (X[i] -mean_x)*(Y[i] - mean_y)
        denomenator += (X[i] - mean_x)**2

    m = numerator / denomenator

    c = mean_y - (m * mean_x)

    print("Slope of Regression line is",m)
    print("Y intercept of Regression line is",c)

    # Findout goodness of fit ie R Square
    ss_t = 0
    ss_r = 0

    for i in range(n):
        y_pred = c + m *X[i]
        ss_t += (Y[i] - mean_y) ** 2
        ss_r += (Y[i] - y_pred) ** 2

    r2 = 1 - (ss_r/ss_t)

    print(r2)

def main():
    print("---- Marvellous Infosystems by Piyush Khairnar-----")
    
    print("Supervised Machine Learning")

    Fun()

if __name__ == "__main__":
    main()


def closest(self,row):
    bestdistance = euc(row,self.TrainingData[0])
    bestindex = 0
    for i in rangr(1,len(self.TrainingData)):
        dist - euc(row,self.TrainingData[i])
        if dist < bestdistance:
            bestdistance = dist
            bestindex = i
    return self.TrainingTarget[bestindex]