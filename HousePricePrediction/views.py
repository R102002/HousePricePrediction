from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

def home(request):
    return render(request, "home.html")

def predict(request):
    return render(request, "predict.html")


def result(request):
    #Loading the dataset
    data=pd.read_csv("D:\django2\HousePricePrediction\HouseData.csv")
    data=data.drop(columns=["ADDRESS"], axis=1)
    #Train test split
    # x -> input variable, y -> o/p variable
    x = data.drop('TARGET(PRICE_IN_LACS)', axis=1)
    y = data['TARGET(PRICE_IN_LACS)']

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    #Training and predicting
    model = LinearRegression()
    model.fit(x_train, y_train)

    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])

    pred = model.predict(np.array([var1, var2, var3, var4]).reshape(1,-1))
    pred = round(pred[0] * 100000)
    price = "The predicted price is Rs:" + str(pred) + " lakhs"

    return render(request, "predict.html", {"result2": price})


# def result(request):
#     #Loading the dataset
#     data=pd.read_csv("D:\django2\HousePricePrediction\HouseData.csv")
#     data=data.drop(columns=["ADDRESS"], axis=1)
#     #Train test split
#     # x -> input variable, y -> o/p variable
#     x = data.drop('TARGET(PRICE_IN_LACS)', axis=1)
#     y = data['TARGET(PRICE_IN_LACS)']

#     x_train, x_test, y_train, y_test = train_test_split(x, y)

#     #Training and predicting
#     model = LinearRegression()
#     model.fit(x_train, y_train)

#     var1 = float(request.GET['n1'])
#     var2 = float(request.GET['n2'])
#     var3 = float(request.GET['n3'])
#     var4 = float(request.GET['n4'])

#     pred = model.predict(np.array([var1, var2, var3, var4]).reshape(1,-1))
#     pred = round(pred[0])
#     price = "The predicted price is Rs:" + str(pred)

#     return render(request, "predict.html", {"result2": price})


# from django.shortcuts import render;
# #importing required libraries

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics  #->will used at end to calculate accuracy
# def home(request):
#     return render(request,"home.html")


# def predict(request):
#     return render(request,"predict.html")


# def result(request):
#     #Loading the dataset
#     data=pd.read_csv("HouseData.csv")
#     # print(data.head())
#     data=data.drop(columns=["ADDRESS" ,"BHK_OR_RK","POSTED_BY","RERA","UNDER_CONSTRUCTION","BHK_OR_RK","READY_TO_MOVE","RESALE"],axis=1)
#     #Train test split
# #x->input variable y->o/p variable
#     x=data.drop('TARGET(PRICE_IN_LACS)',axis=1)
#     y=data['TARGET(PRICE_IN_LACS)']

#     x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30) #.30->means 70% of data will be in train part and 30% in testing part
#     #Training and predicting
#     model=LinearRegression()
#     model.fit(x_train,y_train)
    
#     var1=float(request.GET['n1'])
#     var2=float(request.GET['n2'])
#     var3=float(request.GET['n3'])
#     var4=float(request.GET['n4'])
    
#     pred=model.predict(np.array([var1,var2,var3,var4]))
#     pred=round(pred[0])
#     price="The predicted price is Rs:"+str(pred)
#     return render(request,"predict.html",{"result2":TARGET(PRICE_IN_LACS)})