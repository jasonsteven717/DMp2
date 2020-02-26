# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:23:11 2019

@author: TsungYuan
"""
'''
----------------------------------------
20 features
----------------------------------------
age : 20-65
company-class : public 0, private 1
education-level : middle school 0, High school 1, university 2, graduate school 3
education-class : public 0, private 1
marital-status : married 0, never married 1
race : white 0, black 1, yellow 2
gender : male 0, female 1
hours-per-day : 6-12
deposit : -100-+100
children : 0-5
cars : 0-3
houses : 0-3
height : 150-190
weight : 50-100
blood group : A 0, B 1, O 2, AB 3
constellation : 1-12
region : city center 0, countryside 1
eating : meat 0, vegetarian 1
do exercise : yes 0, no 1
entertainment-per-week : 0-3
----------------------------------------
label
----------------------------------------
income-per-year : >=100 0,<=100 1
----------------------------------------
11 rules
----------------------------------------
age : 30-40 +1, 40-50 +2, 50-65 +3
education-level : university +1, graduate school +2
education-class : public +1
public and graduate school +1
marital-status : married +1
hours-per-day : >=8 +1
deposit : 0-50 +1, >50 +2
cars : >1 +1
houses : =1 +1, >1 +1
region : city center +1
entertainment-per-week : >=2 +1
maxincome=16, income >= 10
'''
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

features = ["age", "company-class", "education-level", "education-class", "marital-status", "race", "gender", "hours-per-day", "deposit", "children",
            "cars", "houses", "height","weight", "blood group", "constellation","region", "eating", "do exercise", "entertainment-per-week", "income-per-year"]
labels = ["income-per-year<100w","income-per-year>=100w"]
data_num = 1100
feature = 20

def createdata():
    x1 = np.random.randint(20,66,(data_num,1))
    x2 = np.random.randint(0,2,(data_num,1))
    x3 = np.random.randint(0,4,(data_num,1))
    x4 = np.random.randint(0,2,(data_num,1))
    x5 = np.random.randint(0,2,(data_num,1))
    x6 = np.random.randint(0,3,(data_num,1))
    x7 = np.random.randint(0,2,(data_num,1))
    x8 = np.random.randint(6,12,(data_num,1))
    x9 = np.random.randint(-50,+100,(data_num,1))
    x10 = np.random.randint(0,6,(data_num,1))
    x11 = np.random.randint(0,4,(data_num,1))
    x12 = np.random.randint(0,4,(data_num,1))
    x13 = np.random.randint(150,190,(data_num,1))
    x14 = np.random.randint(50,100,(data_num,1))
    x15 = np.random.randint(1,13,(data_num,1))        
    x16 = np.random.randint(0,4,(data_num,1)) 
    x17 = np.random.randint(0,2,(data_num,1))
    x18 = np.random.randint(0,2,(data_num,1))
    x19 = np.random.randint(0,2,(data_num,1))
    x20 = np.random.randint(0,4,(data_num,1))
    x21 = np.zeros((data_num,1))

    data = np.hstack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21))
    
    for i in range(data_num):
        income = 0
        if data[i,0] >= 30 and data[i,0] < 40: #age
            income += 1
        elif data[i,0] >= 40 and data[i,0] < 50:
            income += 2 
        elif data[i,0] >= 50:
            income += 3        
        if data[i,2] == 2: #education-level
            income += 1
        elif data[i,2] == 3:
            income += 2
        if data[i,3] == 0: #education-class
            income += 1 
        if data[i,3] == 0 and data[i,2] == 3:
            income += 1 
        if data[i,4] == 0: #marital-status
            income += 1 
        if data[i,7] >= 8: #hours-per-day
            income += 1
        if data[i,8] <= 50 and data[i,8] > 0: #deposit
            income += 1
        elif data[i,8] > 50:
            income += 2
        if data[i,10] > 1: #cars
            income += 1
        if data[i,11] >= 1: #houses
            income += 1
        if data[i,11] == 1: #houses
            income += 1
        if data[i,11] > 1: #houses
            income += 1
        if data[i,16] == 0: #houses
            income += 1
        if data[i,20] >= 2: #houses
            income += 1
        if income >= 10: #region
            data[i,feature] = 1
    #df = pd.DataFrame(data, columns = features)
    #print(df)
    return data

from sklearn import metrics    
from sklearn import tree
import pydotplus
from IPython.display import Image  
from sklearn.tree import DecisionTreeClassifier, plot_tree
def DTmodel(data):
    dt = tree.DecisionTreeClassifier(criterion = "entropy")
    model  = dt.fit(data[0:1000,0:feature-1], data[0:1000,feature])
    accuracy = metrics.accuracy_score(data[1000:1100,feature], model.predict(data[1000:1100,0:feature-1]))
    print("DT testdata accuracy : ",accuracy)
    dot_data = tree.export_graphviz(dt, out_file=None, 
                                    feature_names=features[0:feature-1],  
                                    class_names=labels)
    graph = pydotplus.graph_from_dot_data(dot_data)  
    Image(graph.create_png())
    graph.write_png("decisiontree.png")
 
from sklearn.naive_bayes import GaussianNB    
def NBmodel(data):
    nb = GaussianNB()
    model = nb.fit(data[0:1000,0:feature-1], data[0:1000,feature])
    accuracy = metrics.accuracy_score(data[1000:1100,feature], model.predict(data[1000:1100,0:feature-1]))
    print("NB testdata accuracy : ",accuracy)
    
from sklearn.neighbors import KNeighborsClassifier
def KNNmodel(data):
    knn = KNeighborsClassifier()
    model = knn.fit(data[0:1000,0:feature-1], data[0:1000,feature])
    accuracy = metrics.accuracy_score(data[1000:1100,feature], model.predict(data[1000:1100,0:feature-1]))
    print("KNN testdata accuracy : ",accuracy)

    
if __name__ == '__main__':
    data = createdata()
    DTmodel(data)
    NBmodel(data)
    KNNmodel(data)