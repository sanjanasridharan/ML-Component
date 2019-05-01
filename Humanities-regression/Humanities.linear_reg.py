from random import seed
from random import randrange
from csv import reader
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import pandas as pd

def main():
 filename="Humanities.linear_reg.csv"
 df=load_csv(filename)
 for i in range(len(df[0])):
  str_to_float(df,i)
 split=0.6
 calculate(df,split)

def load_csv(filename):
 df=list()
 with open(filename,'r') as file:
   csv_reader = reader(file)
   for row in csv_reader:
     if not row:
       continue
     df.append(row)
 return df

def str_to_float(df,column):
 for row in df:
  row[column]=float(row[column].strip())
  
def train_N_test(df,split):
  train=[]
  train_size=split*len(df)
  df_copy=list(df)
  while len(train) < train_size:
    index=randrange(len(df_copy))
    train.append(df_copy.pop(index))
  return train,df_copy

def coeff(df):
  x=[row[0] for row in df]
  y=[row[1] for row in df]
  x_mean=mean(x);
  y_mean=mean(y);
  b1=cov(x,x_mean,y,y_mean)/var(x,x_mean)
  b0=y_mean-b1*x_mean
  return [b0,b1]

def mean(x):
 return sum(x)/float(len(x))

def cov(x,xmean,y,ymean):
  cova=0.0
  for i in range(len(x)):
    cova += (x[i]-xmean)*(y[i]-ymean)
  return cova

def var(values,xmean):
  return sum([(x-xmean)**2 for x in values])


def linear(train,test):
  predict=[]
  b0,b1=coeff(train)
  for i in test:
   y=b0+b1*i[0]
   predict.append(y)
  
  return predict


def calculate(df,split):
 train,test = train_N_test(df,split)
 test_set=[]
 for row in test:
   row_copy = list(row)
   row_copy[-1] =None
   test_set.append(row_copy)
 predict = linear(train,test_set)
 actual=[row[-1] for row in test]
 x=[row[0] for row in test]
 y=[row[1] for row in test]
 plt.scatter(x,y)
 plt.plot(x,predict)
 plt.show()
 rmse = rmse_cal(actual,predict)
 print(rmse)
 
 

def rmse_cal(actual,predict):
 sum_error=0.0
 for i in range(len(actual)):
   prediction_error = predict[i]-actual[i]
   sum_error += (prediction_error**2)
 mean_error = sum_error/float(len(actual))
 return sqrt(mean_error)

  




main()
