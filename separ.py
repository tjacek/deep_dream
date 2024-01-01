import numpy as np
import matplotlib.pyplot as plt

def gen_points(n,C=3):
    return C*(np.random.random(size=(n,2))-0.5)

def get_y(points,r=1):
    r2=r*r
    y=[int(np.sum(point_i*point_i)<r2)
           for point_i in points]
    return np.array(y)

def polar_cord(X):
    polar=[]
    for x_i,y_i in X:
        r_i=np.sqrt(x_i**2+y_i**2)
        φ=np.arctan2(x_i,y_i)
        polar.append([r_i,φ])
    return np.array(polar)	          

def show(X,y,xlabel='x',ylabel='y'):
    plt.clf()
    X_zero=X[y==0]
    plt.scatter(X_zero[:,0], X_zero[:,1])
    X_one=X[y==1]
    plt.scatter(X_one[:,0], X_one[:,1])
    plt.xlabel(xlabel,fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    plt.show()

X=gen_points(200)
y=get_y(X)
show(X,y)

X_polar=polar_cord(X)
show(X_polar,y,xlabel='r',ylabel='φ')