#plotting the initial data set given
from matplotlib import pyplot as plt
arr = [[1,23],[4,45],[10,52],[19,42],[30,48],[35,48.3],[44.2,37.263],[50,50],[55,48.3],[60.21,48.3]]
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

plt.xlim(0, 65)
plt.ylim(0, 65)

for i in arr:
    plt.plot(i[0], i[1], color='blue',marker='o', markerfacecolor='blue', markersize=1)
plt.show()
import numpy as np
import random
#normalisation of data set
Max_x = None
Min_x = None
Min_y = None
Max_y = None

for i in range(len(arr)):
    if Max_x ==  None:
        Max_x = arr[i][0]
    else:
        if Max_x < arr[i][0]:
            Max_x = arr[i][0]
for i in range(len(arr)):
    if Max_y ==  None:
        Max_y = arr[i][1]
    else:
        if Max_y < arr[i][1]:
            Max_y = arr[i][1]
for i in range(len(arr)):
    if Min_x ==  None:
        Min_x = arr[i][0]
    else:
        if Min_x > arr[i][0]:
            Min_x = arr[i][0]
for i in range(len(arr)):
    if Min_y ==  None:
        Min_y = arr[i][1]
    else:
        if Min_y > arr[i][1]:
            Min_y = arr[i][1]
#print(Max_x,Max_y,Min_x,Min_y)
new_arr = []
for i in range(len(arr)):
    new_arr.append([])
    new_arr[i].append((arr[i][0] - Min_x)/(Max_x - Min_x))
    new_arr[i].append((arr[i][1] - Min_y)/(Max_y - Min_y))
#print(new_arr)
#plotting the new normalised data points
for i in new_arr:
    plt.plot(i[0], i[1], color='blue',marker='o', markerfacecolor='blue', markersize=1)
plt.show()
#give_test_data gives us the two matrices from normalised data
#the first matrix A is of all the x-data points after they have been put in the polynomial
#the second matrix b is of all the y-data points from the normalised data set
#M is model complexity
def give_test_data(M):
    A = []
    b = []
    for i in range(len(new_arr)):
        A.append([])
        b.append([])
        b[i].append(new_arr[i][1])
        #print(A)
        for j in range(M):
            A[i].append(new_arr[i][0]**j)
            #print(new_arr[i][0],new_arr[i][0]**j,A,j,i)
    return(A,b)
#phi matrix here refers to the matrix of all the coefficients of our model
def give_phi_matrix(A,b):
    phi_matrix = (np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A)),b))
    return(phi_matrix)
#show_graph creates a graph of a given model(phi matrix) but this is only for polynomial type model
#I have set M=2 as default because we need atleast a straight line with some slope as prediction, 
#here M = 1 will mean just a line parallel to x-axis passing through a poin on y-axis
def show_graph(phi_matrix,M = 2,color = "blue"):
    matrix = np.linspace(0,1000,500)
    for i in range(500):
        x_star = matrix[i]*(.001)
        y_star = None
        for j in range(M):
            if y_star == None:
                y_star = phi_matrix[j][0]*((x_star)**j)
            else:
                y_star += phi_matrix[j][0]*((x_star)**j)
        #print(x_star,y_star)
        plt.plot(x_star,y_star, color=color,marker ="_", markerfacecolor=color, markersize=2)
    for i in new_arr:
        plt.plot(i[0], i[1], color='red',marker='o', markerfacecolor='blue', markersize=5)
        plt.legend([M])
#printing all models
for i in range(1,10):
    color_arr = ["blank","violet","black",'blue','red',"green","yellow","pink","cyan","orange",]
    A,b = give_test_data(i)
    phi_matrix = give_phi_matrix(A,b)
    show_graph(phi_matrix,i,color_arr[i])
plt.show()
#from here you can see visually how when we are using a lower model complexity, we have underfitting problems
#Similarly we have overfitting problems when we go to very high model complexities
#high and low model complexities is a comment relative to the number of data points provided which is 10 in our case 

#error_function takes your polynomial model, the normalised data set and gives out least square error
def error_function(new_arr,phi_matrix,M):
    error = 0
    for i in new_arr:
        x_star = i[0]
        y_star = None
        for j in range(M):
            if y_star == None:
                y_star = phi_matrix[j][0]*((x_star)**j)
            else:
                y_star += phi_matrix[j][0]*((x_star)**j)
    error += (y_star - i[1])**2
    return((1/2)*error)
#printing error vs model complexity to choose suitable model
for i in range(1,10):
    color_arr = ["blank","violet","black",'blue','red',"green","yellow","pink","cyan","orange",]
    A,b = give_test_data(i)
    phi_matrix = give_phi_matrix(A,b)
    error = error_function(new_arr,phi_matrix,i)
    plt.plot(i, error, color='red',marker='2', markerfacecolor='blue', markersize=5)
    plt.legend([i])
plt.show()
#17-02-2023 
#now we create a model in gaussian basis
#muj_maker gives you a matrix of muj used in gaussian basis
#here muj_matrix is created at random from a range

def muj_maker(M,minor = 0,major = 1000):
    #M is degree of freedom
    #major is the maximum you want to go with value of muj
        #minor is the minimum you want to go with value of muj
    matrix = []
    for i in range(M):
        matrix.append((random.randint(minor,major))*(.001))
    return matrix
import math
#plots graph in gaussian basis
def show_graph_gbasis(phi_matrix,muj_matrix,S,M,color = "pink"):
    #S is spatial scale
    matrix = np.linspace(0,1000,1000)
    #third argument is number of points we want to plot
    for j in range(1000):
    #here the range will go upto the number of points we have decided to plot
        x_star = matrix[j]*(.001)
        y_star = None
        for i in range(M):
            if y_star == None:
                y_star = phi_matrix[i][0]*math.exp(-((x_star - muj_matrix[i])**2)/(2*(S**2)))
            else:
                y_star += phi_matrix[i][0]*math.exp(-((x_star - muj_matrix[i])**2)/(2*(S**2)))

        plt.plot(x_star,y_star, color=color,marker='_', markerfacecolor=color, markersize=1 )
    for i in new_arr:
        plt.plot(i[0], i[1], color='red',marker='o', markerfacecolor='blue', markersize=5)
    plt.legend([M])
#give_test_data_gaussian gives us the two matrices from normalised data
#the first matrix A is of all the x-data points after they have been put in the gaussian basis model
#the second matrix b is of all the y-data points from the normalised data set
#M is model complexity

def give_test_data_gaussian(M,muj_matrix,S,new_arr):
    A = []
    b = []
    for i in range(len(new_arr)):
        A.append([])
        b.append([])
        b[i].append(new_arr[i][1])
        #print(A)
        for j in range(M):
            A[i].append(math.exp(-((new_arr[i][0] - muj_matrix[j])**2)/(2*(S**2))))
            #print(new_arr[i][0],new_arr[i][0]**j,A,j,i)
    return(A,b)
#this will gives us the value of S here we are using variance as the S value
def variance(arr):
    S  = 0
    mean = 0
    add = 0
    for i in arr:
        add += i[0]
    mean = add/len(arr)
    for i in arr:
        S += (i[0] - mean)**2
    return(S/(len(arr)-1))
#we are choosing to create our muj matrix from the x-data points from the data set
#works fine till here 18-02-2023
muj_matrix = []
for i in new_arr:
    muj_matrix.append(i[0])
#printing all the gaussian models
for M in range(1,10):
    color_arr = ["blank","violet","black",'blue','red',"green","yellow","pink","cyan","orange",]

    S = variance(new_arr)
    A,b = give_test_data_gaussian(M,muj_matrix,S,new_arr)
    phi_matrix = give_phi_matrix(A,b)
    show_graph_gbasis(phi_matrix,muj_matrix,S,M,color = color_arr[M])
    plt.show()
muj_matrix = []
for i in new_arr:
    muj_matrix.append(i[0])
S = variance(new_arr)
for M in range(1,10):
    A,b = give_test_data_gaussian(M,muj_matrix,S,new_arr)
    phi_matrix = give_phi_matrix(A,b)
    color_arr = ["grey","violet","black",'blue','red',"green","yellow","pink","cyan","orange"]
    element = phi_matrix[M-1][0]
    for j in range(0,10):
        gaussian_plot(element,muj_matrix[j],S,color_arr[M])
        plt.legend([M,j])

        plt.show()
for M in range(1,10):
    A,b = give_test_data_gaussian(M,muj_matrix,S,new_arr)
    phi_matrix = give_phi_matrix(A,b)
    color_arr = ["grey","violet","black",'blue','red',"green","yellow","pink","cyan","orange"]
    element = phi_matrix[M-1][0]

    gaussian_plot(element,muj_matrix[M-1],S,color_arr[M])
    plt.legend([M,j])
gaussian_plot(element,muj_matrix[M],S,color_arr[M])

A,b = give_test_data_gaussian(10,muj_matrix,S,new_arr)
phi_matrix = give_phi_matrix(A,b)
S = S = variance(new_arr)
plot_sum_of_all_gaussian(10,phi_matrix,muj_matrix,S,"orange")
plt.show()

# it produces plot of summation[w[i]*exp(-((x_star - mu[i])**2)/(2*(S**2)))]
def plot_sum_of_all_gaussian(M,phi_matrix,muj_matrix,S,color):
    matrix = np.linspace(0,1000,1000)
    #third argument is number of points we want to plot
    for i in range(1000):
    #here the range will go upto the number of points we have decided to plot
        x_star = matrix[i]*(.001)
        y_star = None
        for j in range(M):
            if y_star == None:
                y_star = phi_matrix[j]*math.exp(-((x_star - muj_matrix[j])**2)/(2*(S**2)))
            else:
                y_star += phi_matrix[j]*math.exp(-((x_star - muj_matrix[j])**2)/(2*(S**2)))

        #print(x_star,y_star)
        plt.plot(x_star,y_star, color=color,marker ="_", markerfacecolor=color, markersize=2)
    for i in new_arr:
        plt.plot(i[0], i[1], color='red',marker='o', markerfacecolor='blue', markersize=5)



# w matrix is called phi_matrix
# u matrix is called muj_matrix
A,b = give_test_data_gaussian(10,muj_matrix,S,new_arr)
phi_matrix = give_phi_matrix(A,b)
S = S = variance(new_arr)
plot_sum_of_all_gaussian(10,phi_matrix,muj_matrix,S,"orange")
plt.show()
# w matrix is called phi_matrix
# u matrix is called muj_matrix
A,b = give_test_data_gaussian(10,muj_matrix,S,new_arr)
phi_matrix = give_phi_matrix(A,b)
S = S = variance(new_arr)
plot_sum_of_all_gaussian(10,phi_matrix,muj_matrix,S,"orange")
i = 10
color_arr = ["g","v",'a','s','d','f','f','g',"green","yellow","pink","cyan","orange",]
A,b = give_test_data(i)
phi_matrix = give_phi_matrix(A,b)
show_graph(phi_matrix,i,color_arr[i])
plt.show()
def error_func_gauss(M,muj_matrix,S,new_arr):
    A,b = give_test_data_gaussian(M,muj_matrix,S,new_arr)
    phi_matrix = give_phi_matrix(A,b)
    error = 0
    for i in new_arr:
        x_star = i[0]
        y_star = None
        for j in range(M):
            if y_star == None:
                y_star = phi_matrix[j][0]*math.exp(-((x_star - muj_matrix[j])**2)/(2*(S**2)))
            else:
                y_star += phi_matrix[j][0]*math.exp(-((x_star - muj_matrix[j])**2)/(2*(S**2)))
    error += (y_star - i[1])**2
    return((1/2)*error)

for M in range(1,11):
    print(error_func_gauss(M,muj_matrix,S,new_arr),M)
    plt.plot(M, error_func_gauss(M,muj_matrix,S,new_arr), color='red',marker='o', markerfacecolor='blue', markersize=5)
plt.plot()
M = 10
color_arr = ["blank","violet","black",'blue','red',"green","yellow","pink","cyan","orange","purple"]

S = variance(new_arr)
A,b = give_test_data_gaussian(M,muj_matrix,S,new_arr)
phi_matrix = give_phi_matrix(A,b)
show_graph_gbasis(phi_matrix,muj_matrix,S,M,color = color_arr[M])
    #plt.legend([M])
plt.show()
M = 8
color_arr = ["blank","violet","black",'blue','red',"green","yellow","pink","cyan","orange","purple"]

S = variance(new_arr)
A,b = give_test_data_gaussian(M,muj_matrix,S,new_arr)
phi_matrix = give_phi_matrix(A,b)
show_graph_gbasis(phi_matrix,muj_matrix,S,M,color = color_arr[M])
    #plt.legend([M])
plt.show()
        
