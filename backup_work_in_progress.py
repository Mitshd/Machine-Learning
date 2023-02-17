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
#normalisation
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
print(Max_x,Max_y,Min_x,Min_y)
new_arr = []
for i in range(len(arr)):
    new_arr.append([])
    new_arr[i].append((arr[i][0] - Min_x)/(Max_x - Min_x))
    new_arr[i].append((arr[i][1] - Min_y)/(Max_y - Min_y))
print(new_arr)
#plotting normalised points
plt.rcParams["figure.figsize"] = [10.00, 10]
plt.rcParams["figure.autolayout"] = True

plt.xlim(0, 1.2)
plt.ylim(0, 1.2)

for i in new_arr:
    plt.plot(i[0], i[1], color='blue',marker='o', markerfacecolor='blue', markersize=5)
plt.show()
