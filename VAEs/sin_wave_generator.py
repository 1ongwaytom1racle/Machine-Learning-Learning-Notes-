# from new_new_Move_Class import SCS_Control
import numpy as np
import matplotlib.pyplot as plt
import time
from math import pi, sin
# Move = SCS_Control().Move  #pos,speed,acc,speed_change_array
A_range = [0, 678]
mid_position = 339

ver = 10
goal_position = 300

v_max = 200
def sin_generater(goal_position, v_max): #0.05s变一次
    T = np.arange(100*(pi*goal_position)/v_max)
    T = (T/100)
    V = np.zeros([len(T)])
    X = np.zeros([len(T)])
    change_point = np.zeros([int((T.shape[0])/5)])
    change_speed = np.zeros([int((T.shape[0])/5)])

    for i in range(len(T)):
        V[i] = int(v_max*sin(2*pi*(100*T[i]/len(T))))
    
    for i in range(len(T)):
        roll_x = V[i]*0.01
        X[i] = X[i-1] + roll_x
        
    for i in range(change_point.shape[0]):
        change_point[i] = int(X[5*i])
        change_speed[i] = int(V[5*i])
    time_list = np.zeros([int((T.shape[0])/5)])
    for i in range(1, change_point.shape[0]):
        if change_speed[i] == 0:
            roll_time = 0
        else:
            roll_time = (change_point[i] - change_point[i - 1])/(change_speed[i])
        time_list[i] = time_list[i-1] + roll_time
    return change_point, change_speed, time_list, T, X

sin_data = np.zeros([1000,120,2])
index = 0
# for i in range(50):
#     for j in range(20):
#         change_point, change_speed, time_list, T, X = sin_generater(150+4*i,200 + 5*j)
#         sin_data[i+j,:len(change_point),:] = np.stack([change_point, change_speed]).T
#         index += 1
# change_point, change_speed, time_list, T, X = sin_generater(150,200)
# np.save('sin_data_1000.npy',sin_data)
# plt.plot(time_list,change_speed)
# plt.plot(T,V)
# plt.title('t-v')
# plt.show()
# plt.plot(T,X)
# plt.title('t-x')
# plt.show()
change_point, change_speed, time_list, T, X = sin_generater(350,200)
plt.plot(change_point, change_speed)
plt.title('sin_wave')
plt.xlabel('count_point')
plt.ylabel('speed')
# start = time.time()
# point_a,pos_a = Move(goal_position,1,a,change_point,change_speed)
# mid = time.time()
# poin_b, pos_b = Move(0,1,a,change_point,abs(change_speed))
# end = time.time()
# plt.plot(pos_a)
# plt.show()
# plt.plot(pos_b)
# print(end-mid)
# print(mid-start)
# print(end-start)
