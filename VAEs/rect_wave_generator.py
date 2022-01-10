# from new_new_Move_Class import SCS_Control
import numpy as np
import matplotlib.pyplot as plt
import time
# Move = SCS_Control().Move  #pos,speed,acc,speed_change_array
A_range = [0, 678]
mid_position = 339

ver = 10
goal_position = 300  #v_max/a <T/4
a = 250
v_max = 200
def rect_generater(goal_position, a, v_max):
    T = np.arange(100*(goal_position + v_max*v_max/a)*2/v_max)
    T = (T/100)
    V = np.zeros([len(T)])
    X = np.zeros([len(T)])
    change_point = np.zeros([int((T.shape[0])/5)])
    change_speed = np.zeros([int((T.shape[0])/5)])
    aaa = int(X.shape[0]*(1 - v_max/(a*0.01*len(T))))
    for i in range(len(T)):
        if i <= int(X.shape[0]*(v_max/(a*0.01*len(T)))):
            V[i] = a*T[i]
        elif i <= int(X.shape[0]*(0.5 - v_max/(a*0.01*len(T)))):
            V[i] = v_max
        elif i <= int(X.shape[0]*(0.5 + v_max/(a*0.01*len(T)))):
            V[i] = v_max - a*(T[i] - T[int(X.shape[0]*(0.5 - v_max/(a*0.01*len(T))))])
        elif i <= int(X.shape[0]*(1 - v_max/(a*0.01*len(T)))):
            V[i] = -v_max
        else:
            V[i] = -v_max + a*(T[i] - T[int(X.shape[0]*(1 - v_max/(a*0.01*len(T))))])
    
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

# rect_data = np.zeros([1000,120,2])
# index = 0
# for i in range(10):
#     for j in range(20):
#         for k in range(5): 
#             change_point, change_speed, time_list, T, X = rect_generater(150 + 20*i,200 + 5*j, 150 + 10*k)
#             rect_data[index,:len(change_point),:] = np.stack([change_point, change_speed]).T
#             index += 1
change_point, change_speed, time_list, T, X = rect_generater(300,200, 150)
# np.save('rect_data_1000.npy',rect_data)
# plt.plot(time_list,change_speed)
# plt.plot(T,V)
# plt.show()
# plt.plot(T,X)
# plt.show()
plt.plot(change_point, change_speed)
plt.title('rect_wave')
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
