# from new_new_Move_Class import SCS_Control
import numpy as np
import matplotlib.pyplot as plt
import time
# Move = SCS_Control().Move  #pos,speed,acc,speed_change_array
A_range = [0, 678]
mid_position = 339
ver = 10
goal_position = 300
a = 200
def tri_generator(goal_positiona,a):
    T = np.arange(100*np.sqrt(16*goal_position/a))
    T = (T/100)
    V = np.zeros([len(T)])
    X = np.zeros([len(T)])
    change_point = np.zeros([int((T.shape[0])/5)])
    change_speed = np.zeros([int((T.shape[0])/5)])
    
    for i in range(len(T)):
        if i <= int(X.shape[0]*0.25):
            V[i] = a*T[i]
        elif i <= int(X.shape[0]*0.75):
            V[i] = V[int(X.shape[0]*0.25)] - a*(T[i] - T[int(X.shape[0]*0.25)])
        else:
            V[i] = V[int(X.shape[0]*0.75)] + a*(T[i] - T[int(X.shape[0]*0.75)])
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

tri_data = np.zeros([1000,120,2])
index = 0
# for i in range(50):
#     for j in range(20):
#         change_point, change_speed, time_list, T, X = tri_generator(150+4*i,150 + 5*j)
#         tri_data[i+j,:len(change_point),:] = np.stack([change_point, change_speed]).T
#         index += 1

# np.save('tri_data_1000.npy',tri_data)


# plt.plot(time_list,change_speed)
# plt.show()
# plt.plot(T, X)
change_point, change_speed, time_list, T, X = tri_generator(350,200)
plt.plot(change_point, change_speed)
plt.title('tri_wave')
plt.xlabel('count_point')
plt.ylabel('speed')
# start = time.time()
# point_a,pos_a = Move(goal_position,1,a,change_point,change_speed)
# mid = time.time()
# poin_b, pos_b = Move(0,1,a,change_point,abs(change_speed))
# end = time.time()
# plt.plot(pos_a)
# plt.plot(pos_b)
# print(end-mid)
# print(mid-start)
# print(end-start)