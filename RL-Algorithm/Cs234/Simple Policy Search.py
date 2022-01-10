import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential
from tensorflow.keras.layers import Dense
from Game import Game
env = Game()

from sklearn import preprocessing

label = preprocessing.LabelEncoder()
one_hot = preprocessing.OneHotEncoder(sparse = False)


def build_net():
    eval_net = Sequential([
        Dense(4,activation='tanh',input_shape = (2,),name='el1'),
        Dense(len(env.actions),activation='softmax',name='el2')
        ])       

    return eval_net

def feature(state):
    y = [state,state]
    return np.array([y]).reshape(1,-1)

def choose_action(observation):
    a_prob = np.array(eval_net(observation))
    action = np.random.choice(range(a_prob.shape[1]),p = a_prob.ravel())
    return action

def discount_and_norm_rewards(r_his):
        # discount episode rewards
        discounted_r_his = np.zeros_like(r_his)
        running_add = 0
        for t in reversed(range(0, len(r_his))):
            running_add = running_add * env.gamma + r_his[t]
            discounted_r_his[t] = running_add

        # normalize episode rewards
        discounted_r_his = np.array(discounted_r_his)
        discounted_r_his -= np.mean(discounted_r_his)
        discounted_r_his /= np.std(discounted_r_his)
        return discounted_r_his

# def my_loss(y_true, y_pred):
#         neg_log_prob = tf.reduce_sum(-tf.log(all_act_prob)*tf.one_hot(tf_acts, n_actions), axis=1)
#         loss = tf.reduce_mean(neg_log_prob * tf_vt)  

def learn():
    global a_his
    a_his = one_hot.fit_transform(a_his)
    dis_count_r = discount_and_norm_rewards(r_his) #负号！
    feed_x = np.array(s_his)
    eval_net.train_on_batch(feed_x,a_his,sample_weight = dis_count_r)
    
    


eval_net = build_net()
# episode = 1
step_his = []
eval_net.compile(loss=tf.keras.losses.CategoricalCrossentropy() ,optimizer='adam') 



episode = 10
for i in range(episode):
    current_state = 0
    total_steps = 0 
    s_his = []
    a_his = []
    r_his = []         
    while True:
        current_action = choose_action(feature(current_state))
        done,reward,next_state = env.get_next_state(current_state, current_action)
        next_action = choose_action(feature(next_state))
        a_his.append(current_action),s_his.append(feature(current_state)),r_his.append(reward)

        
        current_state = next_state

        env.update_env(current_state) # 环境相关
        if done:
            break
        total_steps += 1          # 环境相关
    step_his.append(total_steps)
    r_his = np.array(r_his).astype('float64')
    test_r = discount_and_norm_rewards(r_his)
    a_his = np.array(a_his).astype('float64').reshape(-1, 1)
    # a_his = one_hot.fit_transform(a_his)
    s_his = np.array(s_his).astype('float64')
    s_his = s_his.reshape(s_his.shape[0],s_his.shape[2])
    learn()
    
    
    
for i in range(episode):
    print('\rEpisode {}: total_steps = {}'.format(i, step_his[i])) 

prob_table = np.zeros((len(env.states),len(env.actions)))
for i in range(len(env.states)):
    prob_table[i,:] = eval_net(feature(i))
    
print('\nprob_table:',prob_table)    
plt.plot(np.arange(episode),step_his)
plt.xlabel('episode')
plt.ylabel('steps')
    
    
    
    
    
    
    
    

