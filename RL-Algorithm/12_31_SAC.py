import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf1
import scipy.stats as st
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from keras.models import Sequential,load_model,Input,Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense,Activation,Concatenate
from SumTree import SumTree
from SumTree import Memory
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import activations


tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')
# tf.random.set_seed(123)

memory_size = 40000
n_actions = 1
n_features = 1
memory = Memory(capacity=memory_size)
action_bound = 400
batch_size = 16
gamma = 0.99
auto_alpha = True

# log_std_min = 2
# log_std_max = 40

log_std_min = 2  #std = 1
# log_std_max = 5.5  #std = 200
log_std_max = 4.6  #std = 100

target_entropy = -800
log_alpha = tf.Variable(0., dtype=tf.float64)
alpha = tf.Variable(0., dtype=tf.float64)
alpha.assign(tf.exp(log_alpha))

def build_critic(index): 
    regularizer = tf.keras.regularizers.l2(0.001)
    inputs = [tf.keras.Input(shape=(1,)),tf.keras.Input(shape=(n_actions,))]
    x = Concatenate(axis=-1)(inputs)
    x2 = tf.keras.layers.Dense(16,kernel_regularizer=regularizer,
                              name = 'c_%d_dense1'%index)(x)
    x2 = tf.keras.layers.Concatenate()([x, x2])
    x2 = tf.keras.layers.Activation(activations.relu)(x2)
    x3 = tf.keras.layers.Dense(16,kernel_regularizer=regularizer,
                               name = 'c_%d_dense2'%index)(x2)
    x3 = tf.keras.layers.Concatenate()([x, x3])
    x4 = tf.keras.layers.Activation(activations.relu)(x3)
    out = Dense(1,name='value')(x4)
    model = tf.keras.Model(inputs, out) 
    return model

def build_actor():
    inputs = tf.keras.Input(shape=(1,))
    regularizer = tf.keras.regularizers.l2(0.001)
    x2 = tf.keras.layers.Dense(16,kernel_regularizer=regularizer,
                              name = 'a_dense1')(inputs)
    x2 = tf.keras.layers.Concatenate()([inputs, x2])
    x2 = tf.keras.layers.Activation(activations.relu)(x2)
    x3 = tf.keras.layers.Dense(16,kernel_regularizer=regularizer,
                               name = 'a_dense2')(x2)
    x3 = tf.keras.layers.Concatenate()([inputs, x3])
    x4 = tf.keras.layers.Activation(activations.relu)(x3)
    actions_mean = Dense(n_actions, name="Out_mean")(x4)
    actions_std = Dense(n_actions, name="Out_std")(x4)
    model = Model(inputs=inputs, outputs=[actions_mean, actions_std])
    return model

def update_target_weights(model, target_model, tau=0.005):
    weights = model.get_weights()
    target_weights = target_model.get_weights()
    for i in range(len(target_weights)):  # set tau% of target model to be new weights
        target_weights[i] = weights[i] * tau + target_weights[i] * (1 - tau)
    target_model.set_weights(target_weights)

def process_actions(means, log_std, test=False, eps=1e-6):
    std = tf.math.exp(log_std)
    raw_actions = means

    if not test:
        sample_constant = tf.cast(tf.random.normal(shape=means.shape), tf.double)  #s_c ~ N(0,1)
        raw_actions += std * sample_constant

    log_prob_u = tfd.Normal(loc=means, scale=std).log_prob(raw_actions) #概率密度
    actions = tf.math.tanh(raw_actions)

    # log_prob = tf.reduce_sum(log_prob_u - tf.math.log(1 - actions ** 2 + eps))
    log_prob = log_prob_u - tf.math.log(1 - actions ** 2 + eps)

    return actions, log_prob #a, entrophy , constant from N(0,1)

def act(state, test=False, use_random=False):
    state = np.expand_dims(state, axis=0).astype(np.float64)

    if use_random:
        a = tf.random.uniform(shape=(1, n_actions), minval=-1, maxval=1, dtype=tf.float64)
    else:
        means, log_stds = actor(state)
        log_stds = tf.clip_by_value(log_stds, log_std_min, log_std_max)

        a, log_prob = process_actions(means, log_stds, test=test)
    return a

def store_transition(s, a, r, dones, s_):   
    transition = np.hstack((s, a, r, dones, s_))
    # random_memory.append(transition)
    memory.store(transition)
    
def s_norm(s):
    s = float(s)
    s_norm = 2*(s/4096) - 1
    return s_norm

def load_pre_memory():
    pre_memory = np.load('12_26_test_memory_backup_40000.npy')
    for i in range(memory_size):
        memory.store(pre_memory[i,:])
        
def learn():
    tree_idx, batch_memory, ISWeight = memory.sample(batch_size)
    ISWeight /= ISWeight.min()
    ISWeight = ISWeight.reshape(-1,1)
    states = batch_memory[:, :n_features]
    next_states = batch_memory[:, -n_features:]
    actions = batch_memory[:, n_features]
    rewards = batch_memory[:, n_features + 1]
    dones = batch_memory[:, n_features + 2]
    dones = dones.reshape(-1,1)
    rewards = rewards.reshape(-1,1)
    
    actor_loss = 0  #for plot

    with tf.GradientTape(persistent=True) as tape:
        # next state action log probs
        # means log_stds refer to next action
        means, log_stds = actor_tg(next_states)
        log_stds = tf.clip_by_value(log_stds, log_std_min, log_std_max)
        next_actions, log_probs = process_actions(means, log_stds)
        # log_probs_np = np.array(log_probs)

        # critics loss
        current_q_1 = critic_1([states, actions])
        current_q_2 = critic_2([states, actions])
        current_q_min = tf.math.minimum(current_q_1, current_q_2)
        # q2_np = np.array(current_q_2)
        next_q_1 = critic_tg_1([next_states, next_actions])
        next_q_2 = critic_tg_2([next_states, next_actions])
        next_q_min = tf.math.minimum(next_q_1, next_q_2)
        state_values = next_q_min - alpha * log_probs
        # state_values_np = np.array(state_values)
        # Is_test = np.array([1 ,0, 1, 0, 0]).reshape(-1,1)
        target_qs = tf.stop_gradient(rewards + (1 - dones) * state_values * gamma)
        # tg_q_np = np.array(target_qs)
        # loss_np = np.array(tf.math.square(current_q_1 - target_qs))
        # Is_loss = loss_np * Is_test
        critic_loss_1 = tf.reduce_mean(0.5 * ISWeight * tf.math.square(current_q_1 - target_qs))
                        # tf.math.reduce_sum(critic_1.get_layer(name = 'c_1_dense1').losses) + \
                        # tf.math.reduce_sum(critic_1.get_layer(name = 'c_1_dense2').losses)
                        
        # c_loss_1 = np.array(critic_loss_1)
        critic_loss_2 = tf.reduce_mean(0.5 * ISWeight * tf.math.square(current_q_2 - target_qs))
                        # tf.math.reduce_sum(critic_2.get_layer(name = 'c_2_dense1').losses) + \
                        # tf.math.reduce_sum(critic_2.get_layer(name = 'c_2_dense2').losses)

        # current state action log probs
        # replace means, log_stds, now they refer to current action
        means, log_stds = actor(states)
        log_stds = tf.clip_by_value(log_stds, log_std_min, log_std_max)
        actions, log_probs = process_actions(means, log_stds)
        # log_probs_np = np.array(log_probs)

        # actor loss
        if actor_update:        
            current_q_1 = critic_1([states, actions])
            current_q_2 = critic_2([states, actions])
            current_q_min = tf.math.minimum(current_q_1, current_q_2)
            # current_q_min_np = np.array(current_q_min)
            actor_loss = tf.reduce_mean(ISWeight * (alpha * log_probs - current_q_min)) 

        # actor_loss_np =np.array(ISWeight * (alpha * log_probs - current_q_min))

        # temperature loss
        if auto_alpha:
            # alpha_loss = -tf.reduce_mean(
            #     (log_alpha * tf.stop_gradient(log_probs + target_entropy)))
            alpha_loss = -tf.reduce_mean(
                (tf.exp(log_alpha) * tf.stop_gradient(log_probs + target_entropy)))  #other code

    critic_grad = tape.gradient(critic_loss_1, critic_1.trainable_variables)  # compute actor gradient
    critic_optimizer_1.apply_gradients(zip(critic_grad, critic_1.trainable_variables))

    critic_grad = tape.gradient(critic_loss_2, critic_2.trainable_variables)  # compute actor gradient
    critic_optimizer_2.apply_gradients(zip(critic_grad, critic_2.trainable_variables))
    
    if actor_update:        
        actor_grad = tape.gradient(actor_loss, actor.trainable_variables)  # compute actor gradient
        actor_optimizer.apply_gradients(zip(actor_grad, actor.trainable_variables))  
    
    if auto_alpha:
            # optimize temperature
            alpha_grad = tape.gradient(alpha_loss, [log_alpha])
            alpha_optimizer.apply_gradients(zip(alpha_grad, [log_alpha]))
            alpha.assign(tf.exp(log_alpha))
    #update SumTree
    current_q_mean = 0.5 * (current_q_1 + current_q_2)
    abs_errors = tf.reduce_sum(tf.abs(target_qs - current_q_mean), axis=1)
    memory.batch_update(tree_idx, abs_errors)
    
    return np.array(alpha), np.array(critic_loss_1), \
           np.array(critic_loss_2), np.array(actor_loss), \
           np.array(alpha * log_probs).mean(), np.array(next_q_min).mean(), np.array(current_q_min).mean()
           
def Q_plot(Q_array, point):
    # plt.figure(figsize = (20,15))
    plt.plot(Q_array[:, 0],
             label = 'eval_Q%d'%point,linewidth=4)
    plt.plot(Q_array[:, 1],label = 'target_Q%d'%point,
             linewidth=2,linestyle=':')
        
    # plt.plot(np.arange(Q_array.shape[0]),300*np.ones(Q_array.shape[0]),label='ture Q')
    plt.xlabel('steps')
    plt.title('point_%d'%point)
    plt.legend()
    # plt.savefig('Figure/op_Q_visualized_11_17%d.png'%learn_step_counter)
    # plt.show()
    
def action_plot(Q_array, point):
    action_mean = np.zeros([final_steps, 1])
    for i in range(Q_array.shape[0]):
        action_mean[i, 0] = action_bound * np.tanh(Q_array[i, 2])
    plt.plot(action_mean[:, 0],
             label = 'a_mean%d'%point,linewidth=4)
    # plt.plot(Q_array[:, 3],label = 'a_std%d'%point,
    #          linewidth=2,linestyle=':')
        
    # plt.plot(np.arange(Q_array.shape[0]),300*np.ones(Q_array.shape[0]),label='ture Q')
    plt.xlabel('steps')
    plt.title('action_mean')
    plt.legend()

def action_std_plot(Q_array, point):
    action_std = np.zeros([final_steps, 1])
    action_mean = np.zeros([final_steps, 1])
    for i in range(Q_array.shape[0]):
        action_std[i, 0] = np.exp(Q_array[i, 3])
        action_mean[i, 0] = Q_array[i, 2]
        action_std[i, 0] = action_mean[i, 0] + 2 * action_std[i, 0]  # 2 ge biao zhun cha
        action_std[i, 0] = action_bound * np.tanh(action_std[i, 0]) - action_bound * np.tanh(Q_array[i, 2])
    plt.plot(action_std[:, 0],label = 'a_std%d'%point,
             linewidth=2,linestyle=':')
        
    # plt.plot(np.arange(Q_array.shape[0]),300*np.ones(Q_array.shape[0]),label='ture Q')
    plt.xlabel('steps')
    plt.title('action_2std')
    plt.legend()
    
def reward_plot():
    reward_plt = []
    x = np.arange(reward_his.shape[0])
    moving_reward = 0
    for i in range(reward_his.shape[0]):
        moving_reward += reward_his[i, 0]
        reward_plt.append(moving_reward)
    plt.figure(figsize = (12,8))
    plt.plot(x,reward_plt)   
    plt.xlabel('steps')
    plt.ylabel('accum_reward')
    plt.title('reward_record')
    # plt.savefig('Figure/op_acc_r_11_17_2_%d.png'%learn_step_counter)
    plt.show()  

def alpha_plot():
    plt.figure(figsize = (12,8))
    plt.plot(alpha_his)
    plt.title('alpha_change')
    plt.xlabel('steps')
    plt.ylabel('alpha')
    plt.show()

def loss_plot():
    plt.figure(figsize = (12,8))
    plt.plot(loss_his[:, 0], label = 'critic_1')
    plt.plot(loss_his[:, 1], label = 'critic_2')
    plt.title('critic_loss')
    plt.xlabel('steps')
    # plt.ylabel('alpha')
    plt.show()
    # actor_loss = []
    # for i in range(final_steps):
    #     if loss_his[i, 2] > 0.01:
    #         actor_loss.append(loss_his[i, 2])
   
    plt.figure(figsize = (12,8))
    plt.plot(loss_his[:, 2], label = 'actor')
    plt.title('actor_loss')
    plt.xlabel('steps')
    plt.show()
    
def entropy_plot():
    plt.figure(figsize = (12,8))
    plt.plot(loss_his[:, 3], label = 'entropy')
    plt.plot(loss_his[:, 4], label = 'next_q')
    plt.plot(loss_his[:, 5], label = 'current_q')
    plt.title('entropy_compare')
    plt.xlabel('steps')
    plt.legend()
    plt.show()


def learning_rate_schedules_op(starter_learning_rate,
                               end_learning_rate,
                               decay_steps,
                               power):
    starter_learning_rate = starter_learning_rate
    end_learning_rate = end_learning_rate
    decay_steps = decay_steps
    p = power
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power = p)
    my_op = tf.optimizers.Adam(learning_rate = learning_rate_fn)
    return my_op

def same_initial_model():
    critic_1.load_weights('12_27_new_net_c1')
    critic_2.load_weights('12_27_new_net_c2')
    actor.load_weights('12_27_new_net_a')
    
load_pre_memory()

critic_1 = build_critic(1)
critic_2 = build_critic(2)
critic_tg_1 = build_critic(3)
critic_tg_2 = build_critic(4)
actor = build_actor()
actor_tg = build_actor()
# critic_1.save_weights('12_27_new_net_c1')
# critic_2.save_weights('12_27_new_net_c2')
# actor.save_weights('12_27_new_net_a')
# same_initial_model()  #keep model same through all algorithm
update_target_weights(critic_1, critic_tg_1, tau=1.)
update_target_weights(critic_2, critic_tg_2, tau=1.)
update_target_weights(actor, actor_tg, tau=1.)
my_op = tf.optimizers.Adam(learning_rate = 0.0000001)
#  warm up
critic_optimizer_1 = my_op  #
critic_optimizer_2 = my_op
actor_optimizer = my_op  #
alpha_optimizer = my_op #
# alpha_optimizer = my_op

s = 2048
steps = 0
warm_up_steps = 100
final_steps = 200000
# memory_backup = np.zeros((final_steps, 1 * 2 + 3))  #(s,a,r,dones,s')


Q_1000_his = np.zeros([final_steps, 4])  #0:q  1:tg_q 2:a 3:a_std
Q_2000_his = np.zeros([final_steps, 4])  #0:q  1:tg_q 2:a 3:a_std
Q_3250_his = np.zeros([final_steps, 4])  #0:q  1:tg_q 2:a 3:a_std
Q_4000_his = np.zeros([final_steps, 4])  #0:q  1:tg_q 2:a 3:a_std
Q_4090_his = np.zeros([final_steps, 4])  #0:q  1:tg_q 2:a 3:a_std

reward_his = np.zeros([final_steps, 1])
alpha_his = np.zeros([final_steps, 1])
loss_his = np.zeros([final_steps, 6])
# entropy_his = np.zeros([final_steps, 3])
# auto_alpha = False
auto_alpha = True
actor_update = True

while 1:
    print('\r Process {:.0%}'.format((steps/final_steps)), end = '', flush= True)
    # if np.random.random() <= 0.5:
    #     a = np.array(act(s_norm(s))).mean() 
    # else:
    #     a = 2*np.random.random(1) - 1
    a = np.array(act(s_norm(s))).mean()  #a~(-1,1)
    s_ = s + a * action_bound  
    dones = 0
    if steps % 10 == 0 and steps != 0:
        dones = 1
    if s_ < 0 or s_ > 4095:
        r = -1
        store_transition(s_norm(s), a, r, dones, s_norm(s_))
        s_ = 2048  #get pulished and back to start place
    elif s_ >= 1800 and s_ <= 2000:
        r = 1
        store_transition(s_norm(s), a, r, dones, s_norm(s_))
    elif s_ >= 4000 and s_ <= 4050:
        r = 10
        store_transition(s_norm(s), a, r, dones, s_norm(s_))
    else:
        r = -0.01
        store_transition(s_norm(s), a, r, dones, s_norm(s_))
    
    # memory_backup[steps,:] = np.hstack((s_norm(s), a, r, dones, s_norm(s_)))
    alpha_np, c1_loss_np, c2_loss_np, a_loss_np, entorpy_np, next_q_np, now_q_np  = learn()
    update_target_weights(critic_1, critic_tg_1)  # iterates target model
    update_target_weights(critic_2, critic_tg_2)   
    update_target_weights(actor, actor_tg)    
    
    reward_his[steps, :] = r  
    alpha_his[steps, :] = alpha_np  
    loss_his[steps, :3] = [c1_loss_np, c2_loss_np, a_loss_np]
    loss_his[steps, 3:] = [entorpy_np, next_q_np, now_q_np]

    Q_1000_his[steps, :] = [critic_1([np.array(s_norm(1000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]), 
                            critic_tg_1([np.array(s_norm(1000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]),
                            actor(np.array(s_norm(1000)).reshape(-1,1))[0],
                            actor(np.array(s_norm(1000)).reshape(-1,1))[1]]
    Q_2000_his[steps, :] = [critic_1([np.array(s_norm(2000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]), 
                            critic_tg_1([np.array(s_norm(2000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]),
                            actor(np.array(s_norm(2000)).reshape(-1,1))[0],
                            actor(np.array(s_norm(2000)).reshape(-1,1))[1]]
    Q_3250_his[steps, :] = [critic_1([np.array(s_norm(3250)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]), 
                            critic_tg_1([np.array(s_norm(3250)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]),
                            actor(np.array(s_norm(3250)).reshape(-1,1))[0],
                            actor(np.array(s_norm(3250)).reshape(-1,1))[1]]
    Q_4000_his[steps, :] = [critic_1([np.array(s_norm(4000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]), 
                            critic_tg_1([np.array(s_norm(4000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]),
                            actor(np.array(s_norm(4000)).reshape(-1,1))[0],
                            actor(np.array(s_norm(4000)).reshape(-1,1))[1]]
    Q_4090_his[steps, :] = [critic_1([np.array(s_norm(4090)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]), 
                            critic_tg_1([np.array(s_norm(4090)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]),
                            actor(np.array(s_norm(4090)).reshape(-1,1))[0],
                            actor(np.array(s_norm(4090)).reshape(-1,1))[1]]
    
            
    steps += 1
    s = s_  #update state
    if dones:
        s =2048  #new episode
    if steps >= warm_up_steps:
        break

# critic_optimizer_1 = learning_rate_schedules_op(0.0001, 0.00001, 80000, 0.5)  #BASELINE!!
# critic_optimizer_2 = learning_rate_schedules_op(0.0001, 0.00001, 80000, 0.5)
# actor_optimizer = learning_rate_schedules_op(0.00005, 0.000001, 80000, 0.5)  
# alpha_optimizer = learning_rate_schedules_op(0.00005, 0.000001, 80000, 0.5) 
# alpha_optimizer = my_op
critic_optimizer_1 = learning_rate_schedules_op(0.0001, 0.0005, 80000, 0.5)  #BASELINE!!
critic_optimizer_2 = learning_rate_schedules_op(0.0001, 0.0005, 80000, 0.5)
actor_optimizer = learning_rate_schedules_op(0.000001, 0.00005, 80000, 0.5)  
alpha_optimizer = learning_rate_schedules_op(0.000001, 0.00005, 80000, 0.5) 

while 1:
    print('\r Process {:.0%}'.format((steps/final_steps)), end = '', flush= True)
    # if np.random.random() <= 0.5:
    #     a = np.array(act(s_norm(s))).mean() 
    # else:
    #     a = 2*np.random.random(1) - 1
    a = np.array(act(s_norm(s))).mean()  #a~(-1,1)
    s_ = s + a * action_bound  
    dones = 0
    if steps % 10 == 0 and steps != 0:
        dones = 1
    if s_ < 0 or s_ > 4095:
        r = -1
        store_transition(s_norm(s), a, r, dones, s_norm(s_))
        s_ = 2048  #get pulished and back to start place
    elif s_ >= 1800 and s_ <= 2000:
        r = 1
        store_transition(s_norm(s), a, r, dones, s_norm(s_))
    elif s_ >= 4000 and s_ <= 4050:
        r = 10
        store_transition(s_norm(s), a, r, dones, s_norm(s_))
    else:
        r = -0.01
        store_transition(s_norm(s), a, r, dones, s_norm(s_))
    
    # memory_backup[steps,:] = np.hstack((s_norm(s), a, r, dones, s_norm(s_)))
    alpha_np, c1_loss_np, c2_loss_np, a_loss_np, entorpy_np, next_q_np, now_q_np  = learn()
    update_target_weights(critic_1, critic_tg_1)  # iterates target model
    update_target_weights(critic_2, critic_tg_2)   
    update_target_weights(actor, actor_tg)    
    
    reward_his[steps, :] = r  
    alpha_his[steps, :] = alpha_np  
    loss_his[steps, :3] = [c1_loss_np, c2_loss_np, a_loss_np]
    loss_his[steps, 3:] = [entorpy_np, next_q_np, now_q_np]

    Q_1000_his[steps, :] = [critic_1([np.array(s_norm(1000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]), 
                            critic_tg_1([np.array(s_norm(1000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]),
                            actor(np.array(s_norm(1000)).reshape(-1,1))[0],
                            actor(np.array(s_norm(1000)).reshape(-1,1))[1]]
    Q_2000_his[steps, :] = [critic_1([np.array(s_norm(2000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]), 
                            critic_tg_1([np.array(s_norm(2000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]),
                            actor(np.array(s_norm(2000)).reshape(-1,1))[0],
                            actor(np.array(s_norm(2000)).reshape(-1,1))[1]]
    Q_3250_his[steps, :] = [critic_1([np.array(s_norm(3250)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]), 
                            critic_tg_1([np.array(s_norm(3250)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]),
                            actor(np.array(s_norm(3250)).reshape(-1,1))[0],
                            actor(np.array(s_norm(3250)).reshape(-1,1))[1]]
    Q_4000_his[steps, :] = [critic_1([np.array(s_norm(4000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]), 
                            critic_tg_1([np.array(s_norm(4000)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]),
                            actor(np.array(s_norm(4000)).reshape(-1,1))[0],
                            actor(np.array(s_norm(4000)).reshape(-1,1))[1]]
    Q_4090_his[steps, :] = [critic_1([np.array(s_norm(4090)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]), 
                            critic_tg_1([np.array(s_norm(4090)).reshape(-1,1),
                            np.array(0.0).reshape(-1,1)]),
                            actor(np.array(s_norm(4090)).reshape(-1,1))[0],
                            actor(np.array(s_norm(4090)).reshape(-1,1))[1]]
    
            
    steps += 1
    s = s_  #update state
    if dones:
        s =2048  #new episode
    if steps >= final_steps:
        break






# np.save('12_26_test_memory_backup_40000.npy', memory_backup)
reward_plot()
alpha_plot()
loss_plot()
entropy_plot()
plt.figure(figsize = (20,15))
Q_plot(Q_1000_his, 1000)
Q_plot(Q_2000_his, 2000)
Q_plot(Q_3250_his, 3250)
Q_plot(Q_4000_his, 4000)
Q_plot(Q_4090_his, 4090)
plt.show()
plt.figure(figsize = (20,15))
action_plot(Q_1000_his, 1000)
action_plot(Q_2000_his, 2000)
action_plot(Q_3250_his, 3250)
action_plot(Q_4000_his, 4000)
action_plot(Q_4090_his, 4090)
plt.show()
plt.figure(figsize = (20,15))
action_std_plot(Q_1000_his, 1000)
action_std_plot(Q_2000_his, 2000)
action_std_plot(Q_3250_his, 3250)
action_std_plot(Q_4000_his, 4000)
action_std_plot(Q_4090_his, 4090)
plt.show()