import numpy as np
import gym

#Khởi tạo môi trường
env = gym.make("MountainCar-v0")
env.reset()
c_learning_rate = 0.1
c_discount_value = 0.9
num_episode = 1500

#Lưu kết quả
max_reward = -99999
action_list = []

#khởi tạo thời điểm render
num_show_each = 500

#Phân đoạn ra thành 20x20 State
q_table_size = [20,20]
q_segment_size = (env.observation_space.high - env.observation_space.low) / q_table_size

#Convert từ state ban đầu sang state trong bảng q_table
def convert_state(real_state):
    state = (real_state - env.observation_space.low) // q_segment_size
    return tuple(state.astype(np.int32))

#Khởi tạo bảng q_table với giá trị-q_value được cho ngẫu nhiên theo phân phối uniform
q_table = np.random.uniform(low = -2, high = 0, size = q_table_size + [env.action_space.n])

for ep in range(num_episode):
    print("Eps = ", ep)
    current_reward = 0
    current_action_list = []
    
    current_state = convert_state(env.reset())
    
    show = False
    if ep % num_show_each == 0:
        show = True
    
    done = False
    
    while not done:
        
        if show:
            env.render()
        
        action = np.argmax(q_table[current_state])
        current_action_list.append(action) 
        
        current_q_value = q_table[current_state+(action,)]
        next_q_real_state, reward, done, _ = env.step(action=action)
        current_reward += reward
        next_q_state = convert_state(next_q_real_state)
        
        if done:
            if next_q_real_state[0] > env.goal_position:
                print("Found at Ep: {}".format(ep))
                
        else:
            #update q_value    
            new_q_value = (1-c_learning_rate)*current_q_value + c_learning_rate*(reward + c_discount_value*np.max(q_table[next_q_state]))
            
            #update q_table
            q_table[current_state+(action,)] = new_q_value
            current_state = next_q_state
    
    if max_reward < current_reward:
        max_reward = current_reward
        action_list = current_action_list

print("Max Reward = ", max_reward)
print("Action: ", action_list)
    