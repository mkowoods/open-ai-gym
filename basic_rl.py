import gym
import math
import time
import pprint
import random

env = gym.make('CartPole-v0')


def get_state(obs):

    box = 0
    deg  = 2*math.pi/360
    #x is postion along the x_axis(rope) 
    #theta is angle of the pole relative to the base
    x, x_dot, theta, theta_dot = obs
    
    if abs(x) > 2.4 or abs(theta) > 12 * deg:
        return -1
    
    if x < -0.8:
        box = 0
    elif x < 0.8:
        box = 1
    else:
        box = 2
    
    if x_dot < -0.5:
        pass
    elif x_dot < 0.5:
        box += 3
    else:
        box += 6
    
    if theta  < -6 * deg:
        pass
    elif theta < -deg:
        box += 9
    elif theta < 0:
        box += 18
    elif theta < deg:
        box += 27
    elif theta < 6*deg:
        box += 36
    else:
        box += 45
        
    if theta_dot < -50*deg:
        pass
    elif theta_dot < 50*deg:
        box += 54
    else:
        box += 108
        
    return box
    
    
MEMORY = {}
TRANSITIONS = {}
EPISODES = 1
POLICY_PROB = 0.975
GAMMA = 0.9



def get_score(state, action):
    if (state, action) in MEMORY:
        score, ct = MEMORY[(state, action)] 
        return score/ct
    else:
        return random.random()
    
def update_transitions(state, action, state_prime):
    if (state, action) in TRANSITIONS:
        TRANSITIONS[(state, action)].setdefault(state_prime, 0)
        TRANSITIONS[(state, action)] += 1
    else:
        TRANSITIONS[(state, action)] = {state_prime : 1}
        




for i in xrange(EPISODES):
    REWARD = 0
    EPISODE = []    
    print 'EPISODE: ', i
    done = False
    ob = env.reset()
    print 'Initital Observation: '
    print ob, get_state(ob)
    prior_state = (ob, get_state(ob))
    while not done:
        env.render()
        
        if random.random() < POLICY_PROB:
            print 'Using Policy'
            ob, state_box = prior_state
            scores = [(get_score(state_box, j), j) for j in range(2)]
            print state_box, scores
            #print MEMORY, scores
            score, action = max(scores)
        else:
            action = env.action_space.sample()

        ob, reward, done, _ = env.step(action)
        box = get_state(ob)
        #print action, ob, reward, done, box
        REWARD += reward
    
        EPISODE.append([prior_state, action, None])
        
        prior_state = (ob, box)
        
        time.sleep(0.1)
    
    print 'Total Reward: ',REWARD    
    
    for frame in EPISODE:
        frame[2] = REWARD
        REWARD -= 1
        
    
    for frame in EPISODE:
        state, action, reward = frame
        arr, box = state
        if (box, action) in MEMORY:
            MEMORY[(box, action)][0] += reward
            MEMORY[(box, action)][1] += 1.0
        else:
            MEMORY[(box, action)] = [reward, 1.0]
    
    
print 'EPISODE'
#pprint.pprint(EPISODE)
#pprint.pprint(MEMORY)

    

env.render(close = True)
    
    