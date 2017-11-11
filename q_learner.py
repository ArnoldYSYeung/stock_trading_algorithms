## Author:     Arnold YS Yeung (http://www.arnoldyeung.com)
## Date:       October 28th, 2017
## This contains a Q-Learner class for training and querying a Q-table.
##  
## 2017-11-11: Completed and tested qLearner class in a maze-solving application
##              (see test_q_learner.py) - dyna not implemented (Yes. I have been lazy/busy recently.)
## 2017-11-11: Added dyna functionalities.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as skl
import random as rdm

class qLearner():

    def __init__(self,  \
                 num_states,        #   number of states to consider        \            
                 num_actions,       #   number of actions available         \            
                 alpha = 0.2,       #   learning rate (0 to 1.0)            \            
                 gamma = 0.9,       #   discount rate (0 to 1.0)            \            
                 rar = 0.5,         #   random action rate (0 to 1.0)       \           
                 radr = 0.99,       #   random action decay rate (0 to 1.0) \            
                 dyna = 0,          \
                 verbose = True):
        #   define input properties
        self.num_states = num_states;
        self.num_actions = num_actions;
        self.alpha = alpha;
        self.gamma = gamma;
        self.rar = rar;
        self.radr = radr;
        self.dyna = dyna;
        self.verbose = verbose;

        #   initial state and action
        self.s = 0;
        self.a = 0;

        #   Q[s,a] = immediate reward + discounted reward for s,a
        self.Q = np.zeros((self.num_states, self.num_actions));
        #   This Q-table contains the Q-value for state x action

        if dyna > 0:
            
            small_num = 0.000001;       #   start with small number to prevent divide by 0

            #   Tc[s, a, s'] = count number of occurrences of ending in s_prime given
            #   s and a
            self.Tc = np.ones((self.num_states, self.num_actions, self.num_states))*small_num;

            #   T[s, a, s'] = probability of ending in s_prime given s and a
            #   Divide by num_states so all states have equally probability initially
            self.T = np.ones((self.num_states, self.num_actions, self.num_states)) / num_states;
            #   This T-table (transition matrix) contains the probability for
            #   state x action x next_state

            #   R[s, a] = expected reward for s and a
            self.R = np.zeros((self.num_states, self.num_actions));

    def best_action(self, s):
    #   Returns best action (policy) to maximize sum of immediate and discounted
    #   reward
        return np.argmax(self.Q[s, :]);

    def weighted_choice(self, weights):
    #   Select an index based on weighted probabilities
    #   From "Simple Linear Approach" here:
    #               https://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python/
    
        totals = [];
        running_total = 0;

        for w in weights:
            #print("w: ", w);
            running_total += w;
            totals.append(running_total);
            #print("totals: ", totals);

        #print("running_total: ",running_total);
        rnd = rdm.random() * running_total;
        #print("Random: ", rnd);
        
        for i in range(len(totals)):
            total = totals[i];
            #print("total: ", total);
            if rnd < total:
                #print("Random index chosen: ",i);
                return i;
                
    def query(self, s_prime, r):
    #   Keeps previous state (s) and action (a) and updates q_table based on
    #   the inputted new state (s_prime) and reward (r) with a new experience
    #   tuple <s, a, s_prime, r>.  Returns the next action (best or random).
    #   Note that in reality, s_prime and r are observed from state s after
    #   action a is taken.  This function is a training function.
    #   For example, s = 1, a = D, s_prime = 4, reward = 102.
    #   We take action D while in state 1.  We observe the outcome to be
    #   state 4 with a reward of 102. The calculated next action to take
    #   is returned and based solely on state 4.

        prev_state = self.s;
        prev_action = self.a;
        new_state = s_prime;
        reward = r;

        if rdm.uniform(0, 1) < self.rar:             #   if random action is selected
            next_action = rdm.randint(0, self.num_actions-1);       #   select random action
            #print('Random action selected...');
        else:
            next_action = self.best_action(new_state);
            #print('Best action selected using policy...');

        #   update Q-table with Q[s,a] where s = prev_state, a = prev_action
        self.Q[prev_state, prev_action] = (1-self.alpha)*self.Q[prev_state, prev_action] \
                                          + self.alpha * (reward + self.gamma * \
                                                          self.Q[new_state, self.best_action(new_state)]);

        if self.dyna > 0:
            #   increment Tc-table (count the occurrence)
            self.Tc[prev_state, prev_action, new_state] += 1;

            count = np.sum(self.Tc[prev_state,prev_action]);            #   total count of prev_state, prev_action

            #   update T-table - all counts involving s and a needs to be updated for s_prime
            for h in range(0, self.num_states):

                self.T[prev_state, prev_action, h] = self.Tc[prev_state, prev_action, h] \
                                                     / count;
            
            #   update R-table
            self.R[prev_state, prev_action] = (1-self.alpha)*self.R[prev_state, prev_action] \
                                              + self.alpha * reward;

            #   hallucinate experience
            for i in range(0, self.dyna):
                dyna_s = rdm.randint(0, self.num_states-1);                             #   randomly select s
                dyna_a = rdm.randint(0, self.num_actions-1);                            #   randomly select a
                dyna_s_prime = self.weighted_choice(self.T[dyna_s, dyna_a, :]);    #   decide on next state
                #print("Dyna_s_prime: ", dyna_s_prime)

                dyna_reward = self.R[dyna_s,dyna_a];

                #   update Q-table with Q[s,a] where s = prev_state, a = prev_action
                self.Q[dyna_s, dyna_a] = (1-self.alpha)*self.Q[dyna_s, dyna_a] \
                                          + self.alpha * (dyna_reward + self.gamma * \
                                                            self.Q[dyna_s_prime, self.best_action(dyna_s_prime)]);

        self.rar = self.radr * self.rar;        #   reduce random action rate

        #   update state and action
        self.a=next_action;
        self.s=s_prime;
            
        
        return next_action;
    
    def querySetState(self, s):
    #   Keeps previous state (s) as new state.  Returns the next action
    #   to take at state s.  Does not update Q-table or reduce random action rate.
    #   2 main uses:
    #       - Set initial state of learner
    #       - Use a learned policy without updating Q-table
        self.s = s;
        if rdm.uniform(0, 1) > self.rar:             #   if random action is selected
            next_action = rdm.randint(0, self.num_actions-1);       #   select random action
            #print('Random action selected...');
        else:
            next_action = self.best_action(s);
            #print('Best action selected using policy...');
    
        return next_action;
