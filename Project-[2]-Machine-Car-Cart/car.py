#!/usr/bin/python3
import argparse
import logging
import sys
import os
import time
import numpy as np

import gym
from gym import wrappers, logger


class MountainCar:
    def __init__(self, env_id, train, test, model, render=False):
        # state =  [pos(x), vel(xdot)]
        self.min_vals = [-1.2,     -0.07]  
        self.max_vals = [0.6,     0.07]  
        self.num_bins = [40,     40]  # This needs to be changed
        self.bins = np.array([np.linspace(self.min_vals[i], self.max_vals[i], self.num_bins[i])
                        for i in range(len(self.max_vals))])
        self.env_id = env_id
        self.train = train
        self.test = test
        self.model = model
        self.render = render

    ################################################################################
    # CS482: this is the function that changes the continuous values of the state 
    ################################################################################
    def discretize_state(self,states):
        discretized_states = 0
        for i, state in enumerate(states):
            discretized_states += ((self.num_bins[i]+1)**i)*np.digitize(state, self.bins[i])
        return discretized_states

    
    def run(self):

        if not self.train and not self.test:
            print("[Warning] Specify train or test flag")
            print("for eg: python3 cart.py --train")
            print("or python3 cart.py --test --model cart.npy")

        if self.test:
            assert self.model is not None,\
                "Error: path to learned model path is not provided."
            if not os.path.exists(self.model):
                print("[Error] invalid model path\nNo such file as '" +
                    self.model+"' found")
                sys.exit(1)


        if self.render:
            env = gym.make(self.env_id, render_mode='human')
        else:
            env = gym.make(self.env_id)


        ############################################################################
        # CS482: This initial Q-table size should change to fit the number of
        # actions (columns) and the number of observations (rows)
        ############################################################################

        if self.train:
            # initialize Q table with zeros
            Q = np.zeros([(self.num_bins[0]+1)*(self.num_bins[1]+1)-1, env.action_space.n])
        if self.test:
            # load the saved model(learned Q table)
            Q = np.load(self.model, allow_pickle=True)

        ############################################################################
        # CS482: Here are some of the RL parameters. You have to tune the
        # learning rate (alpha) and the discount factor (gamma)
        ############################################################################

        alpha = 0.5  # This needs to be changed
        gamma = 0.90  # This needs to be changed
        # epsion-greedy params
        eps_start = 0.9
        eps_end = 0.05
        eps_decay = 100000
        if self.train:
            n_episodes = 5001     # This might need to be changed
            time_ = 1
            for episode in range(n_episodes):
                tick = 0
                reward = 0
                done = False
                # env.reset() returns a tuple of internal state
                state = env.reset()
                s = self.discretize_state(state[0])

                if episode % 100 == 0:
                    print(f"Starting Episode {episode}")

                while done != True:
                    time_ += 1
                    tick += 1
                    action = 0
                    ri = -999
                    ################################################################
                    # CS482: Implement epsilon-greedy strategy that chooses 
                    # actions based on exploration or exploitation phase.
                    ###############################################################
                    if np.random.random() <= eps_start:
                        action = np.random.randint(env.action_space.n)
                    else:
                        action = np.argmax(Q[s])

                    eps_start = max(eps_start - 1/eps_decay, eps_end)

                    state, reward, done, info,_ = env.step(action)
                    sprime = self.discretize_state(state)
                    predicted_value = np.max(Q[sprime])

                    ################################################################
                    # CS482: Implement the update rule for Q learning here
                    ################################################################

                    Q[s, action] += alpha * (reward + gamma * predicted_value - Q[s, action])

                    s = sprime

                if episode % 10 == 0 and self.render:
                    env.render()
                    time.sleep(0.5)
                if episode % 1000 == 0:
                    alpha *= .996

                if state[0] < 0.5:
                    print("fail ")
                else:
                    print("success")
            np.save('car.npy', Q)
            print("Q table saved")

        if self.test:
            ########################################################################
            # CS482: this part of code relates to testing the performance of
            # the loaded (possibly learned) model
            ########################################################################
            reward = 0
            done = False
            state = env.reset()
            s = self.discretize_state(state[0])
            all_states = []
            while done != True:
                action = 0
                ri = -999
                # select action that yields max q value
                for q in range(env.action_space.n):
                    if Q[s][q] > ri:
                        action = q
                        ri = Q[s][q]
                state, reward, done, info, _ = env.step(action)
                sprime = self.discretize_state(state)
                # render the graphics
                if self.render:
                    env.render()
                    time.sleep(0.005)

                all_states.append(state)

                s = sprime
            
            all_states = np.array(all_states)
            assert isinstance(all_states, np.ndarray)
            return all_states

def parse_args():

    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--env_id', dest='env_id', nargs='?',
                        default='MountainCar-v0', help='Select the environment to run')
    parser.add_argument('--train', dest='train', action='store_true',
                        help=' boolean flag to start training (if this flag is set)')
    parser.add_argument('--test', dest='test', action='store_true',
                        help='boolean flag to start testing (if this flag is set)')
    parser.add_argument('--model', dest='model', type=str, default=None,
                        help='path to learned model')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    render=True
    mountain_car = MountainCar(args.env_id, args.train, args.test, args.model, render)
    mountain_car.run()

if __name__ == '__main__':
    main()