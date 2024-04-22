# """"""
# """
# Template for implementing QLearner  (c) 2015 Tucker Balch
#
# Copyright 2018, Georgia Institute of Technology (Georgia Tech)
# Atlanta, Georgia 30332
# All Rights Reserved
#
# Template code for CS 4646/7646
#
# Georgia Tech asserts copyright ownership of this template and all derivative
# works, including solutions to the projects assigned in this course. Students
# and other users of this template code are advised not to share it with others
# or to make it available on publicly viewable websites including repositories
# such as github and gitlab.  This copyright statement should not be removed
# or edited.
#
# We do grant permission to share solutions privately with non-students such
# as potential employers. However, sharing with other current or future
# students of CS 7646 is prohibited and subject to being investigated as a
# GT honor code violation.
#
# -----do not edit anything above this line---
#
# Student Name: Tucker Balch (replace with your name)
# GT User ID: rchen613 (replace with your User ID)
# GT ID: 903971253 (replace with your GT ID)
# """

import random as rand
import numpy as np

class QLearner(object):

    def author(self):
        return 'rchen613'


    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.s = 0
        self.a = 0
        self.exp_tuple = []
        self.Q = np.zeros((num_states, num_actions))


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        self.a = np.argmax(self.Q[s])
        action = self.a
        if self.verbose:
            print(f"s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The new reward
        @returns: The selected action
        """

        # update Q-table
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + self.alpha * (r + self.gamma * self.Q[s_prime, np.argmax(self.Q[s_prime])])

        # update experience tuple
        self.exp_tuple.append((self.s, self.a, s_prime, r))
        prob = rand.uniform(0.0, 1.0)
        if prob < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime])
        self.rar *= self.radr
        if self.verbose:
            print(f"s ={s_prime}, a = {action}, r = {r}")
        self.s = s_prime
        self.a = action
        return action



if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")