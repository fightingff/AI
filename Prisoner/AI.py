import math
import random
import time

"""
if both prisoners choose to confess, both prisoners will be sentenced to 8 years in prison.
if both prisoners choose to remain silent, both prisoners will be sentenced to 1 year in prison.
if one prisoner chooses to confess and the other chooses to remain silent, the one who confesses will be set free and the other will be sentenced to 10 years in prison.
0 means remain silent
1 means confess
"""
round = 10
Judge=[[-1,-10],[0,-8]]

class AI():

    def __init__(self, alpha=0.5, epsilon=0.01):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple of former operations, where pos 2*i,2*i+1 means a round e.g. (1, 0, 0, 1) means 2 round
         - `action` is a number 0 or 1 representing whether the prisoner confesses
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        if (tuple(state), action) in self.q:
            return self.q[(tuple(state), action)]
        return 0

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estiamte of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        self.q[(tuple(state), action)] = old_q + self.alpha * (reward + future_rewards - old_q)
    
    def best_future_reward(self, state):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """
        return max(self.get_q_value(state, 0), self.get_q_value(state, 1))
            
    def choose_action(self, state, epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """
        if epsilon and random.random() < self.epsilon:
            return int(random.random()*2)
        else:
            Silent = self.get_q_value(state, 0)
            Confess = self.get_q_value(state, 1)
            return 0 if Silent > Confess else 1

def selftrain(N):
    ai = AI()
    for i in range(N):
        print('AI training round: ', i+1)
        state1 = []
        state2 = []
        reward1 = 0
        reward2 = 0
        for j in range(round):
            tp1 = state1.copy()
            tp2 = state2.copy()
            action1 = ai.choose_action(state1)
            action2 = ai.choose_action(state2)
            reward1 += Judge[action1][action2]
            reward2 += Judge[action2][action1]
            
            print(Judge[action1][action2],Judge[action2][action1])
            
            state1.append(action1)
            state1.append(action2)
            state2.append(action2)
            state2.append(action1)            
            ai.update(tp1, action1, state1, reward1)
            ai.update(tp2, action2, state2, reward2)
        print(reward1, reward2)
    print('AI training finished')
    return ai  
            
def play(ai,player):
    state = []
    reward = 0
    score = 0
    for i in range(round):
        tp = state.copy()
        action = ai.choose_action(state)
        against = player.choice()
        
        print(Judge[action][against],Judge[against][action])
        
        reward += Judge[action][against]
        score += Judge[against][action]
        state.append(action)
        state.append(against)
        ai.update(tp,action,state,reward)
        player.update(action)
    print(reward,score)