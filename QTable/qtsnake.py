#+AUTHOR: bdunahu
#+TITLE: qtsnake.py
#+DESCRIPTION qtable lookup, training, and handling for multiagent snake

import numpy as np
from GameEngine import multiplayer
from collections import namedtuple

WINDOW_WIDTH = None
WINDOW_HEIGHT = None
GAME_UNITS = None

Point = namedtuple('Point', 'x, y')


def sense_goal(head, goal):
    '''
    maps head and goal location onto an
    integer corresponding to approx location
    '''
    diffs = Point(goal.x - head.x, goal.y - head.y)

    if diffs.x == 0 and diffs.y <  0:
        return 0
    if diffs.x >  0 and diffs.y <  0:
        return 1
    if diffs.x >  0 and diffs.y == 0:
        return 2
    if diffs.x >  0 and diffs.y >  0:
        return 3
    if diffs.x == 0 and diffs.y >  0:
        return 4
    if diffs.x <  0 and diffs.y >  0:
        return 5
    if diffs.x <  0 and diffs.y == 0:
        return 6
    return 7

def load_q(filename):
    ''' loads np array from given file '''
    if not filename.endswith('.npy'):
        exit(1)
    return np.load(filename)

class QSnake:
    def __init__(self, game_engine):
        ''' initialize fields required by model '''
        self.game_engine = game_engine

    def index_actions(self, q, pid):
        '''
        given q, player_id, an array of heads,
        and the goal position,
        indexes into the corresponding expected
        reward of each action
        '''
        heads, tails, goal = self.game_engine.get_heads_tails_and_goal()
        state = sense_goal(heads[pid], goal)
        return state, q[state, :]

    def argmin_gen(self, rewards):
        '''
        Given an array of rewards indexed by actions,
        yields actions in order from most rewarding to
        least rewarding
        '''
        rewards = rewards.copy()
        for i in range(rewards.size):
            best_action = np.argmin(rewards)
            rewards[best_action] = float("inf")
            yield best_action

    def pick_greedy_action(self, q, pid, epsilon=0):
        '''
        given a q table, the id of the player
        taking action, and a randomization factor,
        returns the most rewarding non-lethal action
        or a non-lethal random action.
        '''
        viable_actions = self.game_engine.get_viable_actions(pid)
        state, rewards = self.index_actions(q, pid)

        if np.random.uniform() < epsilon:
            return (state, np.random.choice(viable_actions)) if viable_actions.size > 0 else (state, 0)
        for action in self.argmin_gen(rewards):
            if action in viable_actions:
                return (state, action)
        return (state, 0) # death

    def update_q(self, q, old_state_action, new_state_action, outcome, lr=0.05):
        '''
        given a q table, the previous state/action pair,
        the new state/action pair, the outcome of the last
        action, and the learning rate
        updates q with the temporal difference.
        '''
        if outcome == multiplayer.CollisionType.GOAL:
            q[new_state_action[0], new_state_action[1]] = 0
        else:
            td_error = -1 + q[new_state_action[0], new_state_action[1]] - q[old_state_action[0], old_state_action[1]]
            q[old_state_action[0], old_state_action[1]] += lr * td_error

