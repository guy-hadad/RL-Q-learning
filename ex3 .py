import random
from numpy.random import choice
import numpy as np

ids = ["316508126", "316299098"]

def tr(obs0):
    a = obs0["drone_location"]
    b = obs0["packages"]
    c = obs0["target_location"]
    # return tuple((a[0], a[1], str(b), str(c)))
    return tuple((a[0], a[1], str(b)))


class DroneAgent:
    def __init__(self, n, m):
        self.mode = 'train'  # do not change this!
        self.q = {}
        self.T = 500
        self.epsilon = 0.1  # exploration constant
        self.alpha = 1  # learning constant
        self.gamma = 1  # discount constant
        self.actions = ("move_up", "move_down", "move_left", "move_right")
        self.current = 0
        self.n = n
        self.m = m

    def get_q(self, state, action):
        state = tr(state)
        return self.q.get((state, action), 0.0)

    def select_action(self, obs0):
        drone_loc = obs0["drone_location"]
        b = list(obs0["packages"])
        target_loc = obs0["target_location"]
        if len(b) != 0:
            pac_loc = b[0][1]
            if pac_loc[0] == drone_loc[0] and pac_loc[1] == drone_loc[1]:
                counter = 0
                for i in b:
                    if i[1] == "drone":
                        counter += 1
                if counter < 2:
                    return "pick"
            elif pac_loc == "drone" and drone_loc[0] == target_loc[0] and drone_loc[1] == target_loc[1]:
                return "deliver"
            actions = []
            row = drone_loc[0]
            col = drone_loc[1]
            # check if it is possible for the drone to move UP
            if ((row - 1) >= 0):
                actions.append('move_up')
            # check if it is possible for the drone to move DOWN
            if ((row + 1) <= self.n - 1):
                actions.append('move_down')
            # check if it is possible for the drone to move LEFT
            if ((col - 1) >= 0):
                actions.append('move_left')
            # check if it is possible for the drone to move RIGHT
            if ((col + 1) <= self.m - 1):
                actions.append('move_right')
        else:
            return "reset"

        if self.mode =="test":
            self.epsilon = 0.05
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.get_q(obs0, a) for a in actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = actions[i]
        return action


        #WHATS THIS???
        b_constants = np.exp(np.array([self.get_q(obs0, a) for a in self.actions])/self.T)
        probs = b_constants / sum(b_constants)
        action = choice(self.actions, 1, p=probs)[0]
        if self.T > 1:
            self.T -= 0.2
        return action

    def train(self):
        self.mode = 'train'  # do not change this!

    def eval(self):
        self.mode = 'eval'  # do not change this!

    def update(self, obs0, action, obs1, reward):
        """
                Q-learning:
                    Q(s, a) += alpha * (reward_func(s,a) + max(Q(s')) - Q(s,a))
                """
        q_max = max([self.get_q(obs1, a) for a in self.actions])
        old_q = self.q.get((tr(obs0), action), None)
        # old_q = self.q.get((obs0, action), None)
        if old_q is None:
            self.q[(tr(obs0), action)] = reward
            # self.q[(obs0, action)] = reward
        else:
            self.q[(tr(obs0), action)] = old_q + self.alpha * (reward + self.gamma * q_max - old_q)
