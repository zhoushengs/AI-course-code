import numpy as np
from torch import nn
GridRows = 5
GridCols = 5
REWARD = -5
WinState1 = (0,0)
WinState2 = (4,4)

class State:
    def __init__(self, state=(np.random.randint(0, 5), np.random.randint(0, 5))):
        self.board = np.zeros([GridRows, GridCols])
        self.board[0, 0] = -1
        self.board[4, 4] = -1
        self.state = state
        self.isEnd = False
        self.reward = REWARD

    def isEndFunc(self):
        if (self.state == WinState1) or (self.state == WinState2):
            self.isEnd = True
    def nxtPosition(self, action):
        if action == "up":
            nxtState = (self.state[0] - 1, self.state[1])
        elif action == "down":
            nxtState = (self.state[0] + 1, self.state[1])
        elif action == "left":
            nxtState = (self.state[0], self.state[1] - 1)
        else:
            nxtState = (self.state[0], self.state[1] + 1)
        # if next state legal
        if (nxtState[0] >= 0) and (nxtState[0] <= 4):
            if (nxtState[1] >= 0) and (nxtState[1] <= 4):
                return nxtState
        return self.state

    def SB(self):
        self.board[self.state] = 1
        for i in range(0, GridRows):
            out = '| '
            for j in range(0, GridCols):
                if self.board[i, j] == 1:
                    token = 's'
                if self.board[i, j] == -1:
                    token = '*'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)


st = State()
st.SB()


class build_model:
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLu(),
            nn.Linear(8, 16),
            nn.ReLu(),
            nn.Linear(16, 1)
        )
        return model


class Agent:

    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.reward = -5
        self.exp_rate = 0.3
        self.decay_gamma = 0.7
        self.model = build_model()

        # initial actionValue
        self
        count1 = -1
        self.actionValue = {}
        for a in self.actions:
            self.actionvalue[a] = count1
            count1 -= 1

        # initial stateValue
        self.count2 = 1
        self.stateValue = {}
        for i in range(GridRows):
            for j in range(GridCols):
                self.stateValue[(i, j)] = self.count2
                self.count2 += 1

        # initial Q values
        self.Q_values = {}
        for i in range(GridRows):
            for j in range(GridCols):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = np.random.randint(-5, 1)  # Q value is a dict of dict

    def showQvalue(self):
        print("------------------------------------")
        print("|       | up | down | left | right |")
        for i in range(GridRows):
            for j in range(GridCols):
                print("------------------------------------")
                print("|" + " (" + str(i) + "," + str(j) + ") " + "|", end="")
                self.Q_values[(i, j)]
                for a in self.actions:
                    value = str(self.Q_values[(i, j)][a])
                    print("  " + value + "  ", end="")
                print()

                # build model



    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = -1000
        action = ""
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            for a in self.actions:
                current_position = self.State.state
                train = []
                train.append(self.stateValue[current_position])
                train.append(self.actionValue[a])
                nxt_reward = self.model(train)
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def maxReward(self, state):
        mx_nxt_reward = -1000
        action = ""
        for a in self.actions:
            # current_position = self.State.state
            train = []
            train.append(self.stateValue[state.state])
            train.append(self.actionValue[a])
            nxt_reward = self.model(train)
            if nxt_reward >= mx_nxt_reward:
                action = a
                mx_nxt_reward = nxt_reward
        return mx_nxt_reward

    def takeAction(self, action):
        position = self.State.nxtPosition(action)
        # update State
        return State(state=position)

    def reset(self):
        self.states = []
        self.State = State()
        self.isEnd = self.State.isEnd

    def play(self, rounds=10000):
        i = 0
        while i < rounds:
           if self.State.isEnd:
                self.reset()
                i += 1
           else:
                action = self.chooseAction()
                next_state = self.takeAction(action)
                self.Q_values[self.State.state][action] = self.reward + self.decay_gamma * self.maxReward(next_state)
                self.State = next_state
                self.State.isEndFunc()
                self.isEnd = self.State.isEnd
ag = Agent()
ag.showQvalue()
ag.play()

