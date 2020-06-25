import numpy as np

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


class Agent:

    def __init__(self):
        self.states = []  # record position and action taken at the position
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.isEnd = self.State.isEnd
        self.reward = -5
        self.exp_rate = 0.3
        self.lr = 0.2
        self.decay_gamma = 0.7

        # initial Q values
        self.Q_values = {}
        for i in range(GridRows):
            for j in range(GridCols):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = np.random.randint(-5, 1)  # Q value is a dict of dict

    def showQvalue(self):
        print("|       | up | down | left | right |")
        for i in range(GridRows):
            for j in range(GridCols):
                print("|" + " (" + str(i) + "," + str(j) + ") " + "|", end="")
                self.Q_values[(i, j)]
                for a in self.actions:
                    value = str(self.Q_values[(i, j)][a])
                    print("  " + value + "  ", end="")
                print()

    def chooseAction(self):
        # choose action with most expected value
        mx_nxt_reward = -1000
        action = ""
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            for a in self.actions:
                current_position = self.State.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward >= mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
        return action

    def maxReward(self, state):
        mx_nxt_reward = -1000
        action = ""
        for a in self.actions:
            # current_position = self.State.state
            nxt_reward = self.Q_values[state.state][a]
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
            # to the end of game back propagate reward
            if self.State.isEnd:
                self.reset()
                i += 1
            else:
                action = self.chooseAction()
                # current_q = self.Q_values[self.State.state][action]
                # using Bellman Optimality Equation to update q function
                next_state = self.takeAction(action)
                self.Q_values[self.State.state][action] = self.reward + self.decay_gamma * self.maxReward(next_state)
                # append trace
                # self.states.append([(self.State.state), action])
                # print("current position {} action {}".format(self.State.state, action))
                # by taking the action, it reaches the next state
                self.State = next_state
                # mark is end
                self.State.isEndFunc()
                # print("nxt state", self.State.state)
                # print("---------------------")
                self.isEnd = self.State.isEnd


ag = Agent()
print("Initialize Q-table")
print("show Q-value before iteration :")
ag.showQvalue()
# show Q-value after iteration :
ag.play()
print("Train")
print("show Q-value after iteration :")
ag.showQvalue()