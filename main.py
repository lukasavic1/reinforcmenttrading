import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import math
import numpy as np
import pandas as pd
import random
from collections import deque
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, state_size, is_eval=False, model_name="model1"):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.inventory_type = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = load_model(model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def formatPrice(n):
    return ("-Rs." if n < 0 else "Rs.") + "{0:.5f}".format(abs(n))


def preparePlot(key):
    df = pd.read_csv(key, ",", names=["time", "open", "high", "low", "close", "adjusted_close", "volume"])
    plt.plot(df.index, df["close"])


def split_df_pct(df, pct):
    length = int(len(df) * pct)
    list1 = []
    list2 = []
    for i in range(0, length):
        list1.append(df.iloc[i])
    for i in range(length, len(df)):
        list2.append(df.iloc[i])
    return list1, list2


def getStockDataVec(key):
    vec = []
    lines = open(key, "r").read().splitlines()
    length = len(lines)
    print(length)
    len1 = int(0.2 * length)
    print(len1)
    for line in lines[1:len1]:
        # print(line)
        # print(float(line.split(",")[4]))
        if (line.split(",")[4] == "null"):
            vec.append(last_value)
        else:
            last_value = float(line.split(",")[4])
            vec.append(last_value)
        # print(vec)

    return vec


def sigmoid(x):
    # print(x)
    return 1 / (1 + math.exp(-x))


def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])


import sys

stock_name = input("Enter stock_name, window_size, Episode_count")
window_size = input()
episode_count = input()
stock_name = str(stock_name)
preparePlot(stock_name)
window_size = int(window_size)
episode_count = int(episode_count)
agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32
for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    agent.inventory_type = []
    for t in range(l):
        # for t in range(1000):

        action = agent.act(state)
        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        if action == 1:  # buy
            if len(agent.inventory) == 0:
                agent.inventory.append(data[t])
                agent.inventory_type.append(1)
                plt.plot(t, data[t], "^", markersize=7, color="g")
                print("Buy: " + formatPrice(data[t]))
                print(t)
            else:
                last_order = agent.inventory_type.pop()
                if (last_order == 1):  # this means it was a buy position
                    agent.inventory_type.append(1)
                    agent.inventory.append(data[t])
                    agent.inventory_type.append(1)
                    plt.plot(t, data[t], "^", markersize=7, color="g")
                    print("Buy: " + formatPrice(data[t]))
                    print(t)
                else:
                    while (last_order == 0):
                        last_sold_price = agent.inventory.pop()
                        reward = max(last_sold_price - data[t], 0)
                        total_profit += last_sold_price - data[t]
                        print("Sell Closed: " + formatPrice(data[t]) + " | Profit: " + formatPrice(
                            last_sold_price - data[t]))
                        if (len(agent.inventory) == 0):
                            break
                        last_order = agent.inventory_type.pop()
                        if (last_order == 1):
                            agent.inventory_type.append(1)
                    agent.inventory.append(data[t])
                    agent.inventory_type.append(1)
                    plt.plot(t, data[t], "^", markersize=7, color="g")
                    print("Buy: " + formatPrice(data[t]))
                    print(t)
        elif action == 2:  # sell
            # bought_price = window_size_price = agent.inventory.pop(0)
            # plt.plot(t,data[t],"^",markersize=7,color="r")
            # reward = max(data[t] - bought_price, 0)
            # total_profit += data[t] - bought_price
            # print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
            # print(t)
            if len(agent.inventory) == 0:
                agent.inventory.append(data[t])
                agent.inventory_type.append(0)
                plt.plot(t, data[t], "^", markersize=7, color="r")
                print("Sell: " + formatPrice(data[t]))
                print(t)
            else:
                last_order = agent.inventory_type.pop()
                if (last_order == 0):  # this means it was a sell position
                    agent.inventory_type.append(0)
                    agent.inventory.append(data[t])
                    agent.inventory_type.append(0)
                    plt.plot(t, data[t], "^", markersize=7, color="r")
                    print("Sell: " + formatPrice(data[t]))
                    print(t)
                else:
                    while (last_order == 1):
                        last_bought_price = agent.inventory.pop()
                        reward = max(data[t] - last_bought_price, 0)
                        total_profit += data[t] - last_bought_price
                        print("Buy Closed: " + formatPrice(data[t]) + " | Profit: " + formatPrice(
                            data[t] - last_bought_price))
                        if (len(agent.inventory) == 0):
                            break
                        last_order = agent.inventory_type.pop()
                        if (last_order == 0):
                            agent.inventory_type.append(0)
                    agent.inventory.append(data[t])
                    agent.inventory_type.append(0)
                    plt.plot(t, data[t], "^", markersize=7, color="r")
                    print("Sell: " + formatPrice(data[t]))
                    print(t)
        done = True if t == l - 1 else False
        # 3done = True if t == 1000 - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
    if e % 10 == 0:
        agent.model.save(str(e))

