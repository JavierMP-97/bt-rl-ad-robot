import pandas as pd
import matplotlib.pyplot as plt
import argparse

#parser = argparse.ArgumentParser(description='Moving average.')
#parser.add_argument("-f", "--file", required=True)

#file_name = vars(parser.parse_args())["file"]

df = pd.read_csv("distancesquared_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="distancesquared")


df = pd.read_csv("endfast_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="endfast")


df = pd.read_csv("speedtanh_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="speedtanh")


df = pd.read_csv("tanhfast_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="tanhfast")


plt.legend()

plt.savefig("reward_reward-func.png")
plt.clf()




