import pandas as pd
import matplotlib.pyplot as plt
import argparse

#parser = argparse.ArgumentParser(description='Moving average.')
#parser.add_argument("-f", "--file", required=True)

#file_name = vars(parser.parse_args())["file"]

df = pd.read_csv("1_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="1")


df = pd.read_csv("10_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="10")


df = pd.read_csv("1000_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="1000")


df = pd.read_csv("10000_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="10000")


df = pd.read_csv("1e-05_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="100")


df = pd.read_csv("1e-05_DQNAgentb_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="100_b")


plt.legend()

plt.savefig("reward_tau.png")
plt.clf()




