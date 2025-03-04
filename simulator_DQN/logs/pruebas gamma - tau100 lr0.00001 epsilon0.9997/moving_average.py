import pandas as pd
import matplotlib.pyplot as plt
import argparse

#parser = argparse.ArgumentParser(description='Moving average.')
#parser.add_argument("-f", "--file", required=True)

#file_name = vars(parser.parse_args())["file"]

df = pd.read_csv("0.5_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="0.5")


df = pd.read_csv("0.9_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="0.9")


df = pd.read_csv("0.999_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="0.999")


df = pd.read_csv("0.9999_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="0.9999")


df = pd.read_csv("1e-05_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="0.99")


df = pd.read_csv("1e-05_DQNAgentb_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="0.99_b")


plt.legend()

plt.savefig("reward_gamma.png")
plt.clf()




