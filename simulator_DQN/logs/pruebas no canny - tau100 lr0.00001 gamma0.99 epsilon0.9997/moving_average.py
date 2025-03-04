import pandas as pd
import matplotlib.pyplot as plt
import argparse

#parser = argparse.ArgumentParser(description='Moving average.')
#parser.add_argument("-f", "--file", required=True)

#file_name = vars(parser.parse_args())["file"]

df = pd.read_csv("no_canny_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="base")


df = pd.read_csv("no_canny_glorot_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="glorot")


df = pd.read_csv("no_canny_leaky0.1_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="leaky0.1")


df = pd.read_csv("no_canny_leaky0.5_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="leaky0.5")


df = pd.read_csv("no_canny_precrop_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="precrop")


df = pd.read_csv("no_canny_RGB_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="RGB")




plt.legend()

plt.savefig("reward_no_canny.png")
plt.clf()




