import pandas as pd
import matplotlib.pyplot as plt
import argparse

#parser = argparse.ArgumentParser(description='Moving average.')
#parser.add_argument("-f", "--file", required=True)

#file_name = vars(parser.parse_args())["file"]

df = pd.read_csv("1extralayer_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="1extralayer")


df = pd.read_csv("2extralayer_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="1extralayer")


df = pd.read_csv("Clipnorm0.5_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="Clipnorm0.5")


df = pd.read_csv("SmallImg_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="SmallImg")


df = pd.read_csv("Close-action-policy-end_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="Close-action-policy-end")


df = pd.read_csv("conv-16kern_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="conv-16kern")


df = pd.read_csv("Dropout0.2-0.1_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="Dropout0.2-0.1")


df = pd.read_csv("Dropout0.5-0.1_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="Dropout0.5-0.1")


df = pd.read_csv("Dueling_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="Dueling")


df = pd.read_csv("HLayer512_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="HLayer512")


df = pd.read_csv("LayerSpeed_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="LayerSpeed")


df = pd.read_csv("lrelu0.1_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="lrelu0.1")


df = pd.read_csv("lrelu0.5_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="lrelu0.5")


df = pd.read_csv("Replay64-end-ep_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="Replay64-end-ep")


df = pd.read_csv("Replay64x10-end-ep_DQNAgent_test.txt")

reward = df["acum_reward"]

ma_reward = df["acum_reward"].rolling(50).mean()

plt.plot(ma_reward, label="Replay64x10-end-ep")


plt.legend()

plt.savefig("reward_red.png")
plt.clf()




