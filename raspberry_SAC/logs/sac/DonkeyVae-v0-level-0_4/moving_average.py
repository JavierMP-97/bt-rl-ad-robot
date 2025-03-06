import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Moving average.')
parser.add_argument("-f", "--file", required=True)

file_name = vars(parser.parse_args())["file"]

df = pd.read_csv(file_name+".txt")

reward = df["acum_reward"]

ticks = df["ticks"]

ma_reward = df["acum_reward"].rolling(10).mean()

ma_ticks = df["ticks"].rolling(10).mean()

plt.plot(ma_reward)
plt.savefig("ma_reward_"+file_name+".png")
plt.clf()

plt.plot(ma_ticks)
plt.savefig("ma_ticks_"+file_name+".png")



