import sys
import numpy as np

from tqdm import tqdm
from actor import Actor
from critic import Critic
from utils.stats import gather_stats
from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from utils.memory_buffer import MemoryBuffer

import matplotlib.pyplot as plt
import pandas as pd

class DDPGAgent:
    """ Deep Deterministic Policy Gradient (DDPG) Helper Class
    """

    def __init__(self, act_dim, env_dim, act_range, name="DDPGAgent", path="./save/", batch_size=1, buffer_size = 1, gamma = 0.99, lr = 0.000005, tau = 0.001, sigma = 1):
        """ Initialization
        """

        self.name = name
        self.path = path
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.env_dim = env_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.actor = Actor(self.env_dim, act_dim, act_range, 0.1 * lr, tau)
        self.critic = Critic(self.env_dim, act_dim, lr, tau)
        self.buffer = MemoryBuffer(buffer_size)
        self.batch_size = batch_size
        self.time = 0

        self.episode = 0
        self.tick = 0
        self.tick_episode = 0
        self.acum_reward = 0
        self.best_reward = 0

        self.sigma = 1
        self.noise = OrnsteinUhlenbeckProcess(size=self.act_dim, x0=0, sigma=self.sigma)

        with open(self.path+self.name+".txt", "a") as info:
            info.write("episode,ticks,acum_reward,avg_reward,total_ticks\n")
            info.close()
        with open(self.path+self.name+"_test.txt", "a") as info:
            info.write("episode,ticks,acum_reward,avg_reward\n")
            info.close()

    def act(self, s, train=False):
        """ Use the actor to predict value
        """
        a = self.actor.predict(s)[0]
        print("Action predicted: {}".format(a))
        if train:
            a = np.clip(a+self.noise.generate(self.time), -self.act_range, self.act_range)
            self.time += 1
        print("Action: {}".format(a))
        return a

    def bellman(self, rewards, q_values, dones):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                critic_target[i] = rewards[i]
            else:
                critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def remember(self, state, action, reward, new_state, done):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, done, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)
        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def replay(self, batch_size, save_data=True):
        states, actions, rewards, dones, new_states, _ = self.sample_batch(batch_size)
        # Predict target q-values using target networks
        q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
        # Compute critic target
        critic_target = self.bellman(rewards, q_values, dones)
        # Train both networks on sampled batch, update target networks
        self.update_models(states, actions, critic_target)

    '''
    def train(self, env, args, summary_writer):
        results = []

        # First, gather experience
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            actions, states, rewards = [], [], []
            

            while not done:
                #if args.render: env.render()
                # Actor picks an action (following the deterministic policy)
                a = self.policy_action(old_state)
                # Clip continuous values to be valid w.r.t. environment
                a = np.clip(a+noise.generate(time), -self.act_range, self.act_range)
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                # Add outputs to memory buffer
                self.memorize(old_state, a, r, done, new_state)
                # Sample experience from buffer
                
                # Update current state
                old_state = new_state
                cumul_reward += r
                time += 1

            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()
            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()

        return results
    '''

    def save(self):
        self.actor.save(self.path+self.name)
        self.critic.save(self.path+self.name)
        print("{} saved\n".format(self.path+self.name))

    def load(self):
        try:
            self.critic.load_weights(self.path+self.name)
            self.actor.load_weights(self.path+self.name)
        except:
            print("{} couldn't load. Creating a new {}\n".format(self.path+self.name, self.name))
            self.save()
        else:
            print("{} loaded\n".format(self.path+self.name))

    def next_episode(self, train=True):
        if train:
            print("{} finished\n".format(self.path+self.name))
            print("Episode: {}. Ticks: {}. Total ticks: {}. Acum_Reward: {}. Avg_Reward: {}\n".format(self.episode, self.tick_episode, self.tick, self.acum_reward, self.acum_reward/self.tick_episode))

            with open(self.path+self.name+".txt", "a") as info:
                info.write("{},{},{},{},{}\n".format(self.episode, self.tick_episode, self.acum_reward, self.acum_reward/self.tick_episode, self.tick))
                info.close()
                plt.plot()

            self.save()
            
            grafica = pd.read_csv(self.path+self.name+".txt")

            plt.plot(grafica["ticks"])
            plt.savefig(self.path+self.name+"_ticks.png")
            plt.clf()
            plt.plot(grafica["acum_reward"])
            plt.savefig(self.path+self.name+"_reward.png")
            plt.clf()

            print("Replay started...\n")
            #for q in range(1):
            #    agent.replay(64)

            self.episode += 1

        else:
            print("{} Test finished\n".format(self.path+self.name))
            print("Test: {}. Ticks: {}. Acum_Reward: {}. Avg_Reward: {}\n".format(self.episode, self.tick_episode, self.acum_reward, self.acum_reward/self.tick_episode))

            with open(self.path+self.name+"_test.txt", "a") as info:
                info.write("{},{},{},{}\n".format(self.episode, self.tick_episode, self.acum_reward, self.acum_reward/self.tick_episode))
                info.close()
                plt.plot()
            
            grafica = pd.read_csv(self.path+self.name+"_test.txt")

            plt.plot(grafica["ticks"])
            plt.savefig(self.path+self.name+"_ticks_test.png")
            plt.clf()
            plt.plot(grafica["acum_reward"])
            plt.savefig(self.path+self.name+"_reward_test.png")
            plt.clf()

            if self.acum_reward > self.best_reward:
                self.best_reward = self.acum_reward
                #self.update_best_model()
            
        self.acum_loss = 0
        self.tick_episode = 0    
        self.acum_reward = 0
        self.noise = None
        self.noise = OrnsteinUhlenbeckProcess(size=self.act_dim, x0=0, sigma=self.sigma)