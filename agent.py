from datetime import datetime, timedelta
import argparse
from operator import index
from random import random
import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# generate plots without rendering to screen
matplotlib.use("Agg")

device= 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'
DATE_FORMAT = "%m-%d %H:%M:%S"

class Agent:

    def __init__(self,hyperparameter_set):
        """
        Initialize the Agent with a given hyperparameter set.

        Parameters:
            hyperparameter_set (str): The name of the hyperparameter set to use.

        Attributes:
            replay_memory_size (int): The size of the replay memory.
            batch_size (int): The batch size to use for training.
            epsilon_init (float): The initial value of epsilon.
            epsilon_decay (float): The decay rate of epsilon.
            epsilon_min (float): The minimum value of epsilon.
        """

        with open("hyperparameters.yaml", 'r') as file:
            all_hyperparameter_Sets = yaml.safe_load(file)
            hyperparameters=all_hyperparameter_Sets[hyperparameter_set]
        self.env_id = hyperparameters['env_id']
        self.replay_memory_size=hyperparameters['replay_memory_size']
        self.mini_batch_size=hyperparameters['mini_batch_size']
        self.epsilon_init=hyperparameters['epsilon_init']
        self.epsilon_decay=hyperparameters['epsilon_decay']
        self.epsilon_min=hyperparameters['epsilon_min']
        self.learning_rate_a=hyperparameters['learning_rate_a']
        self.network_sync_rate=hyperparameters['network_sync_rate']
        self.discount_factor_g=hyperparameters['discount_factor_g']
        self.stop_on_reward=hyperparameters['stop_on_reward']
        self.fc1_nodes=hyperparameters['fc1_nodes']
        self.env_make_params=hyperparameters.get('env_make_params',{})
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']    

        self.loss_fn=torch.nn.MSELoss()
        self.optimizer=None

        self.LOG_FILE=os.path.join(RUNS_DIR,f'{hyperparameter_set}.log')
        self.MODEL_FILE=os.path.join(RUNS_DIR,f'{hyperparameter_set}.pt')
        self.GRAPH_FILE=os.path.join(RUNS_DIR,f'{hyperparameter_set}.png')


    def run(self,is_training=True,render=False):

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time
            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
            
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        num_states=env.observation_space.shape[0]
        num_actions=env.action_space.n

        rewards_per_episode=[]

        policy_dqn=DQN(num_states,num_actions,self.fc1_nodes).to(device)


        if is_training:
            
           
            epsilon = self.epsilon_init
           
            memory=ReplayMemory(self.replay_memory_size)
           

            # we creae another network that will be used to compute the target Q values to evaluate the policy DQN
            # target DQN weights are updated every fixed number of steps from the policy DQN weights
            target_dqn=DQN(num_states,num_actions,self.fc1_nodes).to(device)           
            
            target_dqn.load_state_dict(policy_dqn.state_dict())

            self.optimizer=torch.optim.Adam(policy_dqn.parameters(),lr=self.learning_rate_a)
            
            epsilon_history=[]
            
            step_count=0

            best_reward=-9999999
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            policy_dqn.eval()



        for episode in itertools.count():
            state, _ = env.reset()
            state=torch.tensor(state,dtype=torch.float , device=device)

            terminated = False
            episode_reward = 0.0

            while (not terminated and episode_reward < self.stop_on_reward):
                
                if is_training and random.random() < epsilon:
                    # enviroments provides a random action to choose from
                    action =torch.tensor( env.action_space.sample(),dtype=torch.int64, device=device)
                else:
                    # using the policy DQN we select the action
                    # get the largest Q value action from the policy DQN between the 2 actions (flap or not flap)
                    with torch.no_grad():    
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

                # action = env.action_space.sample() wrong line , ovverrifes the DQN action selection

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    # Storing the transition:
                    memory.append((state, action, reward, new_state, terminated))

                    step_count += 1
                
                state = new_state

                # Checking if the player is still alive
                if terminated:
                    break
            
            rewards_per_episode.append(episode_reward)

            if is_training:
                if episode_reward > best_reward:
                    log_msg = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_msg)
                    with open(self.LOG_FILE, 'a') as log_file:
                        log_file.write(f"{log_msg}\n")
                    
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch,policy_dqn,target_dqn )

                    epsilon=max(epsilon*self.epsilon_decay,self.epsilon_min )
                    epsilon_history.append(epsilon)


                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0 

    def optimize(self,mini_batch,policy_dqn,target_dqn):
        
        states, actions, rewards, new_states, terminations = zip(*mini_batch)



    # convert to tensors , stack tensors to batch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        with torch.no_grad():
                if self.enable_double_dqn:
                    best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                    target_q = rewards + (1-terminations) * self.discount_factor_g * \
                                    target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
                else:
                    # Calculate target Q values (expected returns)
                    target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
                    '''
                        target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                            .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                                [0]             ==> tensor([3,6])
                    '''

            # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases

    
    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

        






if __name__ == "__main__":
    parser =argparse.ArgumentParser(description="Train or test model")
    parser.add_argument('hyperparameters',help='')
    parser.add_argument('--train', action='store_true', help='Training mode')
    args=parser.parse_args()

    dql=Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        # test mode
        dql.run(is_training=False,render=True)

