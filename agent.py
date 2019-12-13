############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################
# Import modules from other libraries
import numpy as np
import collections
import torch

class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # The previous distance to goal stores the "reward" of previous step
        self.prev_distance_to_goal = 0
        # The episode ended flag is set to true if distance reached goal/exceed episode length
        self.episode_ended = False

        # Hyperparameters
        self.episode_length = 550           # Episode length
        self.batch_size = 100               # Training batch size
        self.buffer_size = 1000000          # Size of Experience replay buffer 
        self.buffer_start = 1000            # Number of transition to store before training
        self.gamma = 0.99                   # Bellman discount factor
        self.network_update_frequency = 100 # Frequency of updating Target network
        self.lr = 0.001                     # Learning rate
        self.epsilon = 0.9                  # Exploration probability
        self.epsilon_min = 0.70             # Minimum epsilon
        self.delta = 0.02                   # Epsilon decay rate 

        # The network variable stores the deep Q-Network
        self.network = DQN(lr=self.lr)
        # The buffer variable stores the Experience replay buffer
        self.buffer = ReplayBuffer(size=self.buffer_size, start=self.buffer_start)

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        return self.episode_ended
        

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Choose epsilon greedy action
        discrete_action = self._get_epsilon_greedy_action(state)
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = discrete_action
        return continuous_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = self._calculate_reward(distance_to_goal)
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Now you can do something with this transition ...
        # Add transition to replay buffer
        self.buffer.add(transition)
        # Sample transitions from buffer
        sample_batch = self.buffer.sample(self.batch_size)
        if sample_batch:
            # Train Q-network
            loss = self.network.train(sample_batch, self.gamma)
        # Update target network
        if self.num_steps_taken % self.network_update_frequency == 0:
            self.network.update_target_network()
        # Episode ended if reached goal or exceed episode length
        self.episode_ended = distance_to_goal < 0.03 or self.num_steps_taken == self.episode_length
        # When episode end
        if self.episode_ended:
            # If agent reached goal
            if distance_to_goal < 0.03:
                # Allow epsilon to decrease more
                self.epsilon_min = max(0.01, self.epsilon_min - 0.1)
                # Decrease episode length
                self.episode_length = max(100, self.episode_length - 20)
            # Reset num steps taken
            self.num_steps_taken = 0
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon - self.delta)

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        discrete_action = self._get_greedy_action(state)
        return self._discrete_action_to_continuous(discrete_action)

    # Function to get a random action
    def _get_random_action(self):
        # Returns a random action from [0, 1, 2, 3]
        return np.random.choice(4, p=[0.39, 0.3, 0.01, 0.3])

    # Function to get greedy action (discrete)
    def _get_greedy_action(self, state):
        return self.network.get_greedy_action(state)

    # Function to get a epsilon greedy action for a particular state
    def _get_epsilon_greedy_action(self, state):
        # Return random action with probability epsilon
        random_action = self._get_random_action()
        greedy_action = self._get_greedy_action(state)
        action = random_action if np.random.rand() < self.epsilon else greedy_action
        return action

    # Function to convert discrete action to continuous action
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0: # Move right
            return np.array([0.02, 0.0], dtype=np.float32)
        elif discrete_action == 1: # Move down
            return np.array([0.0, -0.02], dtype=np.float32)
        elif discrete_action == 2: # Move left
            return np.array([-0.02, 0.0], dtype=np.float32)
        elif discrete_action == 3: # Move up
            return np.array([0.0, 0.02], dtype=np.float32)
        else:
            print("Agent(_discrete_action_to_continuous): Invalid action")

    # Function to calculate reward
    def _calculate_reward(self, distance_to_goal):
         # Reward state
        if distance_to_goal < 0.03:
            reward = 5
        # Penalty state
        elif abs(distance_to_goal - self.prev_distance_to_goal) < 1e-7:
            reward = -0.3
        else:
            reward = 1 - distance_to_goal
        self.prev_distance_to_goal = distance_to_goal
        return reward

class Network(torch.nn.Module):

    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        # Define neural network architecture
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=500)
        self.layer_2 = torch.nn.Linear(in_features=500, out_features=800)
        self.layer_3 = torch.nn.Linear(in_features=800, out_features=300)
        self.output_layer = torch.nn.Linear(in_features=300, out_features=output_dimension)

    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        output = self.output_layer(layer_3_output)
        return output

# Deep Q-Network
class DQN:

    def __init__(self, lr):
        # Q-Network
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        # Initialise Target Network with same weight as Q-Network
        self.target_network = Network(input_dimension=2, output_dimension=4)
        self.update_target_network()

    # Train Q-Network given a batch of transitions
    def train(self, transitions, gamma):
        # Zero the parameter gradients
        self.optimiser.zero_grad()
        # Calculate loss
        loss = self._calculate_loss(transitions, gamma)
        # Backward propagation
        loss.backward()
        # Update weights
        self.optimiser.step()
        # Return loss
        return loss.item()

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        input_tensor = torch.tensor(state)
        output_tensor = self.q_network.forward(input_tensor).detach()
        _, action = output_tensor.max(0)
        return action

    # Copy weights from Q-Network to target network
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Calculate loss given a batch of transitions
    def _calculate_loss(self, transitions, gamma):
        states, actions, rewards, next_states = zip(*transitions)
        # Network input
        input_tensor = torch.tensor(states)
        # Predict highest Q-values for next states
        next_state_tensor = torch.tensor(next_states)
        next_state_predictions = self.target_network.forward(next_state_tensor).detach()
        max_qvalue_tensor = torch.unsqueeze(next_state_predictions.max(1).values, 1)
        # Find Network label using Bellman equation
        reward_tensor = torch.unsqueeze(torch.tensor(rewards), 1)
        label_tensor = reward_tensor + gamma * max_qvalue_tensor
        # Network prediction
        action_tensor = torch.unsqueeze(torch.tensor(actions), 1)
        output_tensor = torch.gather(self.q_network.forward(input_tensor), 1, action_tensor)
        # Calculate loss
        return torch.nn.MSELoss()(output_tensor, label_tensor)

class ReplayBuffer:

    def __init__(self, size, start):
        self.buffer = collections.deque(maxlen=size)
        self.start = start
        self.new_transition = None

    def add(self, transition):
        self.new_transition = transition
        self.buffer.append(transition)

    def sample(self, batch_size):
        # Wait until buffer contains batch_size samples
        if len(self.buffer) < self.start:
            return None
        else:
            # Select random batch of size batch_size
            batch = [self.buffer[np.random.randint(len(self.buffer))] for _ in range(batch_size - 1)]
            # Ensure the newly added transition is trained
            batch.append(self.new_transition)
            return batch













