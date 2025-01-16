import gymnasium as gym
import tensorflow as tf
import sonnet as snt
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.ptr = 0

        self.state_memory = np.zeros((self.max_size, 4), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_size, 4), dtype=np.float32)
        self.action_memory = np.zeros((self.max_size, 1), dtype=np.float32)
        self.reward_memory = np.zeros(self.max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_size, dtype=np.float32)

    def append(self, state, action, reward, next_state, done):
        index = self.ptr % self.max_size

        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.ptr += 1

    def sample_epoch(self, epoch_size, batch_size):
        max_mem = min(self.ptr, self.max_size)

        epoch = []

        for _ in range(epoch_size // batch_size):
            batch = np.random.choice(max_mem, batch_size)

            states = self.state_memory[batch]
            actions = self.action_memory[batch]
            rewards = self.reward_memory[batch]
            rewards = tf.expand_dims(rewards, axis=-1)
            new_states = self.new_state_memory[batch]
            dones = self.terminal_memory[batch]
            dones = tf.expand_dims(dones, axis=-1)

            epoch.append((states, actions, rewards, new_states, dones))

        return epoch

    def sample_batch(self, batch_size):
        max_mem = min(self.ptr, self.max_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        rewards = tf.expand_dims(rewards, axis=-1)
        new_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        dones = tf.expand_dims(dones, axis=-1)

        return states, actions, rewards, new_states, dones

    def reset(self):
        self.ptr = 0

        self.state_memory = np.zeros((self.max_size, 4))
        self.new_state_memory = np.zeros((self.max_size, 4))
        self.action_memory = np.zeros((self.max_size, 1))
        self.reward_memory = np.zeros(self.max_size)
        self.terminal_memory = np.zeros(self.max_size, dtype=np.float32)


class QFunction(snt.Module):

    def __init__(self,
                 obs_space,
                 act_space,
                 ):
        super().__init__()
        self.optim = snt.optimizers.SGD(learning_rate=1e-3)

        self.MLP = snt.nets.MLP(
            [64, 64, 1],
        )

        dummy_obs = np.float32(obs_space.sample()[np.newaxis])
        dummy_act = np.float32(act_space.sample()[np.newaxis])
        self(dummy_obs, dummy_act)

    def __call__(self, observation, action):
        _input = tf.concat(
            [
                observation, action
            ],
            axis=-1
        )
        return self.MLP(_input)

    def get_loss(self, true, predicted):
        return tf.reduce_mean(tf.math.square(true - predicted))  # MSE


class Policy(snt.Module):

    def __init__(self,
                 obs_space,
                 act_space,
                 ):
        super().__init__()

        self.optim = snt.optimizers.SGD(learning_rate=1e-3)

        self.MLP = snt.nets.MLP(
            [64, 64, act_space.shape[0]],
        )

        dummy_obs = np.float32(obs_space.sample()[np.newaxis])
        self(dummy_obs)

    def __call__(self, state):
        _input = state

        return tf.nn.tanh(self.MLP(_input)) * 3


def update_target_weights(model, target_model, tau=0.):
    for var, target_var in zip(model.trainable_variables, target_model.trainable_variables):
        target_var.assign((1 - tau) * var + tau * target_var)


@tf.function
def train_ddpg(observations, actions, rewards, next_states, dones, discount,
               Q, Q_target,
               pi, pi_target,
               update_target,
               ):
    with tf.GradientTape() as tape:
        y = tf.stop_gradient(rewards + discount * (1. - dones) * Q_target(next_states, pi_target(next_states)))
        Q_loss = Q.get_loss(y, Q(observations, actions))

    # Update networks
    gradients = tape.gradient(Q_loss, Q.trainable_variables)
    Q.optim.apply(gradients, Q.trainable_variables)

    with tf.GradientTape() as tape:
        pi_loss = -tf.reduce_mean(Q(observations, pi(observations)))

    gradients_pi = tape.gradient(pi_loss, pi.trainable_variables)
    pi.optim.apply(gradients_pi, pi.trainable_variables)

    # Update target networks
    tau = 0.995
    update_target_weights(Q, Q_target, tau)
    update_target_weights(pi, pi_target, tau)

    return {"Q_loss": Q_loss, "pi_loss": pi_loss}


if __name__ == '__main__':

    seed = 0
    np.random.seed(seed)
    gamma = 0.99

    B = 128
    num_interactions = 0
    update_every = 50
    start_steps = 10_000
    update_after = 1000

    T = 1_000_000

    iteration = 0
    target_update_freq = 200

    noise_theta = 0.15
    noise_mu = 0
    noise_sigma_decay = 0.005
    noise_x0 = 0

    buffer = ReplayBuffer(T)

    env = gym.make('InvertedPendulum-v5')
    pi = Policy(env.observation_space, env.action_space)
    Q = QFunction(env.observation_space, env.action_space)
    pi_target = Policy(env.observation_space, env.action_space)
    update_target_weights(pi, pi_target)
    Q_target = QFunction(env.observation_space, env.action_space)
    update_target_weights(Q, Q_target)

    episodic_rewards = []
    mean_episode_length = []

    for episode in range(1000):
        state, _ = env.reset()

        done = False
        truncated = False
        episode_reward = 0
        episode_length = 0
        while not done:
            state = np.float32(state.reshape(1, -1))
            noise = np.random.normal(0, 0.1, env.action_space.shape[0])

            if num_interactions > start_steps:
                deterministic_action = pi(state).numpy().flatten()
                action = np.clip(deterministic_action + noise, -3, 3)
            else:
                action = np.clip(noise, -3, 3)

            action = np.float32(action)

            next_state, reward, done, truncated, info = env.step(action)
            buffer.append(state, action, reward, next_state, done)
            state = next_state

            num_interactions += 1
            if num_interactions % update_every == 0 and num_interactions >= update_after:
                for _ in range(update_every):
                    observations, actions, rewards, new_states, dones = buffer.sample_batch(B)
                    iteration += 1
                    metrics = train_ddpg(observations, actions, rewards, new_states, dones, gamma, Q, Q_target, pi,
                                         pi_target, iteration % target_update_freq == 0)

                print(num_interactions, iteration, metrics)
            episode_reward += reward
            episode_length += 1

        episodic_rewards.append(episode_reward)
        mean_episode_length.append(episode_length)

        if len(episodic_rewards) > 100:
            episodic_rewards.pop(0)
            mean_episode_length.pop(0)

        if episode % 10 == 0:
            mean = np.mean(episodic_rewards)
            print(episode, mean)
            if (mean > 1000):
                break

    env.close()
    print("Training done, calculating mean reward")
    env = gym.make('InvertedPendulum-v5')

    mean_rewards = 0
    for _ in range(10):
        current_rewards = 0
        state, _ = env.reset()
        done = False
        truncated = False
        while not done:
            state = np.float32(state.reshape(1, -1))
            action = np.float32(pi(state).numpy().flatten())
            next_state, reward, done, truncated, info = env.step(action)
            current_rewards += reward
            state = next_state
        env.close()
        mean_rewards += current_rewards
    print({"Reward for 10 episodes": mean_rewards})
