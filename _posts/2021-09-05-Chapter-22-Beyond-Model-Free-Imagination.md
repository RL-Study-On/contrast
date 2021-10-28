- Model-based methods
- Imagination-augmented agents

### Model-based versus model-free

- Model: the model of the environment
- Model-based: predicting, understanding, or simulating the environment
- Model-free: learn proper behavior directly (policy) or indirectly (value)

### Advantages of model-based methods

- Low dependency on the real environment → sample efficiency; accurate model → avoid executing action on environment, instead, use only the trained (expected) model
- Transferability
    - Good robot manipulator model can be used at various other tasks

### Dealing with model imperfections

Inaccurate model → policy learned from it may be totally wrong

- Local models family of methods: replace large environment with a small regime-based set of models and train them using trust region method
- Augment model-free policy with model-based paths: give model-free agent extra information and let agent decide if it will be useful

### The imagination-augmented agent (I2A)

Allow agent to imagine future trajectories using the current observations and incorporate these imagined paths into its deicison process

![스크린샷 2021-09-20 오후 5.01.59.png](https://github.com/RL-Study-On/contrast/blob/master/assets/09-05-book-minji/1.png?raw=true)

Agent consists of 2 different paths to transform the observation

1. Model-free
    - Set of convolution layers: observation → high-level feature
2. Imagination
    - Set of trajectories, each called rollout, imagined from the current observation
    - Rollouts are produced for every available action in the environment and fixed number of steps
    - On every step of rollouts, environment model (EM) produces the next observation and predicted immediate reward from current observation and action to be taken
    - Observation → EM → observation → EM... repeat N times
    - For the first step of trajectory, we try every possible action (each becomes one trajectory) but for the subsequent steps, the action is chosen using the small rollout policy network, which is trained with the main agent
    - Steps from single rollout are passed to rollout encoder → encodes them to fixed-size vector → feed to the head of the agent → agent produces usual policy & value estimations for A3C
    - Imagination path architecture with 2 rollout steps and 2 actions
        
        ![스크린샷 2021-09-20 오후 5.09.47.png](https://github.com/RL-Study-On/contrast/blob/master/assets/09-05-book-minji/2.png?raw=true)
        

### EM

- Goal is to convert observation & action → next observation & immediate reward
- Pixel input → EM returns pixel as next observation
- To incorporate the action into the convolution layers, action was one-hot encoded
    
    ![스크린샷 2021-09-20 오후 5.12.50.png](https://github.com/RL-Study-On/contrast/blob/master/assets/09-05-book-minji/3.png?raw=true)
    

### Rollout policy

- Goal is to decide action to be taken during imagined trajectory
- Since we intend to generate imaginary path, separate rollout policy network is trained but trained to produce similar output to our main agent's policy
- Similar architecture with A3C
- Trained in parallel to the main I2A network using a cross-entropy loss between the rollout policy network output and the output of the main network

### Rollout encoder

- Every rollout step is preprocessed with a small convolution network to extract the features from the observation
- Features → LSTM → fixed-size vector
- Vectors from every rollouts are concatenated together wit hfeatures from the model-free path and used to produce the policy & value estimation in the same way as A3C

### I2A on Atari Breakout

```python
FRAMES_COUNT = 2
IMG_SHAPE = (FRAMES_COUNT, 84, 84)
ROLLOUT_HIDDEN = 256
EM_OUT_SHAPE = (1, ) + IMG_SHAPE[1:]
```

```python
class EnvironmentModel(nn.Module):
    def __init__(self, input_shape, n_actions):
		'''
				Goal is to predict next observation and immediate reward
				using current observation and action as input
		'''
        super(EnvironmentModel, self).__init__()

        self.input_shape = input_shape
        self.n_actions = n_actions

        # input color planes will be equal to frames plus one-hot encoded actions
        n_planes = input_shape[0] + n_actions
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_planes, 64, kernel_size=4,
                      stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # next observation
        self.deconv = nn.ConvTranspose2d(
            64, 1, kernel_size=4, stride=4, padding=0)

				# immediate reward
        self.reward_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        rw_conv_out = self._get_reward_conv_out(
            (n_planes, ) + input_shape[1:])
        self.reward_fc = nn.Sequential(
            nn.Linear(rw_conv_out, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_reward_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.reward_conv(o)
        return int(np.prod(o.size()))

    def forward(self, imgs, actions):
				# one-hot encode the actions
        batch_size = actions.size()[0]
        act_planes_v = torch.FloatTensor(
            batch_size, self.n_actions, *self.input_shape[1:])
        act_planes_v.zero_()
        act_planes_v = act_planes_v.to(actions.device)
        act_planes_v[range(batch_size), actions] = 1.0

				# concatenate action and observation
        comb_input_v = torch.cat((imgs, act_planes_v), dim=1)

				# convolution layers
        c1_out = self.conv1(comb_input_v)
        c2_out = self.conv2(c1_out)
        c2_out += c1_out
				# next observation
        img_out = self.deconv(c2_out)
				
				# reward
        rew_conv = self.reward_conv(c2_out).view(batch_size, -1)
        rew_out = self.reward_fc(rew_conv)
        return img_out, rew_out
```

```python
class RolloutEncoder(nn.Module):
    def __init__(self, input_shape, hidden_size=ROLLOUT_HIDDEN):
        super(RolloutEncoder, self).__init__()
		'''
				Goal is to encode every rollout paths 
				to feed the I2A network
		'''
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.rnn = nn.LSTM(input_size=conv_out_size+1,
                           hidden_size=hidden_size,
                           batch_first=False)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, obs_v, reward_v):
        """
        Input is in (time, batch, *) order
        """
        n_time = obs_v.size()[0]
        n_batch = obs_v.size()[1]
        n_items = n_time * n_batch
        obs_flat_v = obs_v.view(n_items, *obs_v.size()[2:])
        conv_out = self.conv(obs_flat_v)
        conv_out = conv_out.view(n_time, n_batch, -1)
        rnn_in = torch.cat((conv_out, reward_v), dim=2)
        _, (rnn_hid, _) = self.rnn(rnn_in)
        return rnn_hid.view(-1)
```

```python
# rollout policy
class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        fc_out = self.fc(conv_out)
        return self.policy(fc_out), self.value(fc_out)
```

```python
class I2A(nn.Module):
    def __init__(self, input_shape, n_actions,
                 net_em, net_policy, rollout_steps):
        super(I2A, self).__init__()

        self.n_actions = n_actions
        self.rollout_steps = rollout_steps
				
				# Main policy (A2C)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32,
                      kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        fc_input = conv_out_size + ROLLOUT_HIDDEN * n_actions

        self.fc = nn.Sequential(
            nn.Linear(fc_input, 512),
            nn.ReLU()
        )

        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

        # used for rollouts
        self.encoder = RolloutEncoder(EM_OUT_SHAPE)
        self.action_selector = \
            ptan.actions.ProbabilityActionSelector()
        # save refs without registering
				# rollout policy
        object.__setattr__(self, "net_em", net_em)
        object.__setattr__(self, "net_policy", net_policy)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 255
        enc_rollouts = self.rollouts_batch(fx)
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        fc_in = torch.cat((conv_out, enc_rollouts), dim=1)
        fc_out = self.fc(fc_in)
        return self.policy(fc_out), self.value(fc_out)

    def rollouts_batch(self, batch):
        batch_size = batch.size()[0]
        batch_rest = batch.size()[1:]
        if batch_size == 1:
            obs_batch_v = batch.expand(
                batch_size * self.n_actions, *batch_rest)
        else:
            obs_batch_v = batch.unsqueeze(1)
            obs_batch_v = obs_batch_v.expand(
                batch_size, self.n_actions, *batch_rest)
            obs_batch_v = obs_batch_v.contiguous()
            obs_batch_v = obs_batch_v.view(-1, *batch_rest)
        actions = np.tile(np.arange(0, self.n_actions,
                                    dtype=np.int64), batch_size)
        step_obs, step_rewards = [], []

				# generate rollout trajectories using experiences in the batch
        for step_idx in range(self.rollout_steps):
            actions_t = torch.LongTensor(actions).to(batch.device)
            obs_next_v, reward_v = \
                self.net_em(obs_batch_v, actions_t)
            step_obs.append(obs_next_v.detach())
            step_rewards.append(reward_v.detach())
            # don't need actions for the last step
            if step_idx == self.rollout_steps-1:
                break
            # combine the delta from EM into new observation
            cur_plane_v = obs_batch_v[:, 1:2]
            new_plane_v = cur_plane_v + obs_next_v
            obs_batch_v = torch.cat(
                (cur_plane_v, new_plane_v), dim=1)
            # select actions
            logits_v, _ = self.net_policy(obs_batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            probs = probs_v.data.cpu().numpy()
            actions = self.action_selector(probs)
        step_obs_v = torch.stack(step_obs)
        step_rewards_v = torch.stack(step_rewards)
				# rollout encoder 
        flat_enc_v = self.encoder(step_obs_v, step_rewards_v)
        return flat_enc_v.view(batch_size, -1)
```

### Results

![스크린샷 2021-09-20 오후 5.42.41.png](https://github.com/RL-Study-On/contrast/blob/master/assets/09-05-book-minji/4.png?raw=true)

The reward (left) and steps (right) for the I2A test episodes

![스크린샷 2021-09-20 오후 5.43.12.png](https://github.com/RL-Study-On/contrast/blob/master/assets/09-05-book-minji/5.png?raw=true)

I2A training episodes (one life): the reward (left) and steps (right)
