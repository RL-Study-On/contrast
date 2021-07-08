[Paper Link](https://arxiv.org/abs/1707.01495)

## Introduction

Sample-efficient learning for sparse, binary rewards

Manipulating robot arms

- 3 tasks: pushing, sliding, pick-and-place
- Rewards: 1 when completed, 0 when not → sparse & binary

Hindsight Experience Replay (HER)

- Human learn from undesired outcomes as from the desired one
- HER deploys that kind of ability of human
- Applicable whenever there are multiple goals

Replay each episode with a different goal than the one the agent was trying to achieve, e.g. one of the goals which was achieved in the episode.

## Background

### Deep Deterministic Policy Gradients (DDPG)

DDPG is a model-free RL algorithm for continuous action spaces. In DDPG we maintain two neural networks: a target policy (also called an actor) $$\pi: S\rarr A$$ and an action-value function approximator (called the critic) $$Q: S \times A \rarr \R$$. The critic’s job is to approximate the actor’s action-value function $$Q^\pi$$.

Episodes are generated using a behavioral policy which is a noisy version of the target policy, e.g. $$\pi_b(s) = \pi(s)+N(0,1)$$

- Training critic: similar way as the Q-function in DQN but the
targets $$y_t$$ are computed using actions outputted by the actor, i.e. $$y_t = r_t + \gamma Q(s_{t+1}, \pi(s_{t+1}))$$.
- Training actor: mini-batch gradient descent on the loss $$L_a = -E_sQ(s,\pi(s))$$, where s is sampled from the replay buffer.

### Universal Value Function Approximators (UVFA)

UVFA is an extension of DQN when there is more than one goal. Let G be the be the space of possible goals. Every goal $$g \in G$$  corresponds to some reward function $$r_g : S \times A \rarr \R$$.  The goal stays fixed for the whole episode. The Q function now depends on state-action pair but also on a goal. 

→ $$Q^\pi (s_t, a_t, g)= E[R_t\|s_t,a_t,g]$$ 

### Multi-goal RL

Training an agent to perform multiple tasks can be easier than training it to perform only one task

Every goal $$g \in G$$ corresponds to some predicate $$f_g : S \rarr \{0,1\}$$ and the agent's goal is to achieve any state s that satisfies $$f_g (s) = 1$$, in other words, to complete some task. 

We assume that there is given a mapping $$m: S \rarr G \ s.t. \ \forall_{s\in S} f_{m(s)} (s) =1$$. In the case where each goal corresponds to a state we want to achieve, i.e., $$G = S$$ and $$f_g (s) = [s=g]$$, the mapping m is just an identity. For the case of 2d state and 1d goals this mapping is also very simple $$m((x,y)) = x$$.

## Hindsight Experience Replay (HER)

Reward function $$r_g(s,a) = - [f_g(s)=0]$$ didn't work well because it is sparse and not informative

→ Introduce HER

### Algorithm

When experiencing some episode $$s_0, s_1, ...,s_T$$, HER store in the replay buffer every transition $$s_t \rarr s_{t+1}$$ not only with the original goal used for the episode but also with a subset of other goals. 

Goal being pursued in episode influences the agent’s actions but not the environment dynamics and therefore we can replay each trajectory with an arbitrary goal assuming that we use an off-policy RL algorithm like DQN or DDPG. Set of additional goals used for replay leverages HER.

![/_posts/Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.06.09.png](Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.06.09.png)

## Experiment

### Video

![/Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.09.10.png](Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.09.10.png)

[https://goo.gl/SMrQnI](https://goo.gl/SMrQnI).

### Environments

7-DOF Fetch Robotics arm which has a two-fingered parallel gripper, simulated using the MuJoCo physics engine.

![/Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.10.53.png](Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.10.53.png)

Different tasks: pushing (top row), sliding (middle row) and pick-and-place (bottom row). The red ball denotes the goal position.

- Policy: represented as MLPs with ReLU
- Training: DDPG Algorithm
- Optimizer: Adam

**Tasks**

- Pushing: box is placed on a table in front of the robot and the task is to move it to the target location on the table. The robot fingers are locked to prevent grasping. The learned behaviour is a mixture of pushing and rolling.
- Sliding: puck is placed on a long slippery table and the target position is outside of the robot’s reach so that it has to hit the puck with such a force that it slides and then
stops in the appropriate place due to friction.
- Pick-and-place: target position is in the air and the
fingers are not locked. To make exploration in this task easier we recorded a single state in which the box is grasped and start half of the training episodes from this state.

**States**

- Angles and velocities of all robot joints
- Positions, rotations and velocities (linear and angular) of
all objects

**Goals**

- Desired position of the object with some fixed tolerance

**Rewards**

- $$r(s,a,g) = -[f_g(s')=0]$$
- $$s'$$ is state after the excution of the action
- Sparse, binary reward

**Obesrvations**

- Absolute position of the gripper
- Relative position of the object and the target
- Distance between the fingers
- Linear velocity of the gripper and fingers
- Linear and angular velocity of the object

**Actions**

- 4-dimensional
- 3 dimensions: desired realtive gripper position at the next timestep
- 1 dimension: desired distance between 2 fingers

### Performance of HER

![_posts/Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.22.08.png](Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.22.08.png)

Learning curves for multi-goal setup. An episode is considered successful if the distance between the object and the goal at the end of the episode is less than 7cm for pushing and pick-and-place and less than 20cm for sliding. The results are averaged across 5 random seeds and shaded areas represent one standard deviation. The red curves correspond to the future strategy with k = 4 while the blue one corresponds to the final strategy.

DDPG without HER is unable to solve any of the tasks7
and DDPG with count-based exploration is only able to make some progress on the sliding task. On the other hand, DDPG with HER solves all tasks almost perfectly.

![_posts/Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.23.49.png](Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.23.49.png)

Learning curves for the single-goal case

DDPG+HER performs much better than pure DDPG even if the goal state is identical in all episodes. More importantly, HER learns faster if training episodes contain multiple goals, so in practice it is advisable to train on multiple goals even if we care only about one of them.

### Goal strategies

We evaluted different strategies for choosing additional goals to use with HER.

- future: replay with k random states which come from the same episode as the transition being replayed and were observed after it
- episode: replay with k random states coming from the same episode as the transition being replayed
- random: replay with k random states encountered so far in the whole training procedure

![_posts/Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.27.38.png](Hindsight%20Experience%20Replay%200ee1f876b5f845d994c3c62d29374e40/_2021-07-01__2.27.38.png)

Number of additional goals used to replay each transition with

All of these strategies have a hyperparameter k which controls the ratio of HER data to data coming from normal experience replay in the replay buffer.

In all cases future with k equal 4 or 8 performs best and it is the only strategy which is able to solve the sliding task almost perfectly.

## Implementation

[https://github.com/TianhongDai/hindsight-experience-replay](https://github.com/TianhongDai/hindsight-experience-replay)

```python
class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
```
