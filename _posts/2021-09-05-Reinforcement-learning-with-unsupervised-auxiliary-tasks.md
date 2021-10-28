Jaderberg, Max, et al. "Reinforcement learning with unsupervised auxiliary tasks." arXiv preprint arXiv:1611.05397 (2016).

### Abstract

- Maximizes many other pseudo-reward functions simultaneously
- Learn representation upon extrinsic rewards
- First-person, three-dimensional Labyrinth tasks

### Overview

![image title](https://github.com/RL-Study-On/contrast/blob/master/assets/09-05-review-minji/1.png?raw=true)
- Base A3C agent: CNN-LSTM agent trained on-policy with the A3C loss
- Replay buffer: observations, rewards, and actions of A3C agent are stored
- Pixel control: maximise change in pixel intensity of different regions of the input
- Reward prediction: given the recent frames, predict the reward that will be obtained in the next step
- Value function replay: further training of the value function using the agent network

### Auxiliary control tasks

- Additional pseudo-reward functions in the environment
- Define auxiliary control task c by a reward function $$r^{(c)} : \mathcal{S} \times \mathcal{A} \rarr \R$$
- Given a set of auxiliary control tasks C, let $$\pi ^ {(c)}$$ be the agent's policy for each auxiliary task $$c \in C$$ and $$\pi$$ be the agent's policy on the base task. Objective is to maximise total performance across all auxiliary tasks,  $$\underset{\theta}{\operatorname{argmax}} \mathbb{E}_\pi [R_{1:\infty} ] + \lambda _c \underset{c\in C}{\operatorname{\sum}} \mathbb{E}_{\pi_c}[R_{1:\infty}^{(c)}]$$ where $$R_{t:t+n} ^ {(c)} = \sum ^n _{k=1} \gamma ^k r_t^{(c)}$$  is the discounted return for auxiliary reward $$r^{(c)}$$, and $$\theta$$ is the set of parameters of $$\pi$$ and all $$\pi ^ {(c)}$$s.
- $$\lambda _c$$ balance improving globar reward $$r_t$$ with improving performance on the auxiliary tasks
- To efficiently learn to maximize many different pseudo-rewards in parallel from a single stream of experience, it is necessary to use off-policy methods → Value-based methods that use Q-learning
- Pixel changes: Changes in perceptual stream often corresponds to important events in an environment. Agents learn a separate policy for maximally changing the pixels in each cell of $$n \times n$$ non-overlapping grid placed over the input image
- Network features: Policy or value networks of agent learn to extract task-relevant high-level features of the environment → activation of hidden unit itself can be an auxiliary reward → agent learn a separate policy for maximally activating each of the units in specific hidden layer

### Auxiliary reward tasks

- Recognize states that lead to high reward and value
- Good representation of rewarding states → good value functions → good policy
- Reward is encountered very sparsely → long time to train feature extractor adapt at recognizing states with high reward → introduce the auxiliary task of reward prediction
- Train the reward prediction task on sequence $$S_\tau = (s_{\tau -k} , s_{\tau-k+1}, ..., s_{\tau-1})$$ to predict the reward $$r_\tau$$ but sample $$S_\tau$$ in skewed amnner so as to over-represent rewarding events when rewards are sparse, i.e., $$P(r_\tau \neq 0) =0.5$$.

### Experience replay

- Split the samples into rewarding and non-rewarding subsets and replay equally from both subsets → rewarding states will be oversampled → learnt far more frequently than when sampled directly from the behavior policy
- Value function replay:  resampling recent historical sequences from the behaviour policy distribution and performing extra value function regression in addition to the on-policy value function regression in A3C → exploits newly discovered features shaped by reward prediction (skewed distribution is not used here)
- Increase the efficiency and stability of the auxiliary control tasks

### UNREAL Agent

- Primary policy is trained with A3C: learn from parallel streams of experience, updated online using PG methods, uses RNN to encode the complete history of exeperience
- Auxiliary tasks are trained on very recent sequences of experience off-policy by Q-learning; use simpler feedforward network
- UNREAL algorithm optimizes single combined loss function $$\mathcal{L}_{UNREAL} (\theta) = \mathcal{L}_{A3C} + \lambda _ {VR} \mathcal{L}_{VR} + \lambda _ {PC} \underset{c}{\sum} \mathcal{L}_Q ^ {(c)} + \lambda_{RP}\mathcal{L}_{RP}$$ where $$\lambda_{VR}, \lambda_{PC}, \lambda_{RP}$$ are weighting terms on the individual loss components
- Policy gradient loss $$\mathcal{L}_\pi = - \mathbb{E}_{s \sim \pi} [R_{1:\infty}]$$
- A3C loss $$\mathcal{L}_{A3C} \approx \mathcal{L}_{VR}+\mathcal{L}_\pi -\mathbb{E}_{s\sim \pi} [\alpha H(\pi (s,\cdot,\theta)]$$ is minimized on-policy
- Value function loss $$\mathcal{L}_{VR} = \mathbb{E}_{s \sim \pi} [(R_{t:t+n} + \gamma ^ n V(s_{t+n+1}, \theta ^ -) -V(s_t, \theta))^2 ]$$ is optimized from replayed data, in addition to A3C loss
- Auxiliary control loss $$\mathcal{L}_{PC} = \underset{c}{\sum} \mathcal{L}_Q ^ {(c)} = \underset{c}{\sum}\mathbb{E} [(R_{t:t+n} + \gamma ^ n \max_{a'}  Q ^{(c)} (s', a';\theta ^ -)-Q^{(c)} (s, a;\theta))^2 ]$$ is optimized off-policy from replayed data, by n-step Q-learning
- Reward prediction loss $$\mathcal{L}_{RP}$$ is optimized from rebalanced replay data: multiclass cross-entropy classification loss across three classes (zero, positive, or negative reward) in the paper

### Result

![image title](https://github.com/RL-Study-On/contrast/blob/master/assets/09-05-review-minji/2.png?raw=true)
