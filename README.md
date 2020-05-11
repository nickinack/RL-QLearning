# RL-QLearning
### Solved Mountain Car and Taxi problems using simple Reinforcement Learning by implementing Q-Tables

To execute and run these files, download and open them in your Jupyter lab/ Jupyter notebook

If you do not have jupyter notebook/lab, download them using the following code :
```bash
pip install jupyterlab
```
```bash
pip install notebook
```

#### After installing , you may open your notebooks using the following command:

```bash
jupyter lab
```
```bash
jupyter notebook
```

#### Explanation
The weight for a step from a state {\displaystyle \Delta t}\Delta t steps into the future is calculated as {\displaystyle \gamma ^{\Delta t}}\gamma ^{{\Delta t}}, where {\displaystyle \gamma }\gamma  (the discount factor) is a number between 0 and 1 ({\displaystyle 0\leq \gamma \leq 1}0\leq \gamma \leq 1) and has the effect of valuing rewards received earlier higher than those received later (reflecting the value of a "good start"). {\displaystyle \gamma }\gamma  may also be interpreted as the probability to succeed (or survive) at every step {\displaystyle \Delta t}\Delta t.

Before learning begins, {\displaystyle Q}Q is initialized to a possibly arbitrary fixed value (chosen by the programmer). Then, at each time {\displaystyle t}t the agent selects an action {\displaystyle a_{t}}a_{t}, observes a reward {\displaystyle r_{t}}r_{t}, enters a new state {\displaystyle s_{t+1}}s_{t+1} (that may depend on both the previous state {\displaystyle s_{t}}s_{t} and the selected action), and {\displaystyle Q}Q is updated. The core of the algorithm is a Bellman equation as a simple value iteration update, using the weighted average of the old value and the new information:

{\displaystyle Q^{new}(s_{t},a_{t})\leftarrow \underbrace {Q(s_{t},a_{t})} _{\text{old value}}+\underbrace {\alpha } _{\text{learning rate}}\cdot \overbrace {{\bigg (}\underbrace {\underbrace {r_{t}} _{\text{reward}}+\underbrace {\gamma } _{\text{discount factor}}\cdot \underbrace {\max _{a}Q(s_{t+1},a)} _{\text{estimate of optimal future value}}} _{\text{new value (temporal difference target)}}-\underbrace {Q(s_{t},a_{t})} _{\text{old value}}{\bigg )}} ^{\text{temporal difference}}}


{\displaystyle Q^{new}(s_{t},a_{t})\leftarrow \underbrace {Q(s_{t},a_{t})} _{\text{old value}}+\underbrace {\alpha } _{\text{learning rate}}\cdot \overbrace {{\bigg (}\underbrace {\underbrace {r_{t}} _{\text{reward}}+\underbrace {\gamma } _{\text{discount factor}}\cdot \underbrace {\max _{a}Q(s_{t+1},a)} _{\text{estimate of optimal future value}}} _{\text{new value (temporal difference target)}}-\underbrace {Q(s_{t},a_{t})} _{\text{old value}}{\bigg )}} ^{\text{temporal difference}}}

where {\displaystyle r_{t}}{\displaystyle r_{t}} is the reward received when moving from the state {\displaystyle s_{t}}s_{{t}} to the state {\displaystyle s_{t+1}}s_{t+1}, and {\displaystyle \alpha }\alpha  is the learning rate ({\displaystyle 0<\alpha \leq 1}0<\alpha \leq 1).

The action for a given state during an episode is decided using Markov Decision process (MDP). After MDP statements, the system makes a transition using the following statement:

```python
env.step(action)
```

#### Further comments and instructions have been included in the code.
