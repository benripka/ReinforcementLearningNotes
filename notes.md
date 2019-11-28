
<h1>Reinforcement Learning Glossary of Terms</h1>

- **Reinforcement Learning:** The field of AI concerned with deciding which actions a software agent ought to take in each state of an environment so as to maximize a defined cumulative reward signal over time. In RL you think of the world (or at least problem space) as a set of states. These states are snapshots of all the details (data) that are relavent to the agent. The agent is able to take actions, that will result in a change in state of the environment. Notice the level of generality? This allows the concepts of RL to be used on an extremely wide scope of problems. 

- **Reward:** The reward is a designer defined signal that encapsulates the goal that the agent is trying to achieve. It is simply a value that the agent recieves when it transitions from a state to another state by performing an action. All these values considered sequentially form the reward signal. Deffining the reward function is our only way of telling the agent whether its doing a good job. In other words its our only way of *defining the goal* Ideally the reward function should only encapsulate the actual goal. In the example of chess, if the goal is simply to win, the rewward function should only reward the agent on wins. If you start rewarding the agent for taking the opponents players, or taking the queen, you run the risk of limiting the agent from finding the true optimal solution to your goal. In some cases though it can be useful to impart some knowledge into the reward funcion to limit the scope of possibilities in compution heavy problems. It is possible for the reward function to be *non-stationary* what this essentially means is that the probability of getting a certain reward given a scertain state-action pair may to remain the same throughout the agents life. For example playing many many games of chess against your friend. If you found a really good move that works a few times in a row, chances are they will catch on and see whats comming, making it less advantageous. These type of problems are much more challenging to solve.
- **action-selection methods:** This is a rule that the agent will use to choose which action it will choose given a value function. The method may be greedy, e-greedy. Greedy action selection is only good if the rewards signal is not noisy at all, or the reward variance = 0. Noisier reward signals (those that have rewards of variable value for the same action-value) benefit from more exploration therefore larger epsilon. Another actoin -selection method is called the UCBS or Upper Confidence Bound Selection. It uses a could of how many times each action has been selected from a state to choose either very valuable actions or actions that have not been taken much. In this case c is a parameter to be varied, t is the timestep and N gives teh number of times that teh action has been selected. 
$$
A_t=argmax_a[Q_t(a) + c\sqrt{ln(t)/N_t(a)}]
$$
This function is really not that great though when you deal with large state spaces.

- **Sample Averaging:** Fundamentally sample averaging is just taking all the average returns that were had from a state-action pair and averaging them. There is a simple computationally / memory friendly way to do this given by:
$$
NewEstimate = OldEstimate + stepSize(Target - OldEstimate)
$$
This stepsize parameter can be seen as the weighting on how much the current error in guess should matter. You can have it be constant, but it is often handy to let it vary throughout the agents life by making it some sort of function. You might choose the step size to be exponentially decreasing with the steps. This is a kin to saying "care alot about the error in guess initially, but less and less as you take more and more steps from this state". The step size is bounded between 0 and 1 as it serves only to attenuate.
  
- **Value Funciton**: The value function gives the value of each state. The *value* of a state provides a notion of how good it is to be in a given state, given the reward function (Thus goal, if written properly). This is obviously something that needs to be learned. If you know the value function, you can simply always choose to take an action that will end you up in the most valuable next states. The value is different than the reward, because value takes into consideration the future rewards that the agent has experienced in the past form this state. The degree to which it places importance on those future values will very by algorithm. The value function is essentially what the agent is hoping to learn.

- **Model**: In RL there is always some form of environment. A model is used to encapsulate the transition dynamics of the environment, that is, it returns the next state that the agent will likely end up in if it takes an action in a given state. The policy defines which action SHOULD be taken in each state, the model can help show what new state will result from that acion in that state. It gives the TRANSITIONS. It is important to realize that the model IS NOT always the same as the environement. This is only true for simple or virtual problems. In real world applications you may still use a model to help learn, but this will likely be some idealization of reality. Learning from a model is called  **planning** and there are a whole host of planning algorithms. RL algorithms can be model-free or model-based. Some RL algorithms do not start with one but are actually able to devise and environement model on the fly and add a planning step on aswell! See Dyna-Q for an example. Model-free learning is usually more interested in the q(s, a) functions, or state-action value functions. This is because knowing the value of a state is great, but if you don't know where an action will take you (s') from the state, its not that useful.

- **Exploration vs. Exploitation:** This is a key tradeoff in RL algorithms that shows up again and again. A given algorithm must alternate between trying to choose actions it knows from experience to be valuable, whilst still probing (occasionally taking) unknown or less known state-actions to monitor for potential new valueble actions. Exploration is done to avoid getting stuck in a suboptimal policy at any given time. Exploitation is the heart of why we are doing RL. Should you go with what you already know works, or look for something better? You will often hear the word **greedy**
 or **act greedily** used to describe an algorithm / agent that heavily or exclusively exploits the current state-action of highest value.

- **Greedy Behaviour:** an agent is said to act greedily if it chooses to exploit, or take the actions in each state that it believes will maximize the return from that state. 

- **Delayed Reward:** In RL, whether an action in a state is good or bad is not an immediate property of taking the action. Contrast this to supervised learning where you get instant rewards from the data: action = label as dog, reward = -100 because its a picture of a cat. Each action is independent from the others. The difference is, in RL, the value of a state action depends on the causal chain of future state-action pairs. Therefore an agent must seek to learn what the potential future rewards may be for each following state-action aswell as the immediate rewards. The concept of **return** is introduced to formalize this.
  
- **Policy**: An agents policy is used to map states to actions. The policy is a function that can either be stochastic or deterministic. A deterministic policy will be a function of the state, that tells the agent what action to take in each state. A stochastic policy is a function of state aswell, but returns the entire probability distribution over the potential actions in that state. In other words it tells you how likely it is that each action be taken (under this policy). Using a stochastic policy you can actually come up with a deterministic policy. For instance you could just use the stochastic policy to get the distribution over all actions and then choose the most likely action in each state. Since now only one action is given in each state you could call this entire selection process a deterministic policy.


- **Markov Decision Process:** The MDP is a discrete mathematical process that is used to model stochastic decision making processes. To be an MDP the set of states and actions must be finite and must obey the markov property. The entire MDP may be summerized by the Dynamics function. It is used to give the probability of a given outcome (getting a certain reward for transitioning from a certain state to a certain new state via an action). This an be written in many different forms as seen below:


$$
P(s', r|s, a) \\
P(s'|s, a) = \sum_{reR}P(s', r|s, a) \\
r(s, a) = \sum_{r}\sum_{s'}P(s', r|s, a) 
$$



- **Markov Property:** This property says that all the information needed to predict the future is contained within the current state representation. What this means is that if the agent finds itself in a state, then it can go to a set of new states by performing action, and that set of potential next states does NOT depend on the things that the agent did before it got to the current state. So there CANNOT be a state D such that the ONLY WAY to get to it is by going from state A -> B -> C -> D. The Markov property says that we must define states such that no matter where the agent was before C, if it ends up in C it can go C -> D. With probabilities we can say $P(a|s_t) = P(a|s_t, s_{t-1},...)$. This naturally extends to rewards since the reward function is a function of state-action-nextstate pairs.

- **Return:** The return at a time step t is the summation of all the rewards that are recieved at all time steps in the future: $G_t = R_{t+1} + R_{t+2} + ... + R_{T}$. Usually though we choose to model the returns as a **discounted return**. Discounting puts less value on rewards received farther away from the timestep in question. We usually attenuate the future reward signal with a decreasing exponential to achieve this. By increasing the value of $\lambda$ we can make the agent pay closer attention to the rewards that are recieved farther in the future. Put another way, the larger lambda is the less it discounts future rewards. $\lambda \in (0, 1]$. 

$$
G_t = R_{t+1} +\lambda R_{t+2} +  \lambda^2R_{t+3}... =\sum_{k=0}^{\infty} \lambda^k R_{t+k+1}
$$

- **Consistancy Condition of Return:** This an important property of the return, saying that it can be described as a recursive function. This about why this might be true from above. This feature is exploited in dynamic programming with the bellman equation: $G_t=R_{t+1} + \lambda G_{t+1}$ 
   
- **Bellman Equation:** This equation is used as a recursive function to compute the value funcion of a policy $\pi$. It essentially goes to each state-action pair, and for each possible next state and reward for going to it it computes the new return (using each reward, next state, and the consistency condition). Now each of these values are attenuated by their likelyhood of being true (given by $P(s', r|s, a)$ ) and summed together. For each action this is attenuated by the probability of the action even being taken under the current policy (this is the deffinitino of $\pi(s)$. This is a very intimidating formula and tbh there's alot going on. But its seriously worth the 10, 20, 30 minutes it might take of staring at it to make sense of it. The key for me was to rememeber it is recursive and so will follow each path down the entire tree of possible paths before starting to actually crunch numbers, and then work its way back (Think dynamic programming). Also remember $\pi$ and then entire markov process $P(s', r|s, a)$ are both stochastic. They return probabilities, so although we're doing like a triple for loop over litterally EVERY possibility, only the actions that are likely taken under the policy and the transistions likely to happen will really count for much in the sum!

$$
v_\pi(s) = \sum_a \pi(a|s) \sum_{r, s'}P(s', r|s, a)(r + \lambda v_\pi(s'))
$$

- **Generalized Policy Iteration:** GPI is the essence of RL. It incrementally improves the policy that an agent used to select actions in any given state. This is quantified by saying that each time the agent runs through an episode, the value function of the policy will be greater or equal to the value function of the policy used the last time. GPI is a cycle. The agent uses a policy and evaluates it to find the V(s) for the whole episode. It then chooses its new policy to be: Act greedily with respect to the estimated value function of the last policy. So if any actions were found to be less optimal, they will no longer be taken! This can be repeated again and again until convergance to an optimal policy and therefore optimal value function. GPI will always result in a q(s, a) and v(s) that are better or jsut as good as the previous one, on every step. GPI has an inherent tug of war between policy evaluation and policy improvement. The goal is to have a policy that does not change upon evaluation. By evaluating the policy you make it no longer greedy with respect to its own value function (you changes V(s). V(s) no includes what you did right/wrong in the last episode). By improving the policy to be greedy with respect to this new thing you teh value function estimate is no longer correct! So you have to re evaluate and it all starts again!
  ```
  V(s) = random function 
  pi = random actions

  while(change in V(s) is significant)
  {
    foreach state in states
    {
      V(s) = Bellman(s)
    }
  }

  foreach state in states
  {
    pi(s) = max actions according to v(s)
  }

  if(policy unchanges)
  {
    Must be done. Return.
  }
  ```

- **Monte Carlo Methods:** MC methods are a set of algorithms to perform MODEL-FREE learning. This means that we do not have the transistion dynamics of the environment ($P(s', r|s, a)$) and therefore cannot use DP even if we wanted to. MC usually uses some form of random policy iteration to gather sampled returns from different states, and in so doing constructs a value funciton. MC will use the average of all the values it saw in previous episodes to update the value function over the states. This is used instead of the recursive bootstrapping in DP. MC can either be **First-visit** or **every-visit**. The former means that only the first visit to a state in a given episode is added to the average value of the state. The latter means every time it goes to that state (in one episode) is considered. Recall that a policy might go through the same state many times (loop back). The first visit assumption makes things simpler and less computationally heavy. Monte Carlo is very similar to DP, the main difference is the way that we *evaluate* the current policy. The policy improevement is over just the same )(greedy with respect to the value function). Since monte carlo is model-free, it will use Q(s, a) rather than the value function, since it does not know the next states from taking actions. Monte carlo can use exploring starts or epsilon greedy to maintain exploration.
- **Off-Policy Learning:** OPL is when the agent learns about a policy $\pi$ by interacting with the environment under a different policy b. It allows the behavioural policy to vary wildly and explore alot but lets $\pi$ still converge on the optimal. Off-policy learning needs importance sampling to be useful.
- **Importance sampleing:** This is a way to represent in a single number how difference two policy's are from eachother. If one policy says in every state, you should take action A and the other says always take action B, the ratio would be almost 0. If the two policies tend to take the same actions in the same states, the importance sampling ration will be high. This means that the returns calculated from an off-policy interaction with the environment can be weighted with how IMPORTANT they are, or put another way how relavent they are to the policy we want to learn about. They will be more relavent if the policies are similiar. The importance sampling ration is computed as the product of all rations of state action probabilities in each state. In other words the ration of the two policies in each state. 

$$
\prod_{k=t}^T \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$
- **Exploring-starts:** This is a method used to increase the amount of exploration that an agent does. It starts the agent off at a random state and allows it to complete the episode from there. This results in more accurate average value estimates for all states because each state will be visisted more uniformely. 
- **Epsilon-greedy:** this is an action selection algorithm that selects an action based on a policy with probability 1-$\epsilon$ and chooses a random one otherwise. So it takes a random action with probability $\epsilon$.
- **State-Action Function:** As we know, the value function of a plicy gives the expected return from a given state following a policy. The state-action function gives the expected return from taking an action in a state, and then following the policy thereafter.

- **Bellman Optimality Equation:** This equation assumes that you have an optimal value function. 
- **Temporal Difference Learning:** TD is an RL algorithm for online model-free learning. Unlike MC methods the agent does not need to wait until the end of the episode to update the value estimates at each state! Rather it does so on each step. It uses a notion of how wrong it was in going from one state to the next by taking an action, then seeing where it ends up. It then updates the value of the previous state with the difference between the new return it just witnessed (reward from transition and discounted estimated return at next state) and the expected value (current value of last state) weighted according to the step size. Notice that $R_{t+1} + \gamma V(s_{t+1}) - V(S_t)$ is the error between the value (expected return) that it just witnessed and the previously expected return. This seems strange because we are still using a previous estimate of $V_{t+1}$ to compute the new expected return, but rememeber that we are infact adding something new to the equation. We're adding that $R_{t+1}$ into the understanding of the agent. This is intuitively what makes it 'learn' step by step. The difference intuitively between TD and MC is that TD can pickup on nuances on a changing environment that MC may not excplicity detect. If something changed that made the state transistion A -> B less valuable, MC would report the entire policy as equaly less valuable, whereas since TD updates on each transition, it will report the drop in value RIGHT AT the state A. This is because MC works at teh scale of entire episodes, or policy run-througs. TD works step by step changing values online. <h2>Exploration / Exploitation in TD</h2> There are many algorithms in TD to balance eploration and exploitation, which we need to do since we are learning from experience. Similar to TD we are interested in learning about Q(s, a) since we do not have a model of the env. 
- **SARSA** uses an epsilon greedy policy to update the state-action value (Q(s, a)). SARSA stands for State Action Rewards next State next Action. This maintains exploration and usually ends up being more concervative near hazzards. It will converge to the epsilon-soft optimal solution.
- **Q-Learning** is very similar to SARSA. It will choose an action with epsilon greedy, and use it to get to a next state with a reward. The difference is that it will actually update OFFPOLICY, because it will use the expected reward from the max action from the next state rather than choosing another expected random to update from. Subtle but powerful. The actions that are actually acted on are not the same as the actions used to learn about Q(s, a). This will prevent the agent from decreasing the value of a state-action pair based on a potential shitty random action being taken in the next state. It protects the UPDATES from counting random actions, but lets the actual interactions still take them. You can say that SARSA learns with respect to its epsilon policy. Q learning behaves with respect to its epsilon policy but LEARNS with respect to its optimal (max) policy.
- **Expected SARSA:** Now Expected SARSA is also offline because it does not learn from its the policy it behaves with. Instead, it learns with the sum of the Q values of taking all actions possible from the next state, attenuated by their probability of actually happening! Smart. So Q learning basically said I'm going to ignore the actions I actually take, and instead update based on the action that currently seems best at the next state. Expected SARSA says no, I'm going to update the Q value based on the value of taking each action possible from the next state, but put more emphasis on those that are actually likely! 
- **Double Q Learning** basically runs two estimates in parallel. One of the two wilil be updated with 50% chance on each step. In the update, they will update based on the OTHER Q functions max. This avoids the positive biasing of Q learning that comes from jsut takin  the max every single time, which can cause very slow learning right off the bat before thing settle to near constants. 


$$
V(S_t) = V(S_t) + \alpha[R_{t+1} + \gamma V(s_{t+1}) - V(S_t)]
$$
- **Function Approximation:** Function approximation introduces a new way to update the value of states. Previously and update only improved the value on one state. Functino approximation allows for the rewards recieved from one state-transition to affect the values of multiple states. Therefore the method can gain an understanding of features of the environment that might transcend states. To approximate a state-action value function, you use passed experience as the training data for some kind of supervised learning methods (neural nets, etc.). 
- **Prediction Objective:** There is an inherent tradeoff from using function approximation instead of tabular methods. Since there is usually less weights than states, by changing the weights to make the estimate of one states value higher, we inevitably make the estimate of another state worse. This means we now need some way to tell which states are important, and which are not. To do so we have the mean square value error, which is computed using the probability distribution $\mu(s)$ that is chosen to represent how relevant a state is. It is often chosen to give an idea of how frequently a state is visited by the agent. With it we can compute a standard error of the estimated value function, where errors in states we give more of a fuck about are more heavily represented:

$$
VE(w)=\sum_{s\in S}\mu(s)[v_\pi(s)-v(s, w)]^2
$$
- This function now presents the goal of RL with function approximations: Find the weights w that will minimize this error. This will look like $VE(w^*) \leq VE(w)$. In function approximation you use update the weights in the direction of te gradient. This is called **gradient descent**. Below the value of $U_t$ is some approximation of the actual weight values $V(s, w)$. 
 
$$
w_{t+1}=w_t + \alpha [U_t + v(s, w_t)] \nabla v(s, w_t)
$$
- **Linear methods:** in this  case we assume that the value function can be found as a linear system over the states, and the weights correspond to the factors by which the value of each state is decided, therefore we can say:
$$
v(s, w) = w^T \cdot x(s)
$$
- One another simplification that comes from the linear feature vector is that the gradient simply becomes x(s), since all the other derivatives dissapear in each place. This in turn makes the weight update alot easier.
- **Feature construction** for linear methods becomes very important, and can be a great way to add domain specific knowledge to the AI, molding what kind of state features it should be looking for. One of the limitations however of the linear form is that it cannot address dependencies betweeen the actions of the agent and multiple features. You cannot communicate "in presence of features A and B, do something, but not if only A or b". When you decide upon the feature vector you want to use, you make certain guesses about the types of state interactions that will likely go down between the states. The equations in the feature vector will form the basis of the feature space. 
- **Coarse Coding** is a way to use binary features to assign value to a state. each feature is a subset of the state space, and wether or not the state exists within a particular subset gives that subset a 0 or 1. You can then "locate" the state in the space, and furthermore assign it  a value based on how many subsets it turns on. This works like a giant ven diagram of sorts. By varyingthe size and number of features you modify the type of generization that occurs over the state space. Furthermore the shape of the subsets of the space will shape the nature of the generalization further. Using different parameters in this sense will have a large effect on the way the function is approximated (will differentiate between levels of nuance).
- Tile coding is the best modern form of course coding. To do it you basically break the space up into many "tilings" which are essentially indpendent shifted / resized grids. It is basically doing state aggregation to create a feature, but have the states aggregated in many different ways, one for each tiling. Then you see which meta state the true state ends up in, and together get a more resolved idea of where it is. Tiles will always be the same width, but just offset from eachother by some fraction of the width, therefore ensuring that each is represeted by the same number of tiles (features). This presents a powerful way to start dropping the size of the state space based on locality. It lowers the resolution, but is an interesting way of dropping resolution to save computation over large state spaces. The way that the tiling is offset is up to the designer and has large impacts on the performance of the system.
- Hashing can be used to reduce memory requirments. This quite geniously maps random tiles of each tiling to a smaller subset (4:1 say would decrease memeory to a quarter) but the resolution is lost at the top level, so local nuance between states can often be preserved. You can think of varying the coarse coding partitions as changing the way that the learning system will generalize accross states. 
- **Non-linear Function Approximation** is a general name for methods that attempt to condense the state space to a function approximation by the use of non-linear functions. The most popular method involves the use of artificial neural networks.**Feedforward ANN's** are networks that have inputs to each node that DO NOT depend on their outputs in any way. That is to say that there is no feedback loops in the networks. **Recurent NN's** are what we call networks that DO have those feedback loops. In RL we will not usually be using RNN's. ANN's have an input layer consisting of the same number of nodes as input values. Each of these nodes takes in the single input. Next this is any number of hidden ayers, each layer recieving a linearily weighted version of the output from each node in the previous layer. The node then computes some NONLINEAR function of the input and pushes that through to each node of the next layer.
- This Nonlinear function of the linearily combined inputs is called the **activation function**. The activation function in a node is typically the sigmoid $f(x)=1 /\left(1+e^{-x}\right)$, rectifier or unit step function. It provides some form of threshold on which input combinations are able to properly pass through the NN. The **universal approximation** theorem for NN's with at least a single hidden layer states that with enough training the network will be able to emulate ANY nonlinear function. Therefore in theory you would only ever need a singles layer, however, in practice it is often usedful to use more as each additional layer provides a possibility for more hierarchically abstract representations of the raw input data. 
- FFNN's are trained usually by a gradient descent method that relies on some objective function that is able to rank the performance of the system. In the context of RL this may be the mean squared error in the value function estimation. To do this we need to get an idea of how a change in each weight will impact the change in the overall error. hellooooo partial derivatives? This should sound familiar. In a FFNN the algorithm for training is called backpropagation,  and consistes of a forward pass that will comput the new weights given each new input, and a backward pass that will compute the partial derivatives given the new weights, with repect to the error objective function.
- **Overfitting** is an inherent problem in nonlinear function approximation, and is characterized by the inability to generalize concepts over a set of inputs that the system has not been trained on. The network is fit TO PERFECTLY to he training data and therefore cannot generalize. This is less of a problem for online RL as it has a constant stream of new data flowing in.

<h1>Dynamic Programming</h1>

DP is the unrealistic goal for pretty much all of reinforcement. It assumes complete knowledge of the world around the agent, and uses pure math to find the optimal policy to maximize return. exact solutions are only really possible in the discrete case, or in certain special cases when the state space is continuous or infinit.

<h2>State Functions vs. State-Action Functions</h2>

<p>The state value function gives the expected return for each state in the environment given that the agent starts in that state and acts out the policy from then on out. In the case of episodic tasks this is up until the end of the episode.</p>

<p>The state-action value function gives the expected return if you start in a state s, choose an action a, then from the new state you end up in continue by following the policy pi.</p>

<p>Both functions can be used to assign values to each state in the environment. These give a distilled representation of the way the reward function interacts with the state-dynamics. The are representations of how good the state is with repect to the agents goals (given in the form of reward function).</p>

<p>The principle difference between the two is that the state function assumes that the action that will be taken in a given state is completly determined by the policy. This contrasts the state action function which assumes that different actions can be taken in a given state.</p>

<h1>Monte Carlo Methods</h1>

<h3>Policy Evaluation</h3>

**Policy Evaluation** is the process of deriving the value function for the environemnt under a given policy. 

**The Control Problem** is the problem of estimating the optimal policy, that is the policy that maximizes the overrall return over the entire episode. This is comonly tackled through a process called **generalized policy iteration** or GPI. GPI consists of two steps iterated over and over again. The first is to estimate the value of the current policy. The second is to improve the policy.

**Policy improvement** can be done in many ways. A very common one is act greedily with respect to the evauated stat-action value function. This means just take the action in each state that creates the maximum return! This is the whole point of finding the value of each action in each state in the previous step (policy evaluation). If you always choose the actions in each state that has the maximum value, you will always get a better or at least equally good policy. No matter what. 

<p>MC methods are better than DP for real world applications because they do not require a complete understanding of the environment dynamics or the reward structure of the problem. The MC method involves basically trial and error inside the environment. The state value function is of less use to us as the state-action value function because we do not start our interactions with any policy in mind. It can however be used to evaluate a given policy if we have one of interest</p>

<p>The MC policy evaluation goes like this:</p>
<ol>
    <li>Pick a random state to start in.</li>
    <li>Act out the policy starting from this random state until completion of the episode, storing each (state, reward) tuple as you go.</li>
    <li>Iterate over this aggregated list of tuples in reverse, each updating the running reward and applying the discout. This reverse nature is because the return of a given state is defined recursively depending on the return of the state that follows it, only discounted.</li>
    <li>Do this many times, each time with a different random starting state, and on each new episode average the value functions estimated. This will eventually lead to the evaluated <b>state value</b> fuction of the policy</li>
</ol>

<h3>Better Policy Evaluation with MC</h3><h3>
<p>Evaluating policies is all well and good but it will never allow us to find or estimate an optimal policy. For this, we need to be able to try out different actions in a given state. This is achieved through the application of the <b>state action</b> value function. Recall that this function allows for variation over the selected actions a given state, for all states.</p>

<p>We can use the MC method to evaluate the impact of actions of the agent in a given state in two ways:</p>
<ul>
    <li>By <b>exploring starts</b> which essentially means the agent will run through the policy by starting in every state. It will also run through all possible actions in all possible states and then greedily follow the action that resulted in the highest return. This is like fist</li>
    <li>By <b>non exploring starts</b>. It is very rarely possible to actually start off in any random state. this is due partially to the fact that the states might not even be known yet! Instead we just start at the initial state for every episode. Then, in order to still eventually tryout each action for each state, we continue with an epsilon-greedy approach. This means that we will chose the action that the policy specifies with probability 1 - epsilon, and choose a random action with epsilon probability.</li>
</ul>

**Off-Policy Learning** is done by iterating instead over two seperate policies. One is called the behaviour policy and dictates what the agent actually does. It is not always the best policy, or even a good one, but by using it we get a more wholistic picture of the state action pairs. The other policy is called the target policy, and is the policy that is being "leaned about" using the data from the the behavioural policy. This is called off policy learning because the agent uses data from off a given policy (from the behavrioural policy instead) to learn about that policy. Off-policy learning is in more general and powerful, because of its exploratory nature, but it usually converges more slowly, because alot of unlikely state-action values will be visited.

In order to use the state action values learned from another policy, a few things need to be true. Obviously all of the state actions in the policy you want to learn have to be taken in the behavioural policy.

<h2>Exploring Start Policy Evaluation</h2>

```python 
Evaluate_Policy_With_Exploring_Start(policy):
  let s = select_random_state()
  let a = select_random_action_for_state(s)
  let sar_list = [] # init empty list for state, action, reward
  Loop:
    # Play episode until the end, record every state, action 
    # and reward:
    let (s, r) = make_move(s, a) #make a mov from s using a and get 
                                 #the new state s and the reward r
    if game_over():
       sar_list.add(s, none, r)
       break
    else:
       let a = policy[s]
       sar_list.add(s, a, r)
   End Loop
   # compute the return
   let G = 0
   let sag_list = []# init empty list of state, action, return
   for s, a, r in sar_list.reverse(): #loop from the end
      G = r + gamma * G
      sag_list.add(s, a, G)
    #return the computed list
    return sag_list.reverse()
```

<h2>Policy Evaluation with Greedy Epsilon</h2>

```python 
Evaluate_Policy_Without_Exploring_Start(policy):
  let s = select_start_state() # select the starting state, not a 
                               # random state
  let a = select_random_action_for_state(s)
  let sar_list = [] # init empty list for state, action, reward
  Loop:
    # Play episode until the end, record every state, action 
    # and reward:
    let (s, r) = make_move(s, a) #make a move from s using a, get 
                                 #the new state s and the reward r
    if game_over():
       sar_list.add(s, none, r)
       break
    else:
       # select random action epsilon time and policy[s] (greedy)
       # (1-epsilon) time
       let a = epsilon_random_action(policy[s])
       sar_list.add(s, a, r)
  End Loop
  # compute the return
  let G = 0
  let sag_list = [] # init empty list of state, action, return
  for s, a, r in sar_list.reverse(): #loop from the end
      G = r + gamma * G
      sag_list.add(s, a, G)
  #return the computed list
  return sag_list.reverse()
```

<h2>Generalized Policy Iteration with Monte Carlo Methods</h2>

<p>Generalized policy iteration basically means that we get our agent to interact with the environment over and over again, altering (or <b>iterating on</b>) our policy slightly each tome. We say that a policy is improved if the next policy has equal values for all states, except one that now has greater value. This results in an overall increase of the state-value function. We will modify an existing policy by changing one action at a time. The action we change will be whichever of all the chagnes that has been seen to make the largest change to the current policies state value function. The method through which we evaluate the policy can change, depending on whether or not we are able to change the starting state of the agent.</p>

<h2>Here's the Policy Improvement Algorithm</h2>

```python
Improve_Policy_Without_Exploring_Start():
  # create a random policy starting. Random action for each state  
  let policy = select_random_action_for_each_state() 
  let Q = [][]
  let r = []
  Loop large number of times:
    # Evaluate the policy :
    let sag_list = Evaluate_Policy_Without_Exploring_Start(policy)
    let visited_states_actions = []
    For Each s, a, G in sag_list:
       if not visited_states_actions.contains(s, a): 
          r[s,a].append(G)
          Q[s][a] = average(r[s, a])
          visited_states_actions.add(s, a)
    End For
    
    # update policy
    For Each s in policy:
      # select action that produces the max Q value for state s 
      policy[s] = action_with_max(Q[s])
    End For
End Loop
``` 

<h1>Monte Carlo</h1>

Monte carlo




<h1>Temporal Difference Learning</h1>
The problem with DP is that it needs a complete model of the environment in order to find an optimal path. The advantage however is that it can be implemented recursively and therefore does not actually need to run through episode after episode in order to learn. On the other hand, MC methods are powerful becuase you do not need this entire model, however you do need to iteratively interact with the environemnt. Now **temporal**

<p>References</p>
<ul>
    <li>https://medium.com/@zsalloum/monte-carlo-in-reinforcement-learning-the-easy-way-564c53010511</li>
</ul>

<h1>Review</h1>

<h2>Dynamic Programming</h2>

- Dynamic programming has to do with the bellman equation made into an iterative algorithm that sweeps though all possible states, actions, and rewards.
- Very tedious to do manually. You might get asked to do this. There is a video outlining it.
- DP uses distribution models. These give you ALOT of extra info as compared to sample models. It is needed to use the bellman equation for updates because you need to know the probabilty of each transition as a weighting in the sum over everything.

<h2>Monte Carlo</h2>

- To evaluate a policy, just run it over and over again and collect all the returns you get after each episode. Finally average all the returns you got and make that your guess for the value of the state.
- It can only be used for episodic problems. 

<h2>Temporal difference learning</h2>

- temporal difference propogates the rewards backwards to states and actions that led to the eventual reward.
- It computes the approximate value function (evaulates the policy)
- temporal difference is and online update, it does not need to wait until the end of each episode the way that MC does. It compute the new value from the assumed value of the next one.

<h1>Panning / Learning</h1>

- **Planning** assumes that you have a model of the world to examine and then make decisions based on. Produces or imporvoces policies based on models. Dyna, Dyna Q+, etc.

- **Learning** is changing behaviour based on interactions with the world, and rewards / experiences that it gets.

- **Models** give the set of states and the transition dynamics between those states. They can be sample models or distribution models. Distribution models give the PROBABILITY that you will transition to a new state given a state-action pair. Sample models just spit out a possibility to you.

- **Simulated Experience** is the experience you get from planning, whereas real experiene is actually aquired from real experience with the world.
- The exploration / exploitation trade-off for model based learning needs to try to explore to make sure its model is righ , but at the same time it needs to explore to make sure that its model is correct.

