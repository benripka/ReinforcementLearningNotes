
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

- **Model**: In RL there is always some form of environment. A model is used to encapsulate the transition dynamics of the environment, that is, it returns the next state that the agent will likely end up in if it takes an action in a given state. The policy defines which action SHOULD be taken in each state, the model can help show what new state will result from that acion in that state. It gives the TRANSITIONS. It is important to realize that the model IS NOT always the same as the environement. This is only true for simple or virtual problems. In real world applications you may still use a model to help learn, but this will likely be some idealization of reality. Learning from a model is called  **planning** and there are a whole host of planning algorithms. RL algorithms can be model-free or model-based. Some RL algorithms do not start with one but are actually able to devise and environement model on the fly and add a planning step on aswell! See Dyna-Q for an example.

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