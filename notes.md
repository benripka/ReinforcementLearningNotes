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


<h1>Glossary of Terms</h1>

- **Reinforcement Learning:** The field of AI concerned with deciding which actions a software agent ought to take in each state of an environment so as to maximize a defined cumulative reward signal over time. 

- **Exploration vs. Exploitation:** This is a key tradeoff in RL algorithms that shows up again and again. A given algorithm must alternate between trying to choose actions it knows from experience to be valuable, whilst still probing (occasionally taking) unknown or less known state-actions to monitor for potential new valueble actions. Exploration is done to avoid getting stuck in a suboptimal policy at any given time. Exploitation is the heart of why we are doing RL. Should you go with what you already know works, or look for something better? You will often hear the word **greedy**
 or **act greedily** used to describe an algorithm / agent that heavily or exclusively exploits the current state-action of highest value.

- **Greedy Behaviour:** an agent is said to act greedily if it chooses to exploit, or take the actions in each state that it believes will maximize the return from that state. 

- **Delayed Reward:** In RL, whether an action in a state is good or bad is not an immediate property of taking the action. Contrast this to supervised learning where you get instant rewards from the data: action = label as dog, reward = -100 because its a picture of a cat. Each action is independent from the others. The difference is, in RL, the value of a state action depends on the causal chain of future state-action pairs. Therefore an agent must seek to learn what the potential future rewards may be for each following state-action aswell as the immediate rewards. The concept of **return** is introduced to formalize this.
- **Policy**: An agents policy is used to map states to actions. The policy is a function that can either be stochastic or deterministic. A deterministic policy will be a function of the state, that tells the agent what action to take in each state. A stochastic policy is a function of state aswell, but returns the entire probability distribution over the potential actions in that state. In other words it tells you how likely it is that each action be taken (under this policy). Using a stochastic policy you can actually come up with a deterministic policy. For instance you could just use the stochastic policy to get the distribution over all actions and then choose the most likely action in each state. Since now only one action is given in each state you could call this entire selection process a deterministic policy.