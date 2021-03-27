import numpy as np

max_steps = 1000
#print( env.MDP[0][1] )
# state 0 에서 action 1 을 선택했을 때 [상태 이동 확률, 도착 state, reward]


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):

    V = np.zeros(env.nS)
    while True:
        deriv = 0
        for s in range(env.nS):
            new_V = 0
            for a in range(env.nA):
                for prob, new_state, reward in (env.MDP[s][a]):
                    new_V += policy[s][a]*(reward+gamma*(prob*V[new_state]))
            deriv = max(deriv, np.abs(V[s]- new_V))
            V[s]=new_V
        if deriv < theta:
            return V




def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    for s in range(env.nS):
        new_action = action_by_value(env, V, s, gamma)
        a_list = np.argwhere(new_action == np.max(new_action)).flatten()
        for a in a_list:
            policy[s][a] = 1/len(a_list)
        for i in range(env.nA):
            if i not in a_list:
                policy[s][i]=0
        if sum(policy[s])!=1:
            raise ValueError

    return policy


#Gamma: discount factor, Theta : Standard for stopping iteration.
def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    policy_list = []
    maximum_loop = 1000
    loop_cnt = 0
    while True:

        V = policy_evaluation(env, policy, gamma, theta)
        new_policy= policy_improvement(env, V, gamma)
        if (policy == new_policy).all():
            break
        if (loop_cnt == maximum_loop):
            break
        policy_list.append(policy)
        policy = new_policy
        loop_cnt+=1
    return policy, V


"""
Value Iteration = evaluation + improvement 
Use Bellman Optimality Equation. iteratively
"""
def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    for i in range(1000):
        deriv = 0
        for s in range(env.nS):
            new_V = np.max(action_by_value(env, V, s, gamma))
            deriv = max(deriv, np.abs(V[s] - new_V))
            V[s] = new_V
        print(V)
        if deriv < theta:
            break

    policy = policy_improvement(env, V, gamma)
    return policy, V


def action_by_value(env, V, state, gamma):
    action = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, new_state, reward in env.MDP[state][a]:

            action[a] += prob*(reward+gamma*V[new_state])

    return action
