import numpy as np

max_steps = 1000

def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    #OBJECTIVE : evaluate a given policy pi.
    V = np.zeros(env.nS)
    end_cond = 0
    while True:
        end_cond = 0
        for s in range(env.nS):
            V_add = 0

            for a in range(env.nA):
                for prob, new_state, reward in (env.MDP[s][a]):
                    V_add += policy[s][a]*(reward+gamma*(prob*V[new_state]))

            end_cond = max(end_cond, np.abs(V[s]- V_add))
            print("End_cond", end_cond, "V_dd", V_add)
            V[s]=V_add
        if end_cond < theta:
            return V


def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.nS, env.nA]) / env.nA
    policy_stable = True
    for s in range(env.nS):
        action = V
    return policy


#Gamma: discount factor, Theta : Standard for stopping iteration.
def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:

        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if policy == new_policy:
            break
        policy = new_policy
    return policy, V


"""
Value Iteration = evaluation + improvement 
Use Bellman Optimality Equation. iteratively
"""
def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)


    policy = policy_improvement(env, V, gamma)
    return policy, V
