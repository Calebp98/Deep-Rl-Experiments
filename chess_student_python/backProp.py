import numpy as np

def backProp (eta,gamma,R,Q_next,Q,a_agent,x,out1,W1,W2,bias_W1,bias_W2):

    # calculate activations
    act1 = np.dot(W1, x) + bias_W1
    act2 = np.dot(W2, out1) + bias_W2

    # cost function for q-learning (commented out functions are experiments for other cost functions)
    # cost = 0.5 * (R + gamma *  np.argmax(Q_next)) ** 2
    # cost = 0.5 * (R + gamma *  Q_next - Q) ** 2
    # cost = R + gamma *  gamma * np.max(Q_next) - Q[a_agent]
    cost = R + gamma * np.max(Q_next)


    # learning rules for ouput layer, only updating the one action that was taken
    # should dirac have - q[a_agent]
    if R == 0:
        dirac = (cost - Q[a_agent]) * np.heaviside(act2[a_agent], 1)
    else:
        dirac = (R - Q[a_agent]) * np.heaviside(act2[a_agent], 1)

    delta_wo = eta * dirac * out1
    delta_bo = eta * dirac

    # # learning rules for the hidden layer
    dirac_k1 = dirac * W2[a_agent] * np.heaviside(act1, 1)

    delta_wh = eta * np.outer(dirac_k1 , x)

    delta_bh = eta * dirac_k1
    
    # update network weights
    W1 += delta_wh
    W2[a_agent] += delta_wo

    # update network biases
    bias_W1 += delta_bh
    bias_W2[a_agent] += delta_bo

    # print("backprop for reward {}".format(R))


    return W1, W2, bias_W1, bias_W2, cost
