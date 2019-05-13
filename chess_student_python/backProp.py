import numpy as np

def backProp (eta,gamma,R,Q_next,Q,a_agent,x,out1,W1,W2,bias_W1,bias_W2):
    # calculate activations


#########################################################################
    # # print("testing")
    # # print((cost[a_agent] - Q[a_agent]) * np.heaviside(act2[a_agent], 1))
    # # print(dirac[a_agent])
    # # print(delta_wh.shape)
    # # print(delta_wo.shape)
    # # print(dirac[a_agent])
    # # print(dirac_k1.shape)
    # # print(np.dot(dirac , W2).shape)
    #
    # calculate activations
    act1 = np.dot(W1, x) + bias_W1
    act2 = np.dot(W2, out1) + bias_W2

    # cost function for q-learning
    # cost = 0.5 * (R + gamma *  np.argmax(Q_next)) ** 2
    # cost = 0.5 * (R + gamma *  Q_next - Q) ** 2
    # cost = R + gamma *  gamma * np.max(Q_next) - Q[a_agent]
    cost = R + gamma * np.max(Q_next)

    # print(cost.shape)

    # learning rules for ouput layer, only updating the one action that was taken
    # should dirac have - q[a_agent]
    if R == 0:
        dirac = (cost - Q[a_agent]) * np.heaviside(act2[a_agent], 1)
    else:
        dirac = (R - Q[a_agent]) * np.heaviside(act2[a_agent], 1)
    # print("dirac.shape")
    # print(dirac.shape)
    delta_wo = eta * dirac * out1
    delta_bo = eta * dirac
    # print("delta_wo.shape")
    # print(delta_wo.shape)
    # print(delta_bo.shape)

    #
    # # learning rules for the hidden layer
    dirac_k1 = dirac * W2[a_agent] * np.heaviside(act1, 1)
    # print("w2 shape")
    # print(W2[a_agent].shape)
    # print("dirac_k1.shape")
    # print(dirac_k1.shape)
    delta_wh = eta * np.outer(dirac_k1 , x)
    # print("delta_wh.shape")
    # print(delta_wh.shape)
    # print(dirac_k1.shape)
    # print(x.shape)
    delta_bh = eta * dirac_k1
    #
    # # update network weights
    W1 += delta_wh
    W2[a_agent] += delta_wo
    # print(W2[a_agent].shape)
    #
    # # update network biases
    bias_W1 += delta_bh
    bias_W2[a_agent] += delta_bo

    # print("backprop for reward {}".format(R))


    return W1, W2, bias_W1, bias_W2, cost
