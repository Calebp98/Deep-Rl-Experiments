import numpy as np
import numpy.matlib
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from features import *
from generate_game import *
from Q_values import *
from backProp import *

from mvAvg import *

size_board = 4


def main():
    """
    Generate a new game
    The function below generates a new chess board with King, Queen and Enemy King pieces randomly assigned so that they
    do not cause any threats to each other.
    s: a size_board x size_board matrix filled with zeros and three numbers:
    1 = location of the King
    2 = location of the Queen
    3 = location fo the Enemy King
    p_k2: 1x2 vector specifying the location of the Enemy King, the first number represents the row and the second
    number the colunm
    p_k1: same as p_k2 but for the King
    p_q1: same as p_k2 but for the Queen
    """
    s, p_k2, p_k1, p_q1 = generate_game(size_board)

    """
    Possible actions for the Queen are the eight directions (down, up, right, left, up-right, down-left, up-left,
    down-right) multiplied by the number of squares that the Queen can cover in one movement which equals the size of
    the board - 1
    """
    possible_queen_a = (s.shape[0] - 1) * 8
    """
    Possible actions for the King are the eight directions (down, up, right, left, up-right, down-left, up-left,
    down-right)
    """
    possible_king_a = 8

    # Total number of actions for Player 1 = actions of King + actions of Queen
    N_a = possible_king_a + possible_queen_a

    """
    Possible actions of the King
    This functions returns the locations in the chessboard that the King can go
    dfK1: a size_board x size_board matrix filled with 0 and 1.
          1 = locations that the king can move to
    a_k1: a 8x1 vector specifying the allowed actions for the King (marked with 1):
          down, up, right, left, down-right, down-left, up-right, up-left
    """
    dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Queen
    Same as the above function but for the Queen. Here we have 8*(size_board-1) possible actions as explained above
    """
    dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Enemy King
    Same as the above function but for the Enemy King. Here we have 8 possible actions as explained above
    """
    dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

    """
    Compute the features
    x is a Nx1 vector computing a number of input features based on which the network should adapt its weights
    with board size of 4x4 this N=50
    """
    x = features(p_q1, p_k1, p_k2, dfK2, s, check)

    """
    Initialization
    Define the size of the layers and initialization
    FILL THE CODE
    Define the network, the number of the nodes of the hidden layer should be 200, you should know the rest. The weights
    should be initialised according to a uniform distribution and rescaled by the total number of connections between
    the considered two layers. For instance, if you are initializing the weights between the input layer and the hidden
    layer each weight should be divided by (n_input_layer x n_hidden_layer), where n_input_layer and n_hidden_layer
    refer to the number of nodes in the input layer and the number of nodes in the hidden layer respectively. The biases
     should be initialized with zeros.
    """
    """
    neurons in input layer number of possible states, calculated by unfolding chess board * number of pieces + number of pieces -1 (for all game endings)
    neurons in output layer = number of possible actions player 1 can take
    """

    n_input_layer = (3 * size_board ** 2 + 2)  # Number of neurons of the input layer. TODO: Change this value
    n_hidden_layer = 200  # Number of neurons of the hidden layer
    n_output_layer = N_a  # Number of neurons of the output layer. TODO: Change this value accordingly

    """
    TODO: Define the w weights between the input and the hidden layer and the w weights between the hidden layer and the
    output layer according to the instructions. Define also the biases.
    """

    W1=np.random.uniform(0,1,(n_hidden_layer,n_input_layer))
    # print(W1.shape)

    W1=np.divide(W1,np.matlib.repmat(np.sum(W1,1)[:,None],1,n_input_layer))
    # print(W1.shape)

    W2=np.random.uniform(0,1,(n_output_layer,n_hidden_layer))
    # print(W2.shape)
    # print(np.matlib.repmat(np.sum(W2,1)[:,None],1,n_input_layer).shape)

    W2=np.divide(W2,np.matlib.repmat(np.sum(W2,1)[:,None],1,n_hidden_layer))

    bias_W1 = np.zeros((n_hidden_layer,))
    bias_W2 = np.zeros((n_output_layer,))

# tracking
    reward_history = np.array([])
    error_hist = np.array([])
    k1_hist = np.array([])
    q_hist = np.array([])
    move_hist = np.array([])
    delta_w_hist = np.array([])
    delta_b_hist = np.array([])
    weight_hist_w1 = np.array([])
    weight_hist_w2 = np.array([])

    # YOUR CODES ENDS HERE

    # Network Parameters
    epsilon_0 = 0.2   #epsilon for the e-greedy policy
    beta = 0.00005    #epsilon discount factor
    #     beta = 0.00005    #epsilon discount factor
    gamma = 0.85      #SARSA Learning discount factor
    #     gamma = 0.85      #SARSA Learning discount factor
    eta = 0.0035      #learning rate #should maybe be negative?
    N_episodes = 20000#Number of games, each game ends when we have a checkmate or a draw
# 10000
    ###  Training Loop  ###

    # Directions: down, up, right, left, down-right, down-left, up-right, up-left
    # Each row specifies a direction,
    # e.g. for down we need to add +1 to the current row and +0 to current column
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])

    # THE FOLLOWING VARIABLES COULD CONTAIN THE REWARDS PER EPISODE AND THE
    # NUMBER OF MOVES PER EPISODE, FILL THEM IN THE CODE ABOVE FOR THE
    # LEARNING. OTHER WAYS TO DO THIS ARE POSSIBLE, THIS IS A SUGGESTION ONLY.

    R_save = np.zeros([N_episodes, 1])
    N_moves_save = np.zeros([N_episodes, 1])

    # END OF SUGGESTIONS


    for n in tqdm(range(N_episodes),desc = "Training Agent",smoothing = 0.2):
        # if n%100 == 0: #print episode count every 100 episodes
        #     print("Episode: {}/{}".format(n,N_episodes))

        k1_moves = 0
        q_moves = 0
        k2_moves = 0
        # print(reward_history.size)
        # print(reward_history)
        game_reward = 0
        game_cost = 0
        avg_weight_w1 = 0
        avg_weight_w2 = 0


        epsilon_f = epsilon_0 / (1 + beta * n) #epsilon is discounting per iteration to have less probability to explore
        checkmate = 0  # 0 = not a checkmate, 1 = checkmate
        draw = 0  # 0 = not a draw, 1 = draw
        i = 1  # counter for movements

        # Generate a new game
        s, p_k2, p_k1, p_q1 = generate_game(size_board)

        # Possible actions of the King
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

        while checkmate == 0 and draw == 0:
            R = 0  # Reward

            # Player 1

            # Actions & allowed_actions
            a = np.concatenate([np.array(a_q1), np.array(a_k1)])
            allowed_a = np.where(a > 0)[0]

            # Computing Features
            x = features(p_q1, p_k1, p_k2, dfK2, s, check)

            # FILL THE CODE
            # Enter inside the Q_values function and fill it with your code.
            # You need to compute the Q values as output of your neural
            # network. You can change the input of the function by adding other
            # data, but the input of the function is suggested.

            Q, out1 = Q_values(x, W1, W2, bias_W1, bias_W2)
            #Q is the ouptut of the neural network: the Q values
            # out1 contains the activation of the nodes of the first layer
            # print(Q)

            """
            YOUR CODE STARTS HERE

            FILL THE CODE
            Implement epsilon greedy policy by using the vector a and a_allowed vector: be careful that the action must
            be chosen from the a_allowed vector. The index of this action must be remapped to the index of the vector a,
            containing all the possible actions. Create a vector called a_agent that contains the index of the action
            chosen. For instance, if a_allowed = [8, 16, 32] and you select the third action, a_agent=32 not 3.
            """

            # a_agent =  1# CHANGE THIS VALUE BASED ON YOUR CODE TO USE EPSILON GREEDY POLICY
            prob = np.random.rand()
            # prob = 1 #for q learning

            Q_a = Q[allowed_a] #the q values of allowed actions
            choice = 0

            if prob>epsilon_f:
                a_agent = allowed_a[np.argmax(Q_a)]
            else:
                # choose random value
                ranChoice = np.random.randint(0,allowed_a.size)
                a_agent = allowed_a[ranChoice]

            # print("Agent prob {},epsilon_f at {}, chose {},greedy = {} ".format(prob,epsilon_f,a_agent,prob>epsilon_f))

            #THE CODE ENDS HERE.


            # Player 1 makes the action
            if a_agent < possible_queen_a: #i think this is if the queen moves
                direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
                steps = a_agent - direction * (size_board - 1) + 1

                s[p_q1[0], p_q1[1]] = 0
                mov = map[direction, :] * steps
                s[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
                p_q1[0] = p_q1[0] + mov[0]
                p_q1[1] = p_q1[1] + mov[1]
                q_moves += 1

            else:
                direction = a_agent - possible_queen_a
                steps = 1

                s[p_k1[0], p_k1[1]] = 0
                mov = map[direction, :] * steps
                s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
                p_k1[0] = p_k1[0] + mov[0]
                p_k1[1] = p_k1[1] + mov[1]
                k1_moves += 1

            # Compute the allowed actions for the new position

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

            # Player 2

            # Check for draw or checkmate
            if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
                # King 2 has no freedom and it is checked
                # Checkmate and collect reward
                checkmate = 1
                R = 1  # Reward for checkmate

                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for
                the action made. You computed previously Q values in the Q_values function. Be careful: this is the last
                iteration of the episode, the agent gave checkmate.
                """
                try:
                    W1,W2,bias_W1,bias_W2,cost = backProp(eta,gamma,R,Q_next,Q,a_agent,x,out1,W1,W2,bias_W1,bias_W2)

                except:
                    print("Episode: {}/{}".format(n,N_episodes))

                game_reward = R
                game_cost += cost
                avg_weight_w1 = np.mean(np.abs(W1))
                avg_weight_w2 = np.mean(np.abs(W2))

                # THE CODE ENDS HERE

                if checkmate:
                    break

            elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
                # King 2 has no freedom but it is not checked
                draw = 1
                R = 0.1

                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for
                the action made. You computed previously Q values in the Q_values function. Be careful: this is the last
                iteration of the episode, it is a draw.
                """
# just copied code over from checkmnate section for now, this may well be incorrect


                # try:
                W1,W2,bias_W1,bias_W2,cost = backProp (eta,gamma,R,Q_next,Q,a_agent,x,out1,W1,W2,bias_W1,bias_W2)

                # except:
                # print("Episode: {}/{}".format(n,N_episodes))



                # YOUR CODE ENDS HERE
                game_reward = R
                game_cost += cost
                avg_weight_w1 = np.mean(np.abs(W1))
                avg_weight_w2 = np.mean(np.abs(W2))


                if draw:
                    break

            else:
                # Move enemy King randomly to a safe location
                allowed_enemy_a = np.where(a_k2 > 0)[0]
                a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
                a_enemy = allowed_enemy_a[a_help]

                direction = a_enemy
                steps = 1

                s[p_k2[0], p_k2[1]] = 0
                mov = map[direction, :] * steps
                s[p_k2[0] + mov[0], p_k2[1] + mov[1]] = 3

                p_k2[0] = p_k2[0] + mov[0]
                p_k2[1] = p_k2[1] + mov[1]

            # Update the parameters

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
            # Compute features
            x_next = features(p_q1, p_k1, p_k2, dfK2, s, check)
            # Compute Q-values for the discounted factor
            Q_next, _ = Q_values(x_next, W1, W2, bias_W1, bias_W2)

            """
            FILL THE CODE
            Update the parameters of your network by applying backpropagation and Q-learning. You need to use the
            rectified linear function as activation function (see supplementary materials). Exploit the Q value for
            the action made. You computed previously Q values in the Q_values function. Be careful: this is not the last
            iteration of the episode, the match continues.
            """
    # just copied code over from checkmnate section for now, this may well be incorrect

# some questionable matrix operations going on here

            R = 0

            W1,W2,bias_W1,bias_W2,cost = backProp (eta,gamma,R,Q_next,Q,a_agent,x,out1,W1,W2,bias_W1,bias_W2)

            game_reward = R
            game_cost += cost
            avg_weight_w1 = np.mean(np.abs(W1))
            avg_weight_w2 = np.mean(np.abs(W2))

            # reward_history = np.append(reward_history,[R])
            # YOUR CODE ENDS HERE
            i += 1

        game_cost = np.sqrt(np.mean(game_cost**2))
        reward_history = np.append(reward_history,game_reward)

        error_hist = np.append(error_hist,game_cost)
        k1_hist = np.append(k1_hist,q_moves)
        q_hist = np.append(q_hist,k1_moves)
        move_hist = np.append(move_hist,k1_moves+q_moves)

        weight_hist_w1 = np.append(weight_hist_w1,avg_weight_w1)
        weight_hist_w2 = np.append(weight_hist_w2,avg_weight_w2)

    # print("Weight values for w1 are {}".format(W1))
    # print("Weight values for w2 are {}".format(W2))






    # f, axarr = plt.subplots(1, sharex=True)
    # f.suptitle('debug plots, reward-king moves-queen moves-cost')
    # axarr[0].plot(moving_average(reward_history, n = 100))
    # axarr[1].plot(moving_average(k1_hist, n = 100))
    # axarr[2].plot(moving_average(q_hist, n = 100))
    # axarr[3].plot(moving_average(error_hist, n = 100))
    # axarr[4].plot(moving_average(delta_w_hist, n = 100))

    # better plots
    # f, axarr = plt.subplots(2, sharex=True)
    # plt.title('Plot of Reward and Number of Moves')
    # plt.plot(moving_average(reward_history, n = 100))
    # plt.plot(moving_average(move_hist, n = 100))


    fig, ax1 = plt.subplots()

    color=list(plt.rcParams['axes.prop_cycle'])[1]['color']
    ax1.set_xlabel('Game')
    ax1.set_ylabel('Reward', color=color)
    ax1.plot(exp_moving_average(reward_history, 1/1000),color=color,label='Reward')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color=list(plt.rcParams['axes.prop_cycle'])[0]['color']
    ax2.set_ylabel('Moves',color=color)  # we already handled the x-label with ax1
    ax2.plot(exp_moving_average(move_hist, 1/1000),color=color,label='Moves')
    ax2.tick_params(axis='y', labelcolor=color)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    fig, ax3 = plt.subplots()

    color=list(plt.rcParams['axes.prop_cycle'])[1]['color']
    ax3.set_xlabel('Game')
    ax3.set_ylabel('weight_hist_w1', color=color)
    ax3.plot(exp_moving_average(weight_hist_w1, 1/1000),color=color,label='W1 Weights')
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    color=list(plt.rcParams['axes.prop_cycle'])[0]['color']
    ax4.set_ylabel('weight_hist_w2',color=color)  # we already handled the x-label with ax1
    ax4.plot(exp_moving_average(weight_hist_w2, 1/1000),color=color,label='W2 Weights')
    ax4.tick_params(axis='y', labelcolor=color)

    erh = exp_moving_average(reward_history, 1/1000)
    emh = exp_moving_average(move_hist, 1/1000)
    ema = exp_moving_average(weight_hist_w1, 1/1000)
    ewa = exp_moving_average(weight_hist_w2, 1/1000)

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc=0)

    np.savetxt('test2.csv', (erh, emh, ema, ewa),delimiter=',')

    # outfile = "params_3"
    # np.save(outfile,erh )

    outfile = "Rq_1"
    np.save(outfile,erh )

    outfile = "Rm_1"
    np.save(outfile,emh )

    plt.show()



    # reward is being calculated incorrecrtly i think, should probably only append reward at end of game and total reward appended maybe??


if __name__ == '__main__':
    main()
