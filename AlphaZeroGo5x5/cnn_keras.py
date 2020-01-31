import pickle
import numpy as np
from keras.engine.training import Model
from keras.engine.topology import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
from keras.utils import plot_model


class PolicyValueNet(object):
    """ AlphaGoZero-like Policy Value Net. """

    def __init__(self, size, saved_weights=None):
        """ Initialize Attributes. """
        self.size = size  # board edge size
        self.l2_const = 1e-4  # coef of l2 penalty
        self.build_network()  # build neural network
        if saved_weights:
            self.model.set_weights(pickle.load(open(saved_weights, 'rb')))

    def build_network(self):
        """ Build the Policy Value Neural Net using Keras. """
        inputs = Input(shape=(4, self.size, self.size))

        # 3 common conv layers
        c_conv1 = Conv2D(
            filters=32, kernel_size=(3, 3), padding="same",
            data_format="channels_first", activation="relu",
            kernel_regularizer=l2(self.l2_const))(inputs)
        c_conv2 = Conv2D(
            filters=64, kernel_size=(3, 3), padding="same",
            data_format="channels_first", activation="relu",
            kernel_regularizer=l2(self.l2_const))(c_conv1)
        c_conv3 = Conv2D(
            filters=128, kernel_size=(3, 3), padding="same",
            data_format="channels_first", activation="relu",
            kernel_regularizer=l2(self.l2_const))(c_conv2)

        # policy head
        p_conv = Conv2D(
            filters=4, kernel_size=(1, 1), data_format="channels_first",
            activation="relu", kernel_regularizer=l2(self.l2_const))(c_conv3)
        p_flat = Flatten()(p_conv)
        self.policy_net = Dense(
            self.size*self.size, activation="softmax",
            kernel_regularizer=l2(self.l2_const))(p_flat)

        # value head
        v_conv = Conv2D(
            filters=2, kernel_size=(1, 1), data_format="channels_first",
            activation="relu", kernel_regularizer=l2(self.l2_const))(c_conv3)
        v_flat = Flatten()(v_conv)
        v_dense = Dense(64, kernel_regularizer=l2(self.l2_const))(v_flat)
        self.value_net = Dense(
            1, activation="tanh",
            kernel_regularizer=l2(self.l2_const))(v_dense)

        # connect and build the model
        self.model = Model(inputs, [self.policy_net, self.value_net])
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=Adam(), loss=losses)

    def get_state(self, go):
        """ Convert the go board data to a state of 4 boards.
            The 4 boards are: the agent's pieces, the opponent's pieces,
            difference from previous board, move first or not.
        Params: go: a GO object.
        Returns: a (4, 5, 5) numpy array.
        """
        piece_type = 1 if go.X_move else 2
        cur_board = np.array(go.board)
        state = np.zeros((4, self.size, self.size))
        if go.previous_board:
            pre_board = np.array(go.previous_board)
            state[0] = (cur_board == piece_type).astype(float)
            state[1] = (cur_board == 3 - piece_type).astype(float)
            state[2] = (cur_board != pre_board).astype(float)
        if piece_type == 1:
            state[3][:, :] = 1.0
        return state[:, ::-1, :]

    def policy(self, go):
        """ Policy function for current go board.
        Params: go: a go object.
        Returns: (move, prob) tuples and corresponding values.
        """
        piece_type = 1 if go.X_move else 2
        candidates = []
        for i in range(go.size**2):
            row, col = i // go.size, i % go.size
            if go.valid_place_check(row, col, piece_type):
                candidates.append(i)
        cur_state = self.get_state(go)
        # expand dimension to predict
        move_probs, value = self.model.predict_on_batch(np.array(
            cur_state.reshape(-1, 4, self.size, self.size)))
        move_probs = zip(candidates, move_probs.flatten()[candidates])
        return move_probs, value[0][0]

    def get_entropy(self, probs):
        """ Return entropy according to move probabilities. """
        return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))

    def train_core(self, states, mcts_probs, winners, lr):
        """ Training core function, performs one step of training.
        Params:
            states: list or numpy array, training data.
            mcts_probs: list or numpy array, training labels.
            winners: list or numpy array, training labels.
            lr: float, learning rate.
        Returns: tuple of floats, loss and entropy
        """
        states = np.array(states)
        mcts_probs = np.array(mcts_probs)
        winners = np.array(winners)
        loss = self.model.evaluate(
            states, [mcts_probs, winners],
            batch_size=states.shape[0], verbose=0)
        move_probs, _ = self.model.predict_on_batch(states)
        entropy = self.get_entropy(move_probs)
        K.set_value(self.model.optimizer.lr, lr)
        self.model.fit(
            states, [mcts_probs, winners],
            batch_size=states.shape[0], verbose=0)
        return loss[0], entropy

    def get_weights(self):
        """ Return model weights. """
        return self.model.get_weights()

    def save_weights(self, data_path='best_model.model'):
        """ Save model weights. """
        pickle.dump(self.get_weights(), open(data_path, 'wb'), protocol=2)


if __name__ == '__main__':
    pv_net = PolicyValueNet(5)
    plot_model(pv_net.model, to_file='model.png')
