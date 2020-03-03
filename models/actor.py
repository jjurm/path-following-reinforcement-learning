from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten
from keras.models import Sequential


def build_actor(batch_size, nb_actions, env):
    """
    Docs
    """
    init = RandomNormal(mean=0.0, stddev=0.1, seed=None)
    actor = Sequential([
        Flatten(input_shape=(batch_size,) + env.observation_space.shape),
        Dense(8, activation='relu', kernel_initializer=init),
        Dense(8, activation='relu', kernel_initializer=init),
        Dense(nb_actions, activation='tanh', kernel_initializer=init),
    ])
    print(actor.summary())

    return actor
