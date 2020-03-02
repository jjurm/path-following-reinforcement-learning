from keras.models import Sequential
from keras.layers import Dense, Flatten


def build_actor(batch_size, nb_actions, env):
    """
    Docs
    """
    actor = Sequential([
        Flatten(input_shape=(batch_size,) + env.observation_space.shape),
        Dense(8, activation='relu'),
        Dense(nb_actions, activation='linear'),
    ])
    print(actor.summary())

    return actor
