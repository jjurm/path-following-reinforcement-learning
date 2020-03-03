from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Concatenate
from keras.models import Model


def build_critic(action_input, observation_input):
    """
    Docs
    """
    init = RandomNormal(mean=0.0, stddev=0.2, seed=None)
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(8, activation='relu', kernel_initializer=init)(x)
    x = Dense(8, activation='relu', kernel_initializer=init)(x)
    x = Dense(1, activation='linear', kernel_initializer=init)(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    return critic
