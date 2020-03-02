from keras.models import Model
from keras.layers import Dense, Flatten, Input, Concatenate

def build_critic(action_input, observation_input):
    """
    Docs
    """
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)

    print(critic.summary())

    return critic
