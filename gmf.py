from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, concatenate, multiply
from tensorflow.keras.models import Model
import pandas as pd


def model_mlp(nu, ni):

    input_user = Input(shape=(1,))
    input_items = Input(shape=(1,))

    embedding_user = Embedding(nu, 2)(input_user)
    embedding_items = Embedding(ni, 2)(input_items)

    user_latent = Flatten()(embedding_user)
    item_latent = Flatten()(embedding_items)

    vect = multiply([user_latent, item_latent])

    prediction = Dense(1, activation='linear', name = 'prediction')(vect)

    model = Model(inputs=[input_user, input_items],
                          outputs=prediction)

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


if __name__ == '__main__':
    r = pd.read_csv("/workspaces/Class_AI3_hos/data/ratings.csv")
    mod = model_mlp(611, 193610)
    mod.fit([r.userId, r.movieId], r.rating, validation_split=0.2, epochs=100)