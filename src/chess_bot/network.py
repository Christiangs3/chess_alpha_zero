import tensorflow as tf
from tensorflow.keras import layers, models

def create_residual_block(input_tensor, filters):
  """"
  Crea los 'residual blocks' de la red
  """
  x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(input_tensor)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Add()([input_tensor, x])
  return x

def create_chess_network():
    input_layer = layers.Input(shape=(8, 8, 12))

    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(input_layer)
    x = layers.BatchNormalization()(x)

    # Residual blocks
    for _ in range(10): # Adding 10 residual layers
        x = create_residual_block(x, 256)

    x = layers.Flatten()(x)

    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # Policy output
    policy_head = layers.Dense(4672, activation="softmax", name="policy_output")(x)
    # Value output
    value_head = layers.Dense(1, activation="tanh", name="value_output")(x)

    model = models.Model(inputs=input_layer, outputs=[policy_head, value_head])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
      optimizer=optimizer,
      loss={'policy_output': 'categorical_crossentropy', 'value_output': 'mean_squared_error'}
    )

    return model