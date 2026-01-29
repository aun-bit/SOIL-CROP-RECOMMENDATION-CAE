from tensorflow.keras import layers, models

def build_models(n_features: int, n_classes: int):
    """
    Build:
      - Convolutional Autoencoder (autoencoder)
      - Encoder (latent representation)
      - Classifier on top of encoder (cae_classifier)
    """
    input_shape = (n_features, 1)
    inputs = layers.Input(shape=input_shape, name="input")

    # ----- Encoder -----
    x = layers.Conv1D(16, kernel_size=3, activation="relu",
                      padding="same", name="enc_conv1")(inputs)
    encoded_feat = layers.Conv1D(
        8, kernel_size=3, strides=2, activation="relu",
        padding="same", name="enc_conv2"
    )(x)
    flat = layers.Flatten(name="encoded_flat")(encoded_feat)

    # ----- Decoder (for autoencoder) -----
    x_dec = layers.UpSampling1D(size=2, name="dec_upsample")(encoded_feat)
    x_dec = layers.Conv1D(16, kernel_size=3, activation="relu",
                          padding="same", name="dec_conv1")(x_dec)
    x_dec = layers.Conv1D(1, kernel_size=3, activation="linear",
                          padding="same", name="dec_conv2")(x_dec)

    # Ensure output length == n_features
    decoded = layers.Lambda(lambda t: t[:, :n_features, :],
                            name="decoded")(x_dec)

    autoencoder = models.Model(inputs, decoded, name="conv_autoencoder")
    encoder = models.Model(inputs, flat, name="encoder")

    # ----- Classifier on top of encoder -----
    cls_x = encoder(inputs)
    cls_x = layers.Dense(64, activation="relu", name="cls_dense1")(cls_x)
    outputs = layers.Dense(n_classes, activation="softmax",
                           name="cls_output")(cls_x)

    classifier = models.Model(inputs, outputs, name="cae_classifier")

    return autoencoder, encoder, classifier
