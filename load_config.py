import yaml


class Configuration:
    def __init__(self, path: str):
        config_file = open(path)
        config = yaml.load(config_file, Loader=yaml.Loader)
        network = config["Network_parameters"]
        dataset = config["DataSet_parameters"]
        encoder = network["Encoder"]
        decoder = network["Decoder"]
        mha = network["MultiHeadAttention"]

        # List of layer for the encoder
        self.encoder_layer = encoder["layers"]
        # Dimensions of layers for the encoder
        self.encoder_dim = [[int(d) for d in dims] for dims in encoder["dimensions"]]
        # Dimension of the latent vector
        self.latent_dimension = encoder["latent_dimension"]

        # List of layer for the decoder
        self.decoder_layer = decoder["layers"]
        # Dimensions of layers for the decoder
        self.decoder_dim = [[int(d) for d in dims] for dims in decoder["dimensions"]]

        # Number of heads for the MultiHeadAttention layers
        self.n_head = mha["n_head"]
        # Width of heads for the MultiHeadAttention layers
        self.head_width = mha["head_width"]

        # Size of the data set
        self.dataset_size = dataset["size"]
        # Number of points per set
        self.set_n_points = dataset["n_points"]
        # Number of feature per point
        self.set_n_feature = dataset["n_feature"]
