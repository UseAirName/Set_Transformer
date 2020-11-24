import yaml


class NetworkCfg:
    def __init__(self, encoder_layer, encoder_dim, decoder_layer, decoder_dim):
        self.encoder_layer = encoder_layer
        self.encoder_dim = encoder_dim
        self.decoder_layer = decoder_layer
        self.decoder_dim = decoder_dim


class Configuration:
    def __init__(self, path: str):
        config_file = open(path)
        config = yaml.load(config_file, Loader=yaml.Loader)
        network = config["Network_parameters"]
        baseline = config["Baseline_parameters"]
        dataset = config["DataSet_parameters"]
        parameters = config["Run_parameters"]
        encoder = network["Encoder"]
        decoder = network["Decoder"]
        mha = network["MultiHeadAttention"]

        # List of layer for the encoder
        encoder_layer = encoder["layers"]
        # Dimensions of layers for the encoder
        encoder_dim = [[int(d) for d in dims] for dims in encoder["dimensions"]]
        # Dimension of the latent vector
        self.latent_dimension = encoder["latent_dimension"]

        # List of layer for the decoder
        decoder_layer = decoder["layers"]
        # Dimensions of layers for the decoder
        decoder_dim = [[int(d) for d in dims] for dims in decoder["dimensions"]]

        self.network = NetworkCfg(encoder_layer, encoder_dim, decoder_layer, decoder_dim)

        # Number of heads for the MultiHeadAttention layers
        self.n_head = mha["n_head"]
        # Width of heads for the MultiHeadAttention layers
        self.head_width = mha["head_width"]

        # List of layer for the encoder baseline
        b_encoder_layer = encoder["layers"]
        # Dimensions of layers for the encoder baseline
        b_encoder_dim = [[int(d) for d in dims] for dims in encoder["dimensions"]]

        # List of layer for the decoder baseline
        b_decoder_layer = decoder["layers"]
        # Dimensions of layers for the decoder baseline
        b_decoder_dim = [[int(d) for d in dims] for dims in decoder["dimensions"]]

        self.baseline = NetworkCfg(b_encoder_layer, b_encoder_dim, b_decoder_layer, b_decoder_dim)

        # Loss criterion
        self.criterion = network["Loss"]
        self.residuals = network["Residuals"]

        # Type of the data set
        self.data_type = dataset["type"]
        # Size of the data set
        self.dataset_size = dataset["size"]
        # Number of points per set
        self.set_n_points = dataset["n_points"]
        # Number of feature per point
        self.set_n_feature = dataset["n_feature"]

        # Activate weight and biases
        self.wandb_on = parameters["wandb"]
        # Run the baseline too
        self.baseline_on = parameters["baseline"]
