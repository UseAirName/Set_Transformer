import yaml


class Configuration:
    def __init__(self, path: str):
        config_file = open(path)
        config = yaml.load(config_file, Loader=yaml.Loader)
        network = config["Network_parameters"]
        encoder = network["Encoder"]
        self.encoder_layer = encoder["n_layers"]
        self.encoder_width = encoder["hidden_width"]
        self.latent_dimension = encoder["latent_dimension"]

        transformer = network["Transformer"]
        self.transformer_layer = transformer["n_layers"]
        self.transformer_sharing = transformer["weight_sharing"]
        self.transformer_norm = transformer["normalization"]

        feed_forward = transformer["Feed_forward"]
        self.feed_forward_layers = feed_forward["n_layers"]
        self.feed_forward_width = feed_forward["hidden_width"]

        mha = transformer["MultiHeadAttention"]
        self.attention_heads = mha["n_heads"]
        self.attention_width = mha["head_width"]
        self.attention_dropout = mha["dropout"]

        dataset = config["DataSet_parameters"]
        self.dataset_type = dataset["type"]
        self.dataset_size = dataset["size"]
        self.set_n_points = dataset["n_points"]
        self.set_n_feature = dataset["n_feature"]
