import torch
import torch.nn as nn


# -------------------- Fully Connected Network --------------------
class FCN(nn.Module):
    def __init__(self, dim_layer, drop_out=0.2, use_residual=False):
        super(FCN, self).__init__()
        layers = []
        self.use_residual = use_residual

        for i in range(1, len(dim_layer)):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i]))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            if drop_out > 0:
                layers.append(nn.Dropout(drop_out))
            if self.use_residual and i < len(dim_layer) - 1:
                layers.append(nn.Identity())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            residual = x
            for layer in self.model:
                x = layer(x)
                if isinstance(layer, nn.Identity):
                    x += residual
            return x
        else:
            return self.model(x)


# -------------------- MLP with intermediate bottleneck --------------------
class MLP(nn.Module):
    def __init__(self, layer_dims, drop_out=0.5, use_residual=False):
        super(MLP, self).__init__()
        self.use_residual = use_residual
        layers = []

        for i in range(len(layer_dims) - 1):
            dim_in = layer_dims[i]
            dim_out = layer_dims[i + 1]
            dim_hidden = int(dim_in * 0.5)
            layers.append(nn.Linear(dim_in, dim_hidden))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(drop_out))
            if self.use_residual and i < len(layer_dims) - 2:
                layers.append(nn.Identity())

        layers.append(nn.Linear(dim_hidden, layer_dims[-1]))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            residual = x
            for layer in self.mlp:
                x = layer(x)
                if isinstance(layer, nn.Identity):
                    x += residual
            return x
        else:
            return self.mlp(x)


# -------------------- Encoder Module --------------------
class RecognitionModel(nn.Module):
    def __init__(self, recognition_model_dim, use_dropout, drop_rate, activation):
        super(RecognitionModel, self).__init__()

        recognition_model = []
        for i in range(len(recognition_model_dim) - 1):
            recognition_model.append(nn.Linear(recognition_model_dim[i], recognition_model_dim[i + 1]))
            if i < len(recognition_model_dim) - 2:
                recognition_model.append(nn.BatchNorm1d(recognition_model_dim[i + 1]))
                recognition_model.append(nn.ReLU())
                if use_dropout:
                    recognition_model.append(nn.Dropout(drop_rate))
            else:
                if activation == 'relu':
                    recognition_model.append(nn.ReLU())
                elif activation == 'tanh':
                    recognition_model.append(nn.Tanh())
                elif activation == 'none':
                    recognition_model.append(nn.Identity())

        self.recognition_model = nn.Sequential(*recognition_model)

    def forward(self, x):
        return self.recognition_model(x)


# -------------------- Decoder Module --------------------
class GenerativeModel(nn.Module):
    def __init__(self, generative_model_dim):
        super(GenerativeModel, self).__init__()

        generative_model = []
        for i in range(len(generative_model_dim) - 1):
            generative_model.append(nn.Linear(generative_model_dim[i], generative_model_dim[i + 1]))
            if i < len(generative_model_dim) - 2:
                generative_model.append(nn.ReLU())
            else:
                generative_model.append(nn.Sigmoid())

        self.generative_model = nn.Sequential(*generative_model)

    def forward(self, x):
        return self.generative_model(x)


# -------------------- Multi-View Autoencoder with Clustering --------------------
class MultiViewAutoencoderWithClustering(nn.Module):
    def __init__(self, n_views, recognition_model_dims, generative_model_dims, n_clusters, temperature, drop_rate=0.5, args=None):
        super(MultiViewAutoencoderWithClustering, self).__init__()
        self.n_views = n_views
        self.n_clusters = n_clusters
        self.temperature = temperature

        # Multi-view encoders
        self.online_encoder = nn.ModuleList([
            RecognitionModel(
                recognition_model_dims[i],
                use_dropout=True,
                drop_rate=drop_rate,
                activation='relu'
            ) for i in range(n_views)
        ])

        # Multi-view decoders
        self.decoder = nn.ModuleList([
            GenerativeModel(generative_model_dims[i]) for i in range(n_views)
        ])

        # Cluster centers for clustering loss
        concatenated_embedding_dim = sum([dim[-1] for dim in recognition_model_dims])
        self.cluster_centers = nn.Parameter(torch.randn(self.n_clusters, concatenated_embedding_dim))

    def forward(self, *args):
        # Split inputs by views
        view_data = [args[i] for i in range(self.n_views)]

        # Encode each view
        encoded_views = [self.online_encoder[i](view_data[i]) for i in range(self.n_views)]

        # Decode each view
        reconstructed_views = [self.decoder[i](encoded_views[i]) for i in range(self.n_views)]

        return encoded_views, reconstructed_views

    def extract_feature(self, samples):
        return [self.online_encoder[i](samples[i]) for i in range(self.n_views)]




















