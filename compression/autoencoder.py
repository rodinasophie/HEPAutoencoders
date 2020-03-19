import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    ACTIVATION_SELECTOR = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "leakyRelu": nn.LeakyReLU(),
    }

    def __generate_layers(self):
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])

        n_layers = len(self.layers)

        for i in range(n_layers):
            self.encoder.append(
                nn.Linear(
                    self.n_features if i == 0 else self.layers[i - 1], self.layers[i]
                )
            )

            self.decoder.append(
                nn.Linear(
                    self.layers[n_layers - i - 1],
                    self.n_features
                    if i == n_layers - 1
                    else self.layers[n_layers - i - 2],
                )
            )

    def __generate_description(self):
        description = "-" + str(self.layers[-1]) + "-"
        for i in range(len(self.layers) - 2, -1, -1):
            description = (
                "-" + str(self.layers[i]) + description + str(self.layers[i]) + "-"
            )
        self.description = "in" + description + "out"

    def __init__(self, layers, activation, n_features):
        super(Autoencoder, self).__init__()
        self.n_features = n_features
        self.layers = layers
        self.description = ""
        self.activation = Autoencoder.ACTIVATION_SELECTOR.get(activation, None)

        self.__generate_layers()
        self.__generate_description()

    def forward(self, x):
        enc_len = len(self.encoder)
        for i in range(enc_len):
            x = self.activation(self.encoder[i](x))

        for i in range(enc_len):
            x = self.decoder[i](x)
            if i != enc_len - 1:
                x = self.activation(x)
        return x

    def describe(self):
        if self.description == "":
            self.__generate_description()
        return self.description


class AE_3D_200(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_3D_200, self).__init__(
            layers=[200, 100, 50, 3], n_features=n_features, activation="tanh"
        )


class AE_3D_100(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_3D_100, self).__init__(
            layers=[100, 50, 3], n_features=n_features, activation="tanh"
        )


class AE_3D_small(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_3D_small, self).__init__(
            layers=[3], n_features=n_features, activation="tanh"
        )


class AE_3D_small_v2(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_3D_small_v2, self).__init__(
            layers=[8, 3], n_features=n_features, activation="tanh"
        )


class AE_big(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_big, self).__init__(
            layers=[8, 6, 4, 3], n_features=n_features, activation="tanh"
        )


class AE_3D_50(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_3D_50, self).__init__(
            layers=[50, 50, 20, 3], n_features=n_features, activation="tanh"
        )


class AE_big_2D_v1(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_big_2D_v1, self).__init__(
            layers=[8, 6, 4, 2], n_features=n_features, activation="tanh"
        )


class AE_big_2D_v2(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_big_2D_v2, self).__init__(
            layers=[8, 6, 4, 3, 2], n_features=n_features, activation="tanh"
        )


class AE_2D(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_2D, self).__init__(
            layers=[20, 10, 6, 2], n_features=n_features, activation="tanh"
        )


class AE_2D_v2(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_2D_v2, self).__init__(
            layers=[50, 20, 10, 2], n_features=n_features, activation="tanh"
        )


class AE_big_2D_v3(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_big_2D_v3, self).__init__(
            layers=[8, 6, 2], n_features=n_features, activation="tanh"
        )


class AE_2D_v3(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_2D_v3, self).__init__(
            layers=[100, 200, 100, 2], n_features=n_features, activation="tanh"
        )


class AE_2D_v4(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_2D_v4, self).__init__(
            layers=[500, 200, 100, 2], n_features=n_features, activation="tanh"
        )


class AE_2D_v5(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_2D_v5, self).__init__(
            layers=[200, 100, 50, 2], n_features=n_features, activation="tanh"
        )


class AE_2D_v100(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_2D_v100, self).__init__(
            layers=[100, 100, 100, 2], n_features=n_features, activation="tanh"
        )


class AE_2D_v50(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_2D_v50, self).__init__(
            layers=[50, 50, 50, 2], n_features=n_features, activation="tanh"
        )


class AE_2D_v1000(Autoencoder):
    def __init__(self, n_features=4):
        super(AE_2D_v1000, self).__init__(
            layers=[1000, 400, 100, 2], n_features=n_features, activation="tanh"
        )


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss

