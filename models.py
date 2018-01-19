from keras import(
    optimizers,
)
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
import keras.backend as K
from keras.layers import (
    Activation,
    add,
    AlphaDropout,
    BatchNormalization,
    concatenate,
    Conv2D,
    Dense,
    dot,
    Dropout,
    Embedding,
    Flatten,
    Input,
    Reshape,
)
from keras.models import (
    load_model,
    Model,
)
from keras.regularizers import l2
from keras.utils import Sequence
import numpy as np

def _DefaultDense(num_neurons, reg=1e-3, activation='selu'):
    return Dense(
        num_neurons,
        kernel_regularizer=l2(reg),
        kernel_initializer='lecun_normal',
        use_bias=True,
        activation=activation,
    )


def _dense_identity(num_layers, num_neurons, x):
    x_shortcut = Dense(20)(x)
    for _ in range(num_layers - 1):
        x = _DefaultDense(num_neurons)(x)
    x = _DefaultDense(num_neurons, activation=None)(x)
    x = add([x, x_shortcut])
    return Activation('selu')(x)


def simple_feed_forward(num_features, num_layers):
    features = Input(shape=(num_features,))
    x = _dense_identity(num_layers, 20, features)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=features, outputs=x)


class PseudoLabelingSequence(Sequence):
    def __init__(
        self,
        model,
        train_x, train_y,
        pseudo_x, initial_pseudo_y,
        batch_size,
        pseudo_labeling_proportion=0.33,
        relabel_after_num_epochs=1,
        ):
        self.model = model
        self.x, self.y = train_x.copy(), train_y.copy()
        self.pseudo_x = pseudo_x.copy()
        if initial_pseudo_y:
            self.pseudo_y = initial_pseudo_y
        else:
            self.pseudo_y = model.predict(self.pseudo_x, batch_size=batch_size)
        self.batch_size = batch_size
        self.pseudo_share = pseudo_labeling_proportion
        self.relabel_after_num_epochs = relabel_after_num_epochs
        self.epochs_till_relabeling = relabel_after_num_epochs

    def __len__(self):
        return int((1+self.pseudo_share) * len(self.x) / self.batch_size) + 1

    def __getitem__(self, idx):
        def bound(id_, share):
            return int(id_ * self.batch_size * share)
        real_x = self.x[
            bound(idx, 1 - self.pseudo_share):
            bound(idx + 1, 1 - self.pseudo_share)
        ]
        real_y = self.y[
            bound(idx, 1 - self.pseudo_share):
            bound(idx + 1, 1 - self.pseudo_share)
        ]
        pseudo_x = self.pseudo_x[
            bound(idx, self.pseudo_share):
            bound(idx + 1, self.pseudo_share)
        ]
        pseudo_y = self.pseudo_y[
            bound(idx, self.pseudo_share):
            bound(idx + 1, self.pseudo_share)
        ]
        shuffle_order= np.random.permutation(len(real_x) + len(pseudo_x))
        x = np.concatenate((real_x, pseudo_x))[shuffle_order]
        y = np.concatenate((real_y, pseudo_y))[shuffle_order]
        return x, y

    def on_epoch_end(self):
        shuffle_order = np.random.permutation(len(self.x))
        self.x = self.x[shuffle_order]
        self.y = self.y[shuffle_order]
        self.epochs_till_relabeling -= 1
        if epochs_till_relabeling < 1:
            self.epochs_till_relabeling = self.relabel_after_num_epochs
            np.random.shuffle(self.pseudo_x)
            self.pseudo_y = self.model.predict(
                self.pseudo_x,
                batch_size=self.batch_size
            )
        else:
            shuffle_order = np.random.permutation(len(self.pseudo_x))
            self.pseudo_x = self.pseudo_x[shuffle_order]
            self.pseudo_y = self.pseudo_y[shuffle_order]
