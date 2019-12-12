#!/usr/bin/env python3
import abc


class BasePoolStrategy(abc.ABC):
    r"""Base class for pool strategy.
    .. warning::
    Arguments:
        data_pool (iterable): an iterable of :class:`np.ndarray` s.
            Specifies what Tensors should be selected.
        excluded_indexes (iterable): an iterable of uint.
            Specifies what indexes should be excluded.
        loss_function (tf.keras.losses): an object of :class:`tf.keras.losses`.
            Specifies what loss function should be used for sampling samples.
        n_classes (uint): a number of classes.
        n_samples (uint): a number of samples selected by the strategy.
        batch_size (uint): a size of mini-batch to compute score for sampling instances.
    """

    def __init__(self,
                 data_pool=None,
                 excluded_indexes=None,
                 loss_function=None,
                 n_classes=None,
                 n_samples=None,
                 batch_size=None):

        self.data_pool = data_pool
        self.excluded_indexes = excluded_indexes
        self.loss_function = loss_function
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = batch_size

    def __call__(self,
                 current_model,
                 loss_function=None,
                 n_classes=None,
                 data_pool=None,
                 excluded_indexes=None,
                 n_samples=10,
                 batch_size=32):
        pass

    def __getstate__(self):
        return {
            "data_pool": self.data_pool,
            "excluded_indexes": self.excluded_indexes,
            "loss_function": self.loss_function,
            "n_classes": self.n_classes,
            "n_samples": self.n_samples,
            "batch_size": self.batch_size,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return self.__getstate__()
