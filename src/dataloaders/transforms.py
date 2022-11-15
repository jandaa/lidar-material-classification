"""
Transformations to be applied to augment input data.
"""


class Compose(object):
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class Normalize(object):
    """Normalize data to have 0 mean and unit variance"""

    def __init__(self, mean, std):
        """Initialize any hyperparameters here."""
        self.mean = mean  # 469.47983
        self.std = std  # 6333.1953

    def __call__(self, x):
        """Compute transformtaion."""
        return (x - self.mean) / self.std


class Scale(object):
    """Scale all data points"""

    def __init__(self, scale):
        """Scale that all datapoints will be multiplied by."""
        self.scale = scale  # 1/32752

    def __call__(self, x):
        """Compute transformtaion."""
        return x * self.scale


class ExampleTransform(object):
    """Sample Transform"""

    def __init__(self):
        """Initialize any hyperparameters here."""

    def __call__(self, x):
        """Compute transformtaion."""
        return x
