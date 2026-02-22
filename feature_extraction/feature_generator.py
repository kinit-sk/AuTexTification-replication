class FeatureGenerator:
    """Base class for all feature extractors."""
    def __init__(self):
        pass

    def word_features(self, sentences):
        raise NotImplementedError("Subclasses must implement this method.")
