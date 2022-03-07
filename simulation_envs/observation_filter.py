from ray.rllib.utils.filter import MeanStdFilter

class MeanStdFilterSingleton:
    
    instance = None
    
    @classmethod
    def get_instance(cls, shape=None):
        if cls.instance is None:
            cls.instance = MeanStdFilter(shape)
            return cls.instance
        return cls.instance