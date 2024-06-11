from .engine import EngineLM

class SingletonBackwardEngine:
    """
    A singleton class representing the backward engine.

    This class ensures that only one instance of the backward engine is created and provides methods to set and get the engine."""

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(SingletonBackwardEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'engine'):
            self.engine: EngineLM = None
    
    def set_engine(self, engine: EngineLM, override: bool = False):
        """
        Sets the backward engine.

        :param engine: The backward engine to set.
        :type engine: EngineLM
        :param override: Whether to override the existing engine if it is already set. Defaults to False.
        :type override: bool
        :raises Exception: If the engine is already set and override is False.
        :return: None
        """
        if ((self.engine is not None) and (not override)):
            raise Exception("Engine already set. Use override=True to override cautiously.")
        self.engine = engine

    def get_engine(self):
        """
        Returns the backward engine.

        :return: The backward engine.
        :rtype: EngineLM
        """
        return self.engine

def set_backward_engine(engine: EngineLM, override: bool = False):
    singleton_backward_engine = SingletonBackwardEngine()
    singleton_backward_engine.set_engine(engine, override=override)
