""" This file is the main file for the Expert Agent called Deesp """


class Deesp:
    """
    Summary of class here.

    Long class information ...

    Attributes:
    """
    def __init__(self):
        """Instanciates a Dispatcher Deesp"""
        print("Deesp init executed ...")
        self.param_path = None

    def load(self, path: str):
        """Loads mandatory data from path
        :param path: absolute path
        :return:
        """
        self.param_path = path
        print("The path we have is, ", self.param_path)

