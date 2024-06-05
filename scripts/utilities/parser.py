import argparse

class BaseParser:
    def __init__(self):
        self._parser = argparse.ArgumentParser()

    def arg(self, *args, **kwargs):
        self._parser.add_argument(*args, **kwargs)

    def get_args(self):                
        return NotImplementedError