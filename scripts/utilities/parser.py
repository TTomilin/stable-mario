import argparse

class BaseParser:
    def __init__(self, arg_source):
        self._arg_source = arg_source
        self._parser = argparse.ArgumentParser()

    def arg(self, *args, **kwargs):
        self._parser.add_argument(*args, **kwargs)

    def set_args(self):                
        return NotImplementedError
    
    def validate_args(self):                
        return NotImplementedError
    
    def get_args(self):
        self.set_args()

        return self._parser.parse_args(args=self._arg_source)