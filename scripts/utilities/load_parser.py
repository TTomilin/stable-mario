import argparse
from parser import BaseParser

class LoadParser(BaseParser):
    def set_args(self):                
        self.arg("--directory", type=str, default=None, help="Directory from which to load model. Must contain model zip and textfile with train command.")
        self.arg("--record", default=False, help="Whether to record the trained model.")
        self.arg("--record_every", type=int, default=1, help="Record trained model every n episodes")