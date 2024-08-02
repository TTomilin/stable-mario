from utilities.parser import BaseParser

class ResumeParser(BaseParser):
    def set_args(self):                
        self.arg("--directory", type=str, default=None, help="Directory from which to load model. Must contain model zip and textfile with train command.")
        self.arg("--reset_timesteps", action='store_true', help="Resets timesteps in learning")