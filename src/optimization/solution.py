"""Class that represents an individual solution in the Genetic Algorithm."""
import random

class Solution:
    def __init__(self, all_possible_params):
        """
        Initialize a Solution instance.

        Args:
            all_possible_params (dict): Dictionary containing possible parameter values.
        """
        self.entry = {}
        self.score = 0.0
        self.all_possible_params = all_possible_params
        self.params = {}  # Stores model hyperparameters
        self.model = None
        self.create_random()

    def create_random(self):
        """Randomly assign hyperparameters from possible values."""
        self.params = {key: random.choice(values) for key, values in self.all_possible_params.items()}

    def set_params(self, params):
        """Set specific hyperparameter values."""
        self.params = params

    def train_model(self, fn_train, params_fn):
        """
        Train the model and store the best score.

        Args:
            fn_train (function): Training function.
            params_fn (dict): Additional parameters for training.
        """
        if self.score == 0.0:
            res = fn_train(self.params, params_fn)
            self.score = res["entry"].get("F1", 0.0)  # Default to 0.0 if missing
            self.model = res["model"]
            self.entry = res['entry']

    def print_solution(self):
        """Print the parameters and corresponding score."""
        print(f"For params {self.params}, the score in training = {self.score:.4f}")
