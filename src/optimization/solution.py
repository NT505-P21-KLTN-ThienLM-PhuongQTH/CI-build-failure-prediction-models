# src/optimization/solution.py
import random

class Solution:
    def __init__(self, all_possible_params):
        # Initialize the solution with random hyperparameters.
        self.all_possible_params = all_possible_params
        self.params = {}
        self.entry = {}
        self.score = 0.0
        self.model = None
        self.create_random()
        self.history = None

    def create_random(self):
        # Generate random hyperparameters from the provided ranges.
        self.params = {key: random.choice(values) for key, values in self.all_possible_params.items()}

    def set_params(self, params):
        # Set the hyperparameters for this solution.
        self.params = params

    def train_model(self, fn_train, params_fn, val_set, metric="F1", pretrained_model_path=None):
        if self.score == 0.0:
            res = fn_train(self.params, params_fn, val_set, pretrained_model_path=pretrained_model_path)
            self.model = res['model']
            self.entry = res['entry']
            self.score = self.entry.get(metric, 0.0)
            self.history = res.get("history")

    def print_solution(self):
        # Print the solution's hyperparameters and score.
        print(f"For params {self.params}, the score in training = {self.score:.4f} and entry = {self.entry}")
