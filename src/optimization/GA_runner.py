# src/optimization/ga_runner.py
"""
Genetic Algorithm Runner to optimize hyperparameters for machine learning models.
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Tuple, Any
from tqdm import tqdm
from src.utils import Utils
from src.optimization.optimizer import Optimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GARunner:
    """
    A Genetic Algorithm Runner to optimize hyperparameters for machine learning models.

    Attributes:
        ga_params (dict): Configuration parameters for the Genetic Algorithm.
        all_possible_params (dict): Possible values for hyperparameters.
        optimizer (Optimizer): Instance of the Optimizer class.
    """

    def __init__(self, ga_params: Dict[str, Any] = None):
        """
        Initialize the GARunner with GA parameters.

        Args:
            ga_params (dict, optional): Configuration parameters for the Genetic Algorithm.
                If None, default parameters from Utils.CONFIG are used.
        """
        self.ga_params = ga_params if ga_params is not None else {
            "population_size": Utils.CONFIG.get('NBR_SOL', 5),
            "max_generations": Utils.CONFIG.get('NBR_GEN', 5),
            "retain": 0.7,
            "random_select": 0.1,
            "mutate_chance": 0.1
        }
        self._validate_ga_params()
        self.all_possible_params = None
        self.optimizer = None
        logger.info(f"Initialized GARunner with parameters: {self.ga_params}")

    def _validate_ga_params(self) -> None:
        """
        Validate the Genetic Algorithm parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        required_keys = ["population_size", "max_generations", "retain", "random_select", "mutate_chance"]
        for key in required_keys:
            if key not in self.ga_params:
                raise ValueError(f"Missing required GA parameter: {key}")

        if not isinstance(self.ga_params["population_size"], int) or self.ga_params["population_size"] <= 0:
            raise ValueError("population_size must be a positive integer")
        if not isinstance(self.ga_params["max_generations"], int) or self.ga_params["max_generations"] <= 0:
            raise ValueError("max_generations must be a positive integer")
        if not 0 < self.ga_params["retain"] <= 1:
            raise ValueError("retain must be between 0 and 1")
        if not 0 <= self.ga_params["random_select"] <= 1:
            raise ValueError("random_select must be between 0 and 1")
        if not 0 <= self.ga_params["mutate_chance"] <= 1:
            raise ValueError("mutate_chance must be between 0 and 1")

    def _train_solution(self, solution: Any, fn_train: Callable, params_fn: Dict, index: int) -> None:
        """
        Train a single solution.

        Args:
            solution: The solution to train.
            fn_train (Callable): Function to train the model.
            params_fn (Dict): Additional parameters for the training function.
            index (int): Index of the solution in the population.
        """
        try:
            solution.train_model(fn_train, params_fn)
            logger.info(f"Solution {index} trained successfully")
        except Exception as e:
            logger.error(f"Error training solution {index}: {str(e)}")
            raise

    def train_population(self, population: List[Any], fn_train: Callable, params_fn: Dict) -> None:
        """
        Train an entire population in parallel using a thread pool.

        Args:
            population (List[Any]): List of solutions to train.
            fn_train (Callable): Function to train the model.
            params_fn (Dict): Additional parameters for the training function.
        """
        if not population:
            logger.warning("Population is empty, skipping training")
            return

        with tqdm(total=len(population), desc="Training Population") as pbar:
            with ThreadPoolExecutor(max_workers=len(population)) as executor:
                futures = [
                    executor.submit(self._train_solution, sol, fn_train, params_fn, i)
                    for i, sol in enumerate(population, start=1)
                ]
                for future in futures:
                    future.result()  # Wait for each thread to complete
                    pbar.update(1)

    def get_average_score(self, population: List[Any]) -> float:
        """
        Compute the average fitness score for a population.

        Args:
            population (List[Any]): List of solutions.

        Returns:
            float: Average fitness score.

        Raises:
            ValueError: If population is empty.
        """
        if not population:
            raise ValueError("Cannot compute average score for an empty population")
        avg_score = sum(sol.score for sol in population) / len(population)
        logger.debug(f"Average score for population: {avg_score}")
        return avg_score

    def print_top_solutions(self, population: List[Any], top_n: int = 3) -> None:
        """
        Print the top N solutions based on their scores.

        Args:
            population (List[Any]): List of solutions.
            top_n (int): Number of top solutions to print.
        """
        if not population:
            logger.warning("Population is empty, nothing to print")
            return

        top_solutions = sorted(population, key=lambda x: x.score, reverse=True)[:min(top_n, len(population))]
        for i, sol in enumerate(top_solutions, 1):
            logger.info(f"Top {i} solution: Score = {sol.score}, Params = {sol.params}")

    def generate(self, all_possible_params: Dict[str, List], fn_train: Callable, params_fn: Dict) -> Tuple[Dict, Any, Dict]:
        """
        Run the Genetic Algorithm to optimize hyperparameters.

        Args:
            all_possible_params (Dict[str, List]): Possible values for hyperparameters.
            fn_train (Callable): Function to train the model.
            params_fn (Dict): Additional parameters for the training function.

        Returns:
            Tuple[Dict, Any, Dict]: Best hyperparameters, best model, and best training entry.

        Raises:
            ValueError: If all_possible_params is empty or invalid.
        """
        if not all_possible_params or not isinstance(all_possible_params, dict):
            raise ValueError("all_possible_params must be a non-empty dictionary")

        self.all_possible_params = all_possible_params
        self.optimizer = Optimizer(self.ga_params, self.all_possible_params)
        population = self.optimizer.create_population(self.ga_params['population_size'])
        logger.info(f"Starting GA with population size {self.ga_params['population_size']} and {self.ga_params['max_generations']} generations")

        for generation in range(self.ga_params['max_generations']):
            logger.info(f"******** Generation {generation + 1} ********")
            self.train_population(population, fn_train, params_fn)
            avg_score = self.get_average_score(population)
            logger.info(f"Generation average: {avg_score * 100:.2f}%")

            if generation != self.ga_params['max_generations'] - 1:
                logger.info("Evolving population...")
                evolved_pop = self.optimizer.evolve(population)
                if evolved_pop:
                    population = evolved_pop
                else:
                    logger.warning("Evolved population is empty, continuing with current population")
            else:
                population.sort(key=lambda x: x.score, reverse=True)

        self.print_top_solutions(population)
        best_solution = population[0]
        logger.info(f"Best solution found: Score = {best_solution.score}, Params = {best_solution.params}")
        return best_solution.params, best_solution.model, best_solution.entry

if __name__ == "__main__":
    # Example usage
    def dummy_train_function(params, train_set):
        """Dummy training function for testing."""
        return {
            'validation_loss': np.random.uniform(0, 1),
            'model': None,
            'entry': {
                'AUC': np.random.uniform(0.5, 1.0),
                'accuracy': np.random.uniform(0.5, 1.0),
                'precision': np.random.uniform(0.5, 1.0),
                'recall': np.random.uniform(0.5, 1.0),
                'F1': np.random.uniform(0.5, 1.0)
            }
        }

    all_possible_params = {
        'drop_proba': [0.1, 0.2, 0.3],
        'nb_filters': [16, 32, 64],
        'nb_epochs': [5, 10],
        'nb_batch': [32, 64],
        'nb_layers': [1, 2],
        'optimizer': ['adam', 'rmsprop'],
        'time_step': [10, 20]
    }

    ga_runner = GARunner()
    best_params, best_model, best_entry = ga_runner.generate(
        all_possible_params=all_possible_params,
        fn_train=dummy_train_function,
        params_fn={'train_set': None}
    )
    print(f"Best Parameters: {best_params}")
    print(f"Best Entry: {best_entry}")