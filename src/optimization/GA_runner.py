from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Tuple, Any
from helpers import Utils
from optimization.optimizer import Optimizer

class GARunner:
    def __init__(self, ga_params: Dict[str, Any] = None):
        self.ga_params = ga_params if ga_params is not None else {
            "population_size": Utils.CONFIG.get('NBR_SOL'),
            "max_generations": Utils.CONFIG.get('NBR_GEN'),
            "retain": 0.7,
            "random_select": 0.1,
            "mutate_chance": 0.1
        }
        self._validate_ga_params()
        self.all_possible_params = None
        self.optimizer = None

    def _validate_ga_params(self) -> None:
        # Validate the GA parameters to ensure they are within expected ranges and types.
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
        # Train a single solution and handle exceptions.
        try:
            solution.train_model(fn_train, params_fn)
            print(f"Solution {index} trained")
        except Exception as e:
            print(f"Error training solution {index}: {str(e)}")
            raise

    def train_population(self, population: List[Any], fn_train: Callable, params_fn: Dict) -> None:
        # Train the entire population using multithreading.
        if not population:
            print("Empty population, skipping training")
            return

        with tqdm(total=len(population), desc="\nTraining Population") as pbar:
            with ThreadPoolExecutor(max_workers=len(population)) as executor:
                futures = [
                    executor.submit(self._train_solution, sol, fn_train, params_fn, i)
                    for i, sol in enumerate(population, start=1)
                ]
                for future in futures:
                    future.result()  # Wait for each thread to complete
                    pbar.update(1)

    def get_average_score(self, population: List[Any]) -> float:
        # Calculate the average score of the population.
        if not population:
            raise ValueError("Cannot compute average score for an empty population")
        avg_score = sum(sol.score for sol in population) / len(population)
        print(f"Average score of the population: {avg_score:.4f}")
        return avg_score

    def print_top_solutions(self, population: List[Any], top_n: int = 3) -> None:
        # Print top solutions based on their scores.
        if not population:
            print("No solutions in population to display")
            return

        top_solutions = sorted(population, key=lambda x: x.score, reverse=True)[:min(top_n, len(population))]
        for i, sol in enumerate(top_solutions, 1):
            print(f"Top {i} solution: Score = {sol.score:.4f}, Params = {sol.params}, Entry = {sol.entry}")

    def generate(self, all_possible_params: Dict[str, List], fn_train: Callable, params_fn: Dict) -> Tuple[Dict, Any, Dict]:
        # Generate the best parameters using a genetic algorithm.
        if not all_possible_params or not isinstance(all_possible_params, dict):
            raise ValueError("all_possible_params must be a non-empty dictionary")

        self.all_possible_params = all_possible_params
        self.optimizer = Optimizer(self.ga_params, self.all_possible_params)

        population = self.optimizer.create_population(self.ga_params['population_size'])
        print(f"Starting GA with population size {self.ga_params['population_size']} and {self.ga_params['max_generations']} generations")

        for generation in range(self.ga_params['max_generations']):
            print(f"\n===== Generation {generation + 1} =====")
            self.train_population(population, fn_train, params_fn)

            avg_score = self.get_average_score(population)
            print(f"Generation Average Score: {avg_score:.4f}")

            if generation < self.ga_params["max_generations"] - 1:
                print("Evolving to next generation...")
                evolved_pop = self.optimizer.evolve(population)
                if evolved_pop:
                    population = evolved_pop
                else:
                    print("Evolved population is empty, continuing with current population")
            else:
                population.sort(key=lambda x: x.score, reverse=True)

        self.print_top_solutions(population)
        best_solution = population[0]
        print(f"\nBest Solution: Score = {best_solution.score:.4f}, Params = {best_solution.params}")
        return best_solution.params, best_solution.model, best_solution.entry