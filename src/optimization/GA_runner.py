"""Genetic Algorithm Runner to optimize LSTM hyperparameters."""
import threading
from tqdm import tqdm
from ..utils import Utils as Utils
from .optimizer import Optimizer
# import src.utils.Utils as Utils
# from src.optimization.optimizer import Optimizer

def train_sol_thread(solution, fn_train, params_fn, i):
    """Train a solution in a separate thread."""
    solution.train_model(fn_train, params_fn)
    print(f"Solution {i} trained")

def train_population(pop, fn_train, params_fn):
    """Train an entire population in parallel using threading."""
    pbar = tqdm(total=len(pop), desc="Training Population")
    threads = [threading.Thread(target=train_sol_thread, args=(sol, fn_train, params_fn, i))
               for i, sol in enumerate(pop, start=1)]

    for thread in threads:
        thread.start()
        pbar.update(1)

    for thread in threads:
        thread.join()
    pbar.close()

def get_average_score(pop):
    """Compute the average fitness score for a population."""
    return sum(sol.score for sol in pop) / len(pop)

def generate(all_possible_params, fn_train, params_fn):
    """
    Run the Genetic Algorithm to optimize hyperparameters.

    Args:
        all_possible_params (dict): Possible values for hyperparameters.
        fn_train (function): Function to train the model.
        params_fn (dict): Additional parameters.

    Returns:
        tuple: Best hyperparameters, best model, best training entry.
    """
    GA_params = {
        "population_size": Utils.CONFIG['NBR_SOL'],
        "max_generations": Utils.CONFIG['NBR_GEN'],
        "retain": 0.7,
        "random_select": 0.1,
        "mutate_chance": 0.1
    }

    print(f"GA Parameters: {GA_params}")
    optimizer = Optimizer(GA_params, all_possible_params)
    pop = optimizer.create_population(GA_params['population_size'])

    for i in range(GA_params['max_generations']):
        print(f"******** Generation {i+1} ********")
        train_population(pop, fn_train, params_fn) # Train the population
        avg_accuracy = get_average_score(pop) # Compute average fitness score
        print(f"Generation average: {avg_accuracy * 100:.2f}%") # Print average fitness score

        # Print top solutions
        if i != GA_params['max_generations'] - 1:
            print("Evolving population...")
            evolved_pop = optimizer.evolve(pop)
            if evolved_pop:
                pop = evolved_pop
        else:
            pop.sort(key=lambda x: x.score, reverse=True) # Sort population by fitness score

    # Print top solutions
    print_top_solutions(pop[:min(3, len(pop))])
    return pop[0].params, pop[0].model, pop[0].entry

def print_top_solutions(pop):
    """Print the top N solutions."""
    for sol in pop:
        sol.print_solution()
