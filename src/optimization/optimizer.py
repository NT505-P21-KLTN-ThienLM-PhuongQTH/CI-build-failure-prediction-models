"""Class implementing Genetic Algorithm for Hyperparameter Tuning."""
import random
from functools import reduce
from operator import add
from src.optimization.solution import Solution

class Optimizer:
    def __init__(self, GA_params, all_possible_params):
        """
        Initialize the optimizer.

        Args:
            GA_params (dict): Genetic Algorithm parameters (retain, mutation rate, etc.).
            all_possible_params (dict): Dictionary containing all possible values for hyperparameters.
        """
        self.random_select = GA_params["random_select"]
        self.mutate_chance = GA_params["mutate_chance"]
        self.retain = GA_params["retain"]
        self.all_possible_params = all_possible_params

    def create_population(self, count):
        """Create a random initial population of solutions."""
        return [Solution(self.all_possible_params) for _ in range(count)]

    @staticmethod
    def fitness(solution):
        """Return the fitness score of a solution (F1-score)."""
        return solution.score

    def grade(self, pop):
        """Compute average fitness score for a population."""
        return sum(map(self.fitness, pop)) / len(pop)

    def crossover(self, mother, father):
        """
        Perform crossover between two parent solutions to generate children.

        Args:
            mother (Solution): Parent 1.
            father (Solution): Parent 2.

        Returns:
            list: Two child solutions.
        """
        children = []
        for _ in range(2):
            child_params = {key: random.choice([mother.params[key], father.params[key]]) for key in self.all_possible_params}
            child = Solution(self.all_possible_params)
            child.set_params(child_params)

            # Apply mutation
            if self.mutate_chance > random.random():
                child = self.mutate(child)

            children.append(child)
        return children

    def mutate(self, solution):
        """Apply mutation to a solution."""
        mutation_param = random.choice(list(self.all_possible_params.keys()))
        solution.params[mutation_param] = random.choice(self.all_possible_params[mutation_param])
        return solution

    def evolve(self, pop):
        """
        Evolve the current population using selection, crossover, and mutation.

        Args:
            pop (list): List of Solution objects.

        Returns:
            list: The next generation population.
        """
        # Sort by fitness
        graded = sorted(pop, key=self.fitness, reverse=True)
        retain_length = int(len(graded) * self.retain)
        parents = graded[:retain_length]

        # Randomly add other individuals for genetic diversity
        parents.extend(individual for individual in graded[retain_length:] if self.random_select > random.random())

        if len(parents) < 2:
            print(f"Warning: Only {len(parents)} parents available, adding random solutions")
            while len(parents) < 2:
                new_solution = Solution(self.all_possible_params)
                parents.append(new_solution)

        # Generate offspring
        desired_length = len(pop) - len(parents)
        children = []
        while len(children) < desired_length:
            male, female = random.sample(parents, 2)
            children.extend(self.crossover(male, female))

        parents.extend(children[:desired_length])  # Ensure population size remains constant
        return parents
