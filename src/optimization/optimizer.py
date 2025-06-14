# src/optimization/optimizer.py
import random
from src.optimization.solution import Solution

class Optimizer:
    def __init__(self, GA_params, all_possible_params):
        self.random_select = GA_params["random_select"]
        self.mutate_chance = GA_params["mutate_chance"]
        self.retain = GA_params["retain"]
        self.all_possible_params = all_possible_params

    def create_population(self, count):
        return [Solution(self.all_possible_params) for _ in range(count)]

    @staticmethod
    def fitness(solution):
        return solution.score

    def grade(self, pop):
        return sum(map(self.fitness, pop)) / len(pop)

    def mutate(self, solution):
        mutation_param = random.choice(list(self.all_possible_params.keys()))
        solution.params[mutation_param] = random.choice(self.all_possible_params[mutation_param])
        return solution

    def crossover(self, mother, father):
        children = []
        for _ in range(2):
            child_params = {key: random.choice([mother.params[key], father.params[key]]) for key in self.all_possible_params}
            child = Solution(self.all_possible_params)
            child.set_params(child_params)

            if self.mutate_chance > random.random():
                child = self.mutate(child)

            children.append(child)
        return children

    def evolve(self, pop):
        graded = sorted(pop, key=self.fitness, reverse=True)
        retain_length = int(len(graded) * self.retain)
        parents = graded[:retain_length]

        parents.extend(individual for individual in graded[retain_length:] if self.random_select > random.random())

        if len(parents) < 2:
            print(f"Warning: Only {len(parents)} parents available, adding random solutions")
            while len(parents) < 2:
                new_solution = Solution(self.all_possible_params)
                parents.append(new_solution)

        desired_length = len(pop) - len(parents)
        children = []
        while len(children) < desired_length:
            male, female = random.sample(parents, 2)
            children.extend(self.crossover(male, female))

        parents.extend(children[:desired_length])
        return parents