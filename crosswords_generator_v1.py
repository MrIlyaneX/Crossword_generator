import os
import random
import functools

grid_size = 20

inputs_folder = "inputs_3"
outputs_folder = "outputs"

# Constants for the fitness function, in the function scaled to the powers
WORDS_COEXIST_COEFFICIENT = 10  # coefficient for the words that coexist in each other

WORDS_NOT_CROSSING_PENALTY_COEFFICIENT = 10  # coefficient for the words that do not cross each other
WORDS_INCORRECT_CROSSING_COEFFICIENT = 10  # coefficient for the words that cross each other incorrectly
SYMBOLS_NEIGHBOUR_PENALTY_COEFFICIENT = 10  # coefficient for the symbols that are neighbours

COMPONENT_COEFFICIENT = 10


@functools.lru_cache(maxsize=5000000)
def get_random_word_pos(word: str, g_size: int = 20):
    """
    Function that returns random position and orientation for given word

    :param word: word to place
    :param g_size: size of the grid (default 20)
    :return: x, y, orientation
    """
    orientation = random.choice(["h", "v"])
    x, y = 0, 0

    if orientation == 'h':
        x = random.randint(0, g_size - 1)
        y = random.randint(0, g_size - len(word))
    if orientation == 'v':
        x = random.randint(0, g_size - len(word))
        y = random.randint(0, g_size - 1)

    return x, y, orientation


class Crossword:
    """
    Class that represents crossword (gene).
    Stores words, their coordinates and orientation, fitness score
    """

    def __init__(self, words: list = None, grid_sz: int = 20, crossword=None):
        """
        Constructor of the Crossword class

        :param words: list of words to place in the crossword
        :param grid_sz: size of the grid (default 20)
        :param crossword: crossword copy c-tor
        """
        self.words = []
        self.grid_size = grid_sz
        self.index_storage = dict()
        self.fitness = None
        if words:
            for word in words:
                self.index_storage[word] = len(self.words)
                self.add_word(word)
        if crossword:
            self.words = crossword.words.copy()
            self.grid_size = crossword.grid_size
            self.index_storage = crossword.index_storage.copy()
            self.fitness = None

    def copy(self):
        """
        Makes a copy of the crossword
        :return:
        """
        return Crossword(crossword=self)

    def add_word(self, word: str) -> None:
        """
        Adds word to the crossword. Must be used only in the constructor

        :param word: word to add
        :return: None
        """
        x, y, orientation = get_random_word_pos(word=word)
        coords = set()
        dx, dy = x, y
        for char in word:
            coords.add((char, dx, dy))
            if orientation == "v":
                dx += 1
            if orientation == "h":
                dy += 1

        self.words.append((word, (x, y), coords, orientation))

    def change_word(self, word: str, new_coords: tuple, new_orientation: str):
        """

        :param word:
        :param new_coords:
        :param new_orientation:
        :return:
        """
        index = self.index_storage[word]
        dx, dy = new_coords
        coords = set()
        for char in word:
            coords.add((char, dx, dy))
            if new_orientation == "v":
                dx += 1
            if new_orientation == "h":
                dy += 1
        self.words[index] = (word, new_coords, coords, new_orientation)

    @functools.lru_cache(maxsize=5000000)
    def display(self) -> None:
        """
        Show the crossword in the console via grid, used in the testing

        :return:
        """
        grid = [['.' for _ in range(0, self.grid_size)] for _ in range(0, self.grid_size)]
        for _, (_, _), coords, _ in self.words:
            for char, x, y in coords:
                grid[x][y] = char

        for x in grid:
            for y in x:
                print(y, end=" ")
            print()

    def get_words(self) -> list:
        """
        Returns the list of words in the crossword

        :return: list of words
        """
        words = []
        for word, (x, y), _, orientation in self.words:
            orientation_digit = 0
            if orientation == "v":
                orientation_digit = 1
            if orientation == "h":
                orientation_digit = 0
            words.append((word, x, y, orientation_digit))
        return words


@functools.lru_cache(maxsize=5000000)
def get_fitness(individual: Crossword) -> int:
    """
    Function that evaluate given crossword
    The closer the fitness score to 0, the better. 0 score - crossword without issues

    :param individual: crossword to evaluate
    :return: fitness score of the crossword
    """
    if individual.fitness is not None:
        return individual.fitness

    wrong_cross_counter = 0
    coexists_counter = 0
    neighbourhood_counter = 0
    not_crossed_counter = 0
    component_counter = 0

    words = dict()
    centers = set()
    banned = set()

    for word, (x, y), coord, orientation in individual.words:
        crossed = False
        centers.add((x, y))
        if orientation == "v":
            if (x + len(word), y) in words:
                neighbourhood_counter += 1
            if (x - 1, y) in words:
                neighbourhood_counter += 1

        if orientation == "h":
            if (x, y + len(word)) in words:
                neighbourhood_counter += 1
            if (x, y - 1) in words:
                neighbourhood_counter += 1

        for char, dx, dy in coord:
            if not (0 <= dx < grid_size):
                neighbourhood_counter += 1000
            if not (0 <= dy < grid_size):
                neighbourhood_counter += 1000

            if (dx, dy) not in words:
                words[(dx, dy)] = char
            elif words[(dx, dy)] != char:
                wrong_cross_counter += 1
                banned.add((dx, dy))
                crossed = True
            else:
                crossed = True
                continue

            if orientation == "v":
                if (dx, dy - 1) in words:
                    neighbourhood_counter += 1
                if (dx, dy + 1) in words:
                    neighbourhood_counter += 1
            if orientation == "h":
                if (dx - 1, dy) in words:
                    neighbourhood_counter += 1
                if (dx + 1, dy) in words:
                    neighbourhood_counter += 1

        if not crossed:
            not_crossed_counter += 1

    visited = set()
    component_sizes = dict()

    @functools.lru_cache(maxsize=5000000)
    def dfs(dfs_x, dfs_y):
        if (dfs_x, dfs_y) not in words:
            return
        if (dfs_x, dfs_y) in visited:
            return

        visited.add((dfs_x, dfs_y))

        if (dfs_x, dfs_y) in banned:
            return

        if (dfs_x, dfs_y) in centers:
            component_sizes[component_counter] += 1

        dfs(dfs_x + 1, dfs_y)
        dfs(dfs_x, dfs_y + 1)
        dfs(dfs_x - 1, dfs_y)
        dfs(dfs_x, dfs_y - 1)

    for _, (x, y), _, _ in individual.words:
        if (x, y) not in visited:
            component_sizes[component_counter] = 0
            dfs(x, y)
            component_counter += 1

    component_penalty = 0

    if component_counter > 1:
        for c_size in range(0, component_counter - 1):
            component_penalty += COMPONENT_COEFFICIENT ** (len(centers) - component_sizes[c_size])

    if component_counter == 1 and wrong_cross_counter == 0 and neighbourhood_counter == 0:
        not_crossed_counter = 0

    crossing_penalty = not_crossed_counter * WORDS_NOT_CROSSING_PENALTY_COEFFICIENT ** 3
    coexists_penalty = coexists_counter * WORDS_COEXIST_COEFFICIENT ** (3 * len(centers))
    wrong_cross_penalty = wrong_cross_counter * WORDS_INCORRECT_CROSSING_COEFFICIENT ** 5
    neighbourhood_penalty = neighbourhood_counter * SYMBOLS_NEIGHBOUR_PENALTY_COEFFICIENT ** 4

    fitness = (
            component_penalty +
            wrong_cross_penalty +
            crossing_penalty +
            coexists_penalty +
            neighbourhood_penalty
    )

    individual.fitness = fitness
    return fitness


def crossover(m: Crossword, f: Crossword) -> Crossword:
    crossover_type = random.randint(1, 3)

    if crossover_type == 1:
        crossover_point = random.randint(1, len(m.words) - 1)
        offspring_words = f.words[:crossover_point] + m.words[crossover_point:]

    elif crossover_type == 2:

        crossover_points = sorted(random.sample(range(len(m.words)), 3))

        offspring_words = (
                f.words[:crossover_points[0]]
                + m.words[crossover_points[0]:crossover_points[1]]
                + f.words[crossover_points[1]:crossover_points[2]]
                + m.words[crossover_points[2]:]
        )

    else:
        offspring_words = [m_word if random.randint(0, 1) == 0
                           else f_word for m_word, f_word in zip(m.words, f.words)]

    offspring = Crossword()
    offspring.index_storage = m.index_storage.copy()
    offspring.words = offspring_words

    return offspring


def mutate(offspring: Crossword) -> Crossword:
    """
    Mutation function.
    Takes crossword and randomly changes position of random number of words.
    Mutation do not produce solutions with words out of the grid

    :param offspring: Crossword to mutate
    :return: Mutated crossword
    """
    mutated = offspring.copy()
    words_to_mutate = random.sample(mutated.words, k=random.randint(1, len(mutated.words)))
    for word in words_to_mutate:
        new_x, new_y, new_orientation = get_random_word_pos(word=word[0], g_size=mutated.grid_size)
        mutated.change_word(word[0], (new_x, new_y), new_orientation)

    return mutated


def mutate_shifting(offspring: Crossword) -> Crossword:
    """
    Mutation function.
    Takes crossword and randomly changes position of random number of words. Do not changes word orientation.
    Mutation do not produce solutions with words out of the grid

    :param offspring: Crossword to mutate
    :return: Mutated crossword
    """
    mutated = offspring.copy()

    word, (x, y), _, orientation = random.choice(mutated.words)

    if orientation == 'h':
        x = random.randint(0, mutated.grid_size - 1)
        y = random.randint(0, mutated.grid_size - len(word))
    if orientation == 'v':
        x = random.randint(0, mutated.grid_size - len(word))
        y = random.randint(0, mutated.grid_size - 1)

    mutated.change_word(word, (x, y), orientation)

    return mutated


def get_parents(population: tuple) -> tuple:
    def random_selection() -> tuple:
        return tuple(random.sample(population, 2))

    def roulette_wheel_selection() -> tuple:
        fitness_values = [individual.fitness for individual in population]
        selected_parents = tuple(random.choices(population, weights=fitness_values, k=2))
        return selected_parents

    def rank_based_selection() -> tuple:
        ranked_population = sorted(population, key=lambda x: get_fitness(x))
        selection_probabilities = [i / len(ranked_population) for i in range(1, len(ranked_population) + 1)]
        selected_parents = tuple(random.choices(ranked_population, weights=selection_probabilities, k=2))
        return selected_parents

    prob = random.randint(0, 99)
    if prob <= 33:
        return random_selection()
    elif 33 < prob <= 66:
        return roulette_wheel_selection()
    elif 66 < prob <= 99:
        return rank_based_selection()


def evolve_step(population: tuple, population_size: int, mutation_rate, crossover_rate) -> tuple:
    """
    Evolution step function.
    Generates new population from given one: crossover and mutation of current population, returning best individuals

    :param population: Population List(Crossword)
    :param population_size: Size of population (constant throughout the evolution)
    :param mutation_rate: Mutated offspring coefficient: 2 * population_size * (crossover_rate + mutation_rate)
    :param crossover_rate: Crossover offspring coefficient: crossover_rate * population_size
    :return: New population
    """

    m = []
    f = []

    for _ in range(int(population_size * crossover_rate)):
        m_, f_ = get_parents(population=population)
        m.append(m_)
        f.append(f_)

    new_population = list(population)

    offsprings = [crossover(m_, f_) for m_, f_ in zip(m, f)]
    new_population.extend(offsprings)

    new_population.extend(mutate(offspring) for offspring in offsprings)
    new_population.extend(mutate_shifting(offspring) for offspring in offsprings)

    mutated_size = range(int(population_size * mutation_rate))
    new_population.extend(mutate(random.choice(population)) for _ in mutated_size)
    new_population.extend(mutate_shifting(random.choice(population)) for _ in mutated_size)

    new_population.sort(key=lambda x: get_fitness(x))

    return tuple(new_population[:population_size])


def evolution(init_population: tuple, population_size: int, generations: int, mutation_rate: float,
              crossover_rate: float, words: str):
    """
    Evolution function.
    Gets initial population and evolves it for given number of generations.
    If for the 100 * len(words) generations the best fitness score does not change, the population is reinitialized.
    since the solution stuck in local minimum

    :param init_population: Initial population
    :param population_size: Size of population (constant throughout the evolution)
    :param generations: Max number of generations
    :param mutation_rate: Mutation rate, range: (0.0 - 1.0)
    :param crossover_rate: Crossover rate, range: (0.0 - 1.0)
    :param words: Words to place in the crossword
    :return:
    """
    population = init_population
    f_change = []

    for gen in range(generations):
        population = evolve_step(population=population, population_size=population_size,
                                 mutation_rate=mutation_rate, crossover_rate=crossover_rate)
        best_fitness = get_fitness(population[0])
        f_change.append(best_fitness)

        if best_fitness == 0:
            break

        if gen >= len(words) * 10:
            if f_change[gen - len(words) * 10] == f_change[-1]:
                population = initial_population(population_size=population_size, words=words)

    return f_change, population


def initial_population(population_size: int, words) -> tuple:
    """
    Generates initial population of given size

    :param population_size:
    :param words:
    :return:
    """
    population = [Crossword(words=words) for _ in range(population_size * 2)]
    population.sort(key=lambda x: get_fitness(x))
    return tuple(population[:population_size])


def generate_crossword(words, population_size, generations, mutation_rate, crossover_rate):
    """
    Starter for the evolution of crossword for given words

    :param words: Words to place in the crossword
    :param population_size: population size
    :param generations: Max number of generations
    :param mutation_rate: Mutation rate, range: (0.0 - 1.0)
    :param crossover_rate: Crossover rate, range: (0.0 - 1.0)
    :return:
    """

    population = initial_population(population_size=population_size, words=words)

    f_change, population = evolution(init_population=population, population_size=population_size, words=words,
                                     generations=generations, mutation_rate=mutation_rate,
                                     crossover_rate=crossover_rate)
    return f_change, population


def main():
    """
    Main function

    :return:
    """

    # Main parameters for the evolution algorithm
    population_size = 100  # population size
    generations = 10000000  # max number of generations
    mutation_rate = 1.3  # mutation rate
    crossover_rate = 1  # crossover rate

    counter = 1
    if not os.path.exists(f"{outputs_folder}"):
        os.mkdir(f"{outputs_folder}")

    while True:
        if not os.path.exists(f"{inputs_folder}/input{counter}.txt"):
            break
        if os.path.exists(f"{inputs_folder}/input{counter}.txt"):
            with open(f"{inputs_folder}/input{counter}.txt", "r") as input_file:
                words = input_file.read().splitlines()

            import time
            s = time.time()
            f_change, crosswords = generate_crossword(words=words, population_size=population_size,
                                                      generations=generations,
                                                      mutation_rate=mutation_rate, crossover_rate=crossover_rate)

            total = time.time() - s
            print(f"Score: {get_fitness(crosswords[0])} {len(words)} {total}")
            crosswords[0].display()

            answer = crosswords[0].get_words()
            with open(f"{outputs_folder}/output{counter}.txt", "w") as output_file:
                for word in answer:
                    output_file.write(f"{word[1]} {word[2]} {word[3]}\n")

            fitness_max = crosswords[0].fitness
            fitness_avg = sum([x.fitness for x in crosswords]) / len(crosswords)

        counter += 1


if __name__ == "__main__":
    main()
