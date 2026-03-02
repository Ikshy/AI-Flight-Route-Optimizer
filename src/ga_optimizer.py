import random

def calculate_fitness(individual, start, goal, dist_matrix):
    """Fitness = total distance of route: start -> permuted intermediates -> goal.
    Returns inf if any leg has no path (invalid route)."""
    path = [start] + individual + [goal]
    total_dist = 0.0
    for i in range(len(path) - 1):
        d = dist_matrix[path[i]][path[i + 1]]
        if d == float('inf'):
            return float('inf')
        total_dist += d
    return total_dist

def create_population(intermediates, pop_size):
    population = []
    for _ in range(pop_size):
        ind = intermediates[:]
        random.shuffle(ind)
        population.append(ind)
    return population

def crossover(parent1, parent2):
    """Order Crossover (OX) - preserves relative order, good for TSP-like routing."""
    size = len(parent1)
    if size < 2:
        return parent1[:]
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    used = set(child[start:end])
    j = 0
    for i in range(size):
        if child[i] is None:
            while parent2[j] in used:
                j = (j + 1) % size
            child[i] = parent2[j]
            j = (j + 1) % size
    return child

def mutate(individual, mutation_rate=0.2):
    """Swap mutation for diversity in route order."""
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(individual)), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

def select_parents(population, fitness, tournament_size=3):
    """Tournament selection - favors fitter (shorter) routes."""
    parents = []
    for _ in range(len(population)):
        candidates_idx = random.sample(range(len(population)), tournament_size)
        best_idx = min(candidates_idx, key=lambda idx: fitness[idx])
        parents.append(population[best_idx])
    return parents

def genetic_algorithm(start, goal, intermediates, dist_matrix,
                      pop_size=60, generations=80, mutation_rate=0.25):
    """Genetic Algorithm for multi-stop route optimization.
    Evolves permutations of intermediates to minimize total flight distance."""
    population = create_population(intermediates, pop_size)
    best_fitness = float('inf')
    best_individual = None

    for gen in range(generations):
        # Evaluate fitness
        fitness = [calculate_fitness(ind, start, goal, dist_matrix) for ind in population]

        # Track best
        min_fit = min(f for f in fitness if f < float('inf'))
        if min_fit < best_fitness:
            best_fitness = min_fit
            best_idx = fitness.index(min_fit)
            best_individual = population[best_idx][:]

        # Elitism + selection + crossover + mutation
        new_population = [best_individual[:]]  # keep best
        parents = select_parents(population, fitness)

        for i in range(1, pop_size, 2):
            p1 = parents[i - 1]
            p2 = parents[i % len(parents)]
            child1 = crossover(p1, p2)
            child2 = crossover(p2, p1)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population[:pop_size]

    high_level = [start] + best_individual + [goal]
    print(f"GA Best distance: {best_fitness:.1f} km after {generations} generations")
    return high_level, best_fitness