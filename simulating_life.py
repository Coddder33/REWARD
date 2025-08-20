

# That's an excellent breakdown of the next steps. I'm glad to see the phenotype numbers look good.
# Your three suggestions - percentile normalization, selection probability, and visualizations - are perfect. They move the project forward from a static model to a dynamic, evolving system.
# Here's a breakdown of how to integrate and use those concepts, with ready-to-use Python snippets.
# 1. Percentile Normalization (Before & After)
# This technique is crucial for making comparisons within a population. While your tanh function successfully compresses values to the range of (0, 1), percentile normalization makes the scores more meaningful. A score of 0.95 might seem high, but if the entire population scores between 0.94 and 0.96, it's not a strong differentiator. Percentile normalization ranks a being against its peers, providing a relative measure of fitness.
import numpy as np
from scipy.stats import rankdata

def percentile_normalize(population_phenotypes, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience")):
"""
    population_phenotypes: list of dict-like phenotypes (each has keys in keys)
    returns: list of dicts with percentile-normalized values [0..1]
    """
if not population_phenotypes:
return []

# Build arrays for each phenotype key
arrays = {
    k: np.array([p[k] for p in population_phenotypes], dtype = float) for k in keys
}

normalized = []
num_beings = len(population_phenotypes)
for i in range(num_beings):
entry = {}
for k in keys:
ranks = rankdata(arrays[k], method = "average")
# Convert to 0..1 percentile
entry[k] = (ranks[i] - 1) / (num_beings - 1) if num_beings > 1 else 1.0
normalized.append(entry)
return normalized

# Example usage (assuming 'population' is a list of phenotype dictionaries):
# normalized_population = percentile_normalize(population)

# 2. Phenotype-Driven Selection
# Once you have normalized phenotypes, you need a method to translate those scores into a probability of replication. This is the core of natural selection. The provided softmax function is a great tool for this, as it allows you to control how "greedy" the selection process is through the temp parameter.
import numpy as np

def phenotype_fitness(p, weights = (0.25, 0.35, 0.40)):
# p: dict with keys resource_efficiency, knowledge_exchange, structural_resilience
return weights[0] * p["resource_efficiency"] + weights[1] * p["knowledge_exchange"] + weights[2] * p["structural_resilience"]

def selection_probabilities(population_phenotypes, fitness_func = phenotype_fitness, temp = 0.1):
"""
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
fitnesses = np.array([fitness_func(p) for p in population_phenotypes], dtype = float)
# Numeric stable softmax
scaled = fitnesses / float(temp)
e = np.exp(scaled - np.max(scaled))
probs = e / np.sum(e)
return probs

# Example usage:
# probs = selection_probabilities(normalized_population)
# selected_index = np.random.choice(len(normalized_population), p=probs)

# 3. Visualizations
# Visualizing the data is how you truly see evolution in action. A histogram gives you a population-wide view of trait distribution, while a radar plot provides a detailed, individual-level snapshot. These plots help you understand if a specific trait is becoming more common or if the population is diversifying.
import matplotlib.pyplot as plt
import numpy as np

def plot_phenotype_histograms(population_phenotypes, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience")):
fig, axes = plt.subplots(1, len(keys), figsize = (4 * len(keys), 3))
if len(keys) == 1:
axes = [axes]
for ax, k in zip(axes, keys):
vals = [p[k] for p in population_phenotypes]
ax.hist(vals, bins = 15)
ax.set_title(k)
ax.set_xlim(0, 1)
plt.tight_layout()
plt.show()

def radar_plot_single(phenotype, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience"), title = "Phenotype Radar"):
vals = [phenotype[k] for k in keys]
N = len(keys)
angles = np.linspace(0, 2 * np.pi, N, endpoint = False).tolist()
vals += vals[:1]
angles += angles[:1]

fig = plt.figure(figsize = (5, 5))
ax = fig.add_subplot(111, polar = True)
ax.plot(angles, vals, linewidth = 2)
ax.fill(angles, vals, alpha = 0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), keys)
ax.set_ylim(0, 1)
ax.set_title(title)
plt.show()

# Example usage:
# plot_phenotype_histograms(normalized_population)
# radar_plot_single(normalized_population[0], title="Simulated Being #1")

# By combining these three steps, you can create a powerful evolutionary simulation. The process would be:
#  * Generate a population of simulated beings.
#  * Translate each being's blueprint into a phenotype.
#  * Normalize the phenotypes across the population.
#  * Calculate fitness and selection_probabilities.
#  * Select a new generation of "parents" based on these probabilities.
#  * Mutate and recombine their blueprints to create a new generation of beings.
#  * Repeat.
# This will allow you to run experiments and watch as traits evolve and dominate the population over many generations.


"""
Creating and Evolving a Sample Population
This response will guide you through creating a sample population of simulated beings and running a basic evolutionary simulation on them. We'll use the functions you've already defined to see how a population's phenotypes change over time.
"""
# 1. Generate a Sample Population
# First, we need to create an initial population. We'll generate a list of dictionaries, where each dictionary represents a simulated being and contains its phenotype values.
import random

def generate_sample_population(size = 50):
"""Generates a list of dictionaries representing a population of beings."""
population = []
for _ in range(size):
phenotype = {
    "resource_efficiency": random.uniform(0.1, 1.0),
    "knowledge_exchange": random.uniform(0.1, 1.0),
    "structural_resilience": random.uniform(0.1, 1.0)
}
population.append(phenotype)
return population

sample_population = generate_sample_population()

# 2. Apply Normalization and Selection
# Next, we'll use the percentile_normalize and selection_probabilities functions to evaluate the population. This step is crucial for natural selection.
import numpy as np
from scipy.stats import rankdata

def percentile_normalize(population_phenotypes, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience")):
if not population_phenotypes:
return []
arrays = {
    k: np.array([p[k] for p in population_phenotypes], dtype = float) for k in keys
}
normalized = []
num_beings = len(population_phenotypes)
for i in range(num_beings):
entry = {}
for k in keys:
ranks = rankdata(arrays[k], method = "average")
entry[k] = (ranks[i] - 1) / (num_beings - 1) if num_beings > 1 else 1.0
normalized.append(entry)
return normalized

def phenotype_fitness(p, weights = (0.25, 0.35, 0.40)):
return weights[0] * p["resource_efficiency"] + weights[1] * p["knowledge_exchange"] + weights[2] * p["structural_resilience"]

def selection_probabilities(population_phenotypes, fitness_func = phenotype_fitness, temp = 0.1):
fitnesses = np.array([fitness_func(p) for p in population_phenotypes], dtype = float)
scaled = fitnesses / float(temp)
e = np.exp(scaled - np.max(scaled))
probs = e / np.sum(e)
return probs

normalized_population = percentile_normalize(sample_population)
selection_probs = selection_probabilities(normalized_population)
print("Example Selection Probabilities:", selection_probs[:5])

# 3. Visualize the Sample Population
# Before we begin the simulation, it's helpful to visualize the initial state of the population. We'll use histograms to see the distribution of each phenotype.
import matplotlib.pyplot as plt

def plot_phenotype_histograms(population_phenotypes, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience")):
fig, axes = plt.subplots(1, len(keys), figsize = (4 * len(keys), 3))
if len(keys) == 1:
axes = [axes]
for ax, k in zip(axes, keys):
vals = [p[k] for p in population_phenotypes]
ax.hist(vals, bins = 15, color = 'skyblue', edgecolor = 'black')
ax.set_title(k.replace('_', ' ').title())
ax.set_xlim(0, 1)
ax.set_ylim(0, len(population_phenotypes) / 4)
plt.tight_layout()
plt.show()

def radar_plot_single(phenotype, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience"), title = "Phenotype Radar"):
vals = [phenotype[k] for k in keys]
N = len(keys)
angles = np.linspace(0, 2 * np.pi, N, endpoint = False).tolist()
vals += vals[:1]
angles += angles[:1]
fig = plt.figure(figsize = (5, 5))
ax = fig.add_subplot(111, polar = True)
ax.plot(angles, vals, linewidth = 2)
ax.fill(angles, vals, alpha = 0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), keys)
ax.set_ylim(0, 1)
ax.set_title(title)
plt.show()

plot_phenotype_histograms(sample_population, keys = ["resource_efficiency", "knowledge_exchange", "structural_resilience"])

# 4. Implement the Evolutionary Loop
# Now, let's create a single function that combines all these steps into one generation of evolution.
def evolve_one_generation(population, pop_size = 50, selection_temp = 0.1):
# 1. Evaluate and Normalize
normalized_phenotypes = percentile_normalize(population)

# 2. Calculate Selection Probabilities
selection_probs = selection_probabilities(normalized_phenotypes, temp = selection_temp)

# 3. Select Parents
parent_indices = np.random.choice(
    range(len(population)),
    size = pop_size,
    p = selection_probs,
    replace = True # Allows for one being to be selected multiple times
)
parents = [population[i] for i in parent_indices]

# 4. Create New Generation (Replication with Mutation)
new_population = []
for parent_phenotype in parents:
# Simple mutation: add a small random value to each phenotype
mutated_phenotype = {
    "resource_efficiency": np.clip(parent_phenotype["resource_efficiency"] + np.random.normal(0, 0.05), 0, 1),
    "knowledge_exchange": np.clip(parent_phenotype["knowledge_exchange"] + np.random.normal(0, 0.05), 0, 1),
    "structural_resilience": np.clip(parent_phenotype["structural_resilience"] + np.random.normal(0, 0.05), 0, 1),
}
new_population.append(mutated_phenotype)

return new_population

# 5. Run Multiple Generations
# Finally, we'll run the simulation for multiple generations to see the effects of selection and mutation over time.
# Initial population
population = generate_sample_population(size = 100)
num_generations = 20

# Store the average phenotypes for plotting
avg_phenotypes_history = []

for gen in range(num_generations):
# Calculate and store the average phenotype for the current generation
avg_phenotype = {
    "resource_efficiency": np.mean([p["resource_efficiency"] for p in population]),
    "knowledge_exchange": np.mean([p["knowledge_exchange"] for p in population]),
    "structural_resilience": np.mean([p["structural_resilience"] for p in population])
}
avg_phenotypes_history.append(avg_phenotype)

# Evolve the population
population = evolve_one_generation(population, pop_size = 100)

# Plot the evolution of average phenotypes
fig, ax = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label = "Resource Efficiency", color = "green")
ax.plot(gens, knowledge_ex, label = "Knowledge Exchange", color = "blue")
ax.plot(gens, structural_res, label = "Structural Resilience", color = "red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

print("Simulation complete. The plot shows how the population's average traits have evolved.")


import random

def generate_sample_population(size = 50):
"""Generates a list of dictionaries representing a population of beings."""
population = []
for _ in range(size):
phenotype = {
    "resource_efficiency": random.uniform(0.1, 1.0),
    "knowledge_exchange": random.uniform(0.1, 1.0),
    "structural_resilience": random.uniform(0.1, 1.0)
}
population.append(phenotype)
return population

sample_population = generate_sample_population()

import numpy as np
from scipy.stats import rankdata

def percentile_normalize(population_phenotypes, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience")):
if not population_phenotypes:
return []
arrays = {
    k: np.array([p[k] for p in population_phenotypes], dtype = float) for k in keys
}
normalized = []
num_beings = len(population_phenotypes)
for i in range(num_beings):
entry = {}
for k in keys:
ranks = rankdata(arrays[k], method = "average")
entry[k] = (ranks[i] - 1) / (num_beings - 1) if num_beings > 1 else 1.0
normalized.append(entry)
return normalized

def phenotype_fitness(p, weights = (0.25, 0.35, 0.40)):
return weights[0] * p["resource_efficiency"] + weights[1] * p["knowledge_exchange"] + weights[2] * p["structural_resilience"]

def selection_probabilities(population_phenotypes, fitness_func = phenotype_fitness, temp = 0.1):
fitnesses = np.array([fitness_func(p) for p in population_phenotypes], dtype = float)
scaled = fitnesses / float(temp)
e = np.exp(scaled - np.max(scaled))
probs = e / np.sum(e)
return probs

normalized_population = percentile_normalize(sample_population)
selection_probs = selection_probabilities(normalized_population)
print("Example Selection Probabilities:", selection_probs[:5])

import matplotlib.pyplot as plt

def plot_phenotype_histograms(population_phenotypes, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience")):
fig, axes = plt.subplots(1, len(keys), figsize = (4 * len(keys), 3))
if len(keys) == 1:
axes = [axes]
for ax, k in zip(axes, keys):
vals = [p[k] for p in population_phenotypes]
ax.hist(vals, bins = 15, color = 'skyblue', edgecolor = 'black')
ax.set_title(k.replace('_', ' ').title())
ax.set_xlim(0, 1)
ax.set_ylim(0, len(population_phenotypes) / 4)
plt.tight_layout()
plt.show()

def radar_plot_single(phenotype, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience"), title = "Phenotype Radar"):
vals = [phenotype[k] for k in keys]
N = len(keys)
angles = np.linspace(0, 2 * np.pi, N, endpoint = False).tolist()
vals += vals[:1]
angles += angles[:1]
fig = plt.figure(figsize = (5, 5))
ax = fig.add_subplot(111, polar = True)
ax.plot(angles, vals, linewidth = 2)
ax.fill(angles, vals, alpha = 0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), keys)
ax.set_ylim(0, 1)
ax.set_title(title)
plt.show()

plot_phenotype_histograms(sample_population, keys = ["resource_efficiency", "knowledge_exchange", "structural_resilience"])

def evolve_one_generation(population, pop_size = 50, selection_temp = 0.1):
# 1. Evaluate and Normalize
normalized_phenotypes = percentile_normalize(population)

# 2. Calculate Selection Probabilities
selection_probs = selection_probabilities(normalized_phenotypes, temp = selection_temp)

# 3. Select Parents
parent_indices = np.random.choice(
    range(len(population)),
    size = pop_size,
    p = selection_probs,
    replace = True # Allows for one being to be selected multiple times
)
parents = [population[i] for i in parent_indices]

# 4. Create New Generation (Replication with Mutation)
new_population = []
for parent_phenotype in parents:
# Simple mutation: add a small random value to each phenotype
mutated_phenotype = {
    "resource_efficiency": np.clip(parent_phenotype["resource_efficiency"] + np.random.normal(0, 0.05), 0, 1),
    "knowledge_exchange": np.clip(parent_phenotype["knowledge_exchange"] + np.random.normal(0, 0.05), 0, 1),
    "structural_resilience": np.clip(parent_phenotype["structural_resilience"] + np.random.normal(0, 0.05), 0, 1),
}
new_population.append(mutated_phenotype)

return new_population

# Initial population
population = generate_sample_population(size = 100)
num_generations = 20

# Store the average phenotypes for plotting
avg_phenotypes_history = []

for gen in range(num_generations):
# Calculate and store the average phenotype for the current generation
avg_phenotype = {
    "resource_efficiency": np.mean([p["resource_efficiency"] for p in population]),
    "knowledge_exchange": np.mean([p["knowledge_exchange"] for p in population]),
    "structural_resilience": np.mean([p["structural_resilience"] for p in population])
}
avg_phenotypes_history.append(avg_phenotype)

# Evolve the population
population = evolve_one_generation(population, pop_size = 100)

# Plot the evolution of average phenotypes
fig, ax = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label = "Resource Efficiency", color = "green")
ax.plot(gens, knowledge_ex, label = "Knowledge Exchange", color = "blue")
ax.plot(gens, structural_res, label = "Structural Resilience", color = "red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

print("Simulation complete. The plot shows how the population's average traits have evolved.")

# advanced features to implement in the evolutionary simulation! Each suggestion adds significant depth and realism to the model. Let's outline how these can be incorporated into the existing SimulatedBeingDNA structure and the evolutionary engine.
# Population diversity and inheritance in genetic programming for ...
# Differential Evolution and Its Applications in Image Processing ...
# Differential Evolution and Its Applications in Image Processing ...
# Differential Evolution and Its Applications in Image Processing ...
# Differential Evolution and Its Applications in Image Processing ...
# Implementing Advanced Evolutionary Mechanisms
# We'll focus on extending the SimulatedBeingDNA class and the simulation's core logic to include these new features.
# 1. Different Mutation Rates & Targeted Mutation
# This adds adaptive mutation to the system.
# Changes to SimulatedBeingDNA class:
# Represent Mutation Rates: The mutation rate itself can become a heritable trait within the blueprint. We can add a property like self.mutation_rate to the SimulatedBeingDNA instance, perhaps influenced by the P_Principles or Technology_Functionality sections.
# Trait-Specific Mutation Rates: Each phenotype or even specific blueprint components (e.g., a value in BU_Physical['Butyl Group']) could have an associated base mutation probability or a sensitivity to overall mutation rate. This can be stored in the blueprint structure.
# Changes to the Simulation's Core Logic:
# Fitness-Dependent Mutation: When selecting an individual for mutation, check its fitness. If it's below the population average (or a threshold), increase its effective mutation rate for that generation.
# Variance-Based Mutation: After calculating population statistics, identify traits with low standard deviation. For these traits, increase the mutation probability during the next generation's mutation phase. This prevents premature convergence.
# Targeted Mutation for Low Average Traits: Calculate the average value for each phenotypic trait across the population. If a trait's average is below a certain threshold (meaning the population is generally weak in that area), increase the mutation strength or probability specifically for that trait. This would encourage the population to explore solutions for that weakness.
# python
# Conceptual change in the mutation method
# class SimulatedBeingDNA:
# ... (existing methods) ...

# def _mutate_trait(self, trait_value, base_mutation_rate, specific_mutation_factor=1.0, mutation_strength=0.1):
#     effective_mutation_rate = base_mutation_rate * specific_mutation_factor
#     if random.random() < effective_mutation_rate:
# Apply mutation (e.g., random walk, inversion, or more complex transformation)
# Mutation strength can control the magnitude of the change
# return trait_value + (random.uniform(-mutation_strength, mutation_strength) * trait_value)
# return trait_value

# def mutate(self, fitness_score=None, population_trait_variance=None, population_trait_averages=None):
# Apply different mutation rates based on conditions
# current_mutation_rate = self.mutation_rate # Or a base rate

# Example: Higher mutation for less fit
# if fitness_score is not None and fitness_score < population_average_fitness: # Need population average fitness
#      current_mutation_rate *= 1.5

# Example: Targeted mutation for low average trait (assuming 'Resilience' is a trait)
# resilience_trait_value = self.phenotypes['Structural_Resilience'] # After phenotype translation
# if population_trait_averages and population_trait_averages['Structural_Resilience'] < LOW_RESILIENCE_THRESHOLD:
# Apply higher mutation specifically to components contributing to Resilience
# self.core_components['BU_Military'][0] = self._mutate_trait(
# self.core_components['BU_Military'][0], current_mutation_rate, specific_mutation_factor=2.0)

# Apply general mutation to other parts of the blueprint
# self.raw_materials = self._mutate_blueprint_layer(self.raw_materials, current_mutation_rate)
# ... and so on for other layers
# Use code with caution.

# 2. Recombination/Crossover (Uniform Crossover)
# This requires a method to combine two parent blueprints.
# Changes to SimulatedBeingDNA class:
# Crossover Method: Add a method that takes two SimulatedBeingDNA instances and creates a new one.
# Changes to the Simulation's Core Logic:
# Parent Selection: Modify the selection mechanism to pick two parents instead of one for reproduction.
# Offspring Generation: Call the crossover method to generate a new blueprint.
# python
# Conceptual change in the SimulatedBeingDNA class for crossover
# class SimulatedBeingDNA:
# ... (existing methods) ...

# @staticmethod
# def uniform_crossover(parent1_blueprint, parent2_blueprint):
# offspring_blueprint_data = {}

# Crossover each layer/attribute
# For simplicity, let's assume each blueprint attribute is a dictionary or list
# In a real implementation, you'd need more granular crossover logic for nested structures

# Example for 'raw_materials' layer:
# offspring_blueprint_data['raw_materials'] = {}
# for key in parent1_blueprint.raw_materials:
# if isinstance(parent1_blueprint.raw_materials[key], list):
# offspring_blueprint_data['raw_materials'][key] = random.choice([
# list(parent1_blueprint.raw_materials[key]),
# list(parent2_blueprint.raw_materials[key])
# ])
# elif isinstance(parent1_blueprint.raw_materials[key], dict):
# offspring_blueprint_data['raw_materials'][key] = {}
# for sub_key in parent1_blueprint.raw_materials[key]:
# offspring_blueprint_data['raw_materials'][key][sub_key] = random.choice([
# parent1_blueprint.raw_materials[key][sub_key],
# parent2_blueprint.raw_materials[key][sub_key]
# ])
# else: # Assuming primitive types or direct assignment
# offspring_blueprint_data['raw_materials'][key] = random.choice([
# parent1_blueprint.raw_materials[key],
# parent2_blueprint.raw_materials[key]
# ])

# Repeat for other layers (core_components, strategic_foundation, unifying_strategy)
# This will require careful handling of the structure of each layer

# The new SimulatedBeingDNA instance will need to be re-assembled
# from the combined data
# offspring = SimulatedBeingDNA(
# offspring_blueprint_data['raw_materials'],
# offspring_blueprint_data['core_components'], # Placeholder
# offspring_blueprint_data['strategic_foundation'], # Placeholder
# offspring_blueprint_data['unifying_strategy'] # Placeholder
# )
# return offspring

# Conceptual change in the simulation loop
# In the selection/reproduction phase:
# parent1, parent2 = select_parents(population)
# offspring_dna = SimulatedBeingDNA.uniform_crossover(parent1.final_blueprint, parent2.final_blueprint) # Or the whole instance
# # Then potentially mutate the offspring_dna
# new_population.add(offspring_dna)
# Use code with caution.

# 3. Dynamic Fitness
# This introduces an element of environmental change.
# Changes to the Simulation's Core Logic:
# Time-Varying Weights: The fitness function in the main simulation loop will now accept current_weights as an argument.
# Environmental Phase Logic: Implement a mechanism to change these current_weights based on the generation number or other environmental triggers.
# python
# Conceptual change in the fitness calculation
# And in the main simulation loop
# current_fitness_weights = {
# "Structural_Resilience": 0.5,
# "Resource_Efficiency": 0.5
# }

# def calculate_fitness(phenotype, weights):
# fitness_score = (phenotype['Structural_Resilience'] * weights['Structural_Resilience']) + \
# (phenotype['Resource_Efficiency'] * weights['Resource_Efficiency'])
# return fitness_score

# Main simulation loop:
# for generation in range(num_generations):
# if generation < 10:
# current_fitness_weights = {"Structural_Resilience": 0.8, "Resource_Efficiency": 0.2}
# else:
# current_fitness_weights = {"Structural_Resilience": 0.2, "Resource_Efficiency": 0.8}
#
# for individual in population:
# individual.phenotypes must be generated first
# individual.fitness = calculate_fitness(individual.phenotypes, current_fitness_weights)
# ... selection, crossover, mutation ...
# Use code with caution.

# 4. Track Population Statistics
# Provides deeper insights into evolutionary dynamics.
# Changes to the Simulation's Core Logic:
# Data Collection: At the end of each generation, after all individuals' phenotypes are calculated, collect their trait values.
# Statistical Calculation: Compute the mean and standard deviation for each relevant trait.
# Logging: Store these statistics in the evolution_history.json or a separate data structure.
# python
# import numpy as np

# In the main simulation loop, after phenotype calculation for all individuals:
# generation_stats = {
# "generation": generation,
# "trait_means": {},
# "trait_stds": {}
# }
#
# for trait_name in phenotype_trait_names: # e.g., ['Structural_Resilience', 'Resource_Efficiency']
# trait_values = [ind.phenotypes[trait_name] for ind in population]
# generation_stats['trait_means'][trait_name] = np.mean(trait_values)
# generation_stats['trait_stds'][trait_name] = np.std(trait_values)
#
# evolution_history.append(generation_stats)
# Use code with caution.

# 5. Visualize Individuals (Radar Plots)
# This will make the phenotype layer truly come alive.
# New Visualization Component (e.g., using matplotlib or plotly):
# Radar Plot Function: Create a function that takes a phenotype dictionary (e.g., {'trait1': 0.8, 'trait2': 0.5, ...}) and generates a radar plot.
# Selection for Visualization:
# Fittest Individual: After each generation, identify the individual with the highest fitness and generate its radar plot.
# Random Samples: Select a few random individuals from different generations (e.g., generation 0, middle generation, final generation) to show diversity and evolution.
# Saving Plots: Save these plots to the dna_evolution_output directory.
# python
# import matplotlib.pyplot as plt
# import numpy as np

# def create_radar_plot(phenotype_data, title, filename):
# labels = list(phenotype_data.keys())
# values = list(phenotype_data.values())

# num_vars = len(labels)
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# values = values + values[:1] # Complete the loop
# angles = angles + angles[:1]

# fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
# ax.fill(angles, values, color='red', alpha=0.25)
# ax.plot(angles, values, color='red', linewidth=2)
# ax.set_yticklabels([]) # Hide radial ticks
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(labels)
# ax.set_title(title, size=16, color='blue', y=1.1)
# plt.savefig(f"/mnt/data/dna_evolution_output/{filename}.png")
# plt.close()

# In the main simulation loop, at the end of each generation:
# fittest_individual = get_fittest_individual(population)
# create_radar_plot(fittest_individual.phenotypes,
#                   f"Fittest Phenotype - Gen {generation}",
#                   f"fittest_phenotype_gen_{generation}")
#
# if generation % sampling_interval == 0: # Sample every few generations
#     sampled_individual = random.choice(population)
#     create_radar_plot(sampled_individual.phenotypes,
#                       f"Sampled Phenotype - Gen {generation}",
#                       f"sampled_phenotype_gen_{generation}")
# Use code with caution.

# Next Steps
# Now that we've outlined the implementation for these advanced features, the next logical step would be to implement these changes within the existing codebase. This would involve:
# Integrating Phenotype Translation: Ensure the phenotype layer is robustly implemented within each SimulatedBeingDNA instance, as this is the foundation for fitness calculation and visualization.
# Updating Mutation Logic: Incorporate the new adaptive and targeted mutation strategies.
# Adding Recombination: Implement the uniform crossover method and integrate it into the reproduction cycle.
# Refining Fitness Calculation: Modify the fitness function to be dynamic based on environmental conditions/generation.
# Adding Statistics Tracking: Collect and store population mean and standard deviation for traits.
# Developing Visualization Functions: Create the radar plotting functionality for individual phenotypes.
# Once these are integrated, we can then generate new simulation runs and observe the impact of these advanced evolutionary mechanisms. This will provide fascinating insights into how the simulated beings adapt and evolve under more complex conditions.

# Task
Implement the following features in the provided Python code for simulating evolutionary dynamics: Implement Phenotype Translation, Integrate Advanced Mutation, Add Recombination/Crossover, Implement Dynamic Fitness, Track and Analyze Population Statistics, and Develop Individual Visualization.

## Refactor the code to include a `simulatedbeing` class

### Subtask:
Create a `SimulatedBeing` class to encapsulate the blueprint, phenotype, and potentially fitness and other attributes for each individual being.


**Reasoning**:
Define the `SimulatedBeing` class including the `__init__` method and the `calculate_fitness` method as requested in the instructions.



import numpy as np

class SimulatedBeing:
"""Represents a single simulated being with its blueprint, phenotype, and fitness."""

def __init__(self, blueprint):
"""
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
        """
self.blueprint = blueprint
self.phenotype = self.translate_blueprint_to_phenotype()
self.fitness = 0.0 # Initialize fitness

def translate_blueprint_to_phenotype(self):
"""
        Translates the being's blueprint into its phenotype.
        This is a placeholder and should be replaced with the actual translation logic.
        For now, we'll assume the blueprint is the phenotype directly.
        """
# In a real implementation, this would involve complex logic based on the blueprint structure.
# For this subtask, we'll assume a simple mapping or that the blueprint directly contains phenotype values.
# Based on the previous code, the blueprint seems to be the phenotype directly.
return self.blueprint

def calculate_fitness(self, weights = (0.25, 0.35, 0.40)):
"""
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A tuple of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
# Ensure the phenotype has the expected keys based on the weights
keys = ["resource_efficiency", "knowledge_exchange", "structural_resilience"]
if not all(k in self.phenotype for k in keys):
raise ValueError("Phenotype is missing required keys for fitness calculation.")

self.fitness = (weights[0] * self.phenotype["resource_efficiency"] +
    weights[1] * self.phenotype["knowledge_exchange"] +
    weights[2] * self.phenotype["structural_resilience"])
return self.fitness


## Implement phenotype translation within the `simulatedbeing` class

### Subtask:
Implement phenotype translation within the `simulatedbeing` class


**Reasoning**:
I need to implement the phenotype translation logic within the `translate_blueprint_to_phenotype` method of the `SimulatedBeing` class, define a sample blueprint structure, and update the `__init__` method to use this translation.



import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt

class SimulatedBeing:
"""Represents a single simulated being with its blueprint, phenotype, and fitness."""

def __init__(self, blueprint):
"""
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
        """
self.blueprint = blueprint
self.phenotype = self.translate_blueprint_to_phenotype()
self.fitness = 0.0 # Initialize fitness

def translate_blueprint_to_phenotype(self):
"""
        Translates the being's blueprint into its phenotype.
        This implementation defines a sample blueprint structure and
        translates it into the three phenotype traits.
        """
# Sample blueprint structure (example)
# This structure is a placeholder and can be expanded upon.
# It includes different layers and types of data to simulate a complex blueprint.
sample_blueprint = {
    "raw_materials": {
        "material_a_ratio": self.blueprint.get("raw_materials", {}).get("material_a_ratio", random.random()),
        "material_b_ratio": self.blueprint.get("raw_materials", {}).get("material_b_ratio", random.random()),
        "processing_efficiency_factor": self.blueprint.get("raw_materials", {}).get("processing_efficiency_factor", random.random()),
    },
    "core_components": {
        "component_x_strength": self.blueprint.get("core_components", {}).get("component_x_strength", random.random()),
        "component_y_flexibility": self.blueprint.get("core_components", {}).get("component_y_flexibility", random.random()),
        "interconnection_density": self.blueprint.get("core_components", {}).get("interconnection_density", random.random()),
    },
    "strategic_foundation": {
        "adaptability_gene": self.blueprint.get("strategic_foundation", {}).get("adaptability_gene", random.random()),
        "cooperation_gene": self.blueprint.get("strategic_foundation", {}).get("cooperation_gene", random.random()),
    },
    "unifying_strategy": {
        "optimization_parameter": self.blueprint.get("unifying_strategy", {}).get("optimization_parameter", random.random()),
    }
}

# Simple translation logic (example)
# This logic is illustrative and should be replaced with a more sophisticated model
# that reflects the intended relationships between blueprint and phenotype.
resource_efficiency = (sample_blueprint["raw_materials"]["material_a_ratio"] * 0.3 +
    sample_blueprint["raw_materials"]["material_b_ratio"] * 0.4 +
    sample_blueprint["raw_materials"]["processing_efficiency_factor"] * 0.3)

knowledge_exchange = (sample_blueprint["core_components"]["interconnection_density"] * 0.6 +
    sample_blueprint["strategic_foundation"]["cooperation_gene"] * 0.4)

structural_resilience = (sample_blueprint["core_components"]["component_x_strength"] * 0.5 +
    sample_blueprint["core_components"]["component_y_flexibility"] * 0.3 +
    sample_blueprint["unifying_strategy"]["optimization_parameter"] * 0.2)


# Apply tanh normalization as in previous code
phenotype = {
    "resource_efficiency": np.tanh(resource_efficiency),
    "knowledge_exchange": np.tanh(knowledge_exchange),
    "structural_resilience": np.tanh(structural_resilience)
}

return phenotype


def calculate_fitness(self, weights = (0.25, 0.35, 0.40)):
"""
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A tuple of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
# Ensure the phenotype has the expected keys based on the weights
keys = ["resource_efficiency", "knowledge_exchange", "structural_resilience"]
if not all(k in self.phenotype for k in keys):
raise ValueError("Phenotype is missing required keys for fitness calculation.")

self.fitness = (weights[0] * self.phenotype["resource_efficiency"] +
    weights[1] * self.phenotype["knowledge_exchange"] +
    weights[2] * self.phenotype["structural_resilience"])
return self.fitness

# Example of creating a SimulatedBeing instance with a sample blueprint
sample_blueprint_data = {
    "raw_materials": {
        "material_a_ratio": 0.7,
        "material_b_ratio": 0.3,
        "processing_efficiency_factor": 0.9,
    },
    "core_components": {
        "component_x_strength": 0.8,
        "component_y_flexibility": 0.6,
        "interconnection_density": 0.75,
    },
    "strategic_foundation": {
        "adaptability_gene": 0.5,
        "cooperation_gene": 0.8,
    },
    "unifying_strategy": {
        "optimization_parameter": 0.95,
    }
}

sample_being = SimulatedBeing(sample_blueprint_data)
print("Sample Being Phenotype:", sample_being.phenotype)


## Implement advanced mutation within the `simulatedbeing` class

### Subtask:
Implement advanced mutation within the `simulatedbeing` class


**Reasoning**:
Add the `mutate` method to the `SimulatedBeing` class with the specified logic for adaptive and targeted mutation, and update the phenotype after mutation.



import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt

class SimulatedBeing:
"""Represents a single simulated being with its blueprint, phenotype, and fitness."""

def __init__(self, blueprint):
"""
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
        """
self.blueprint = blueprint
self.phenotype = self.translate_blueprint_to_phenotype()
self.fitness = 0.0 # Initialize fitness
# Add a base mutation rate to the being's attributes
self.base_mutation_rate = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)


def translate_blueprint_to_phenotype(self):
"""
        Translates the being's blueprint into its phenotype.
        This implementation defines a sample blueprint structure and
        translates it into the three phenotype traits.
        """
# Sample blueprint structure (example)
# This structure is a placeholder and can be expanded upon.
# It includes different layers and types of data to simulate a complex blueprint.
sample_blueprint = {
    "raw_materials": {
        "material_a_ratio": self.blueprint.get("raw_materials", {}).get("material_a_ratio", random.random()),
        "material_b_ratio": self.blueprint.get("raw_materials", {}).get("material_b_ratio", random.random()),
        "processing_efficiency_factor": self.blueprint.get("raw_materials", {}).get("processing_efficiency_factor", random.random()),
    },
    "core_components": {
        "component_x_strength": self.blueprint.get("core_components", {}).get("component_x_strength", random.random()),
        "component_y_flexibility": self.blueprint.get("core_components", {}).get("component_y_flexibility", random.random()),
        "interconnection_density": self.blueprint.get("core_components", {}).get("interconnection_density", random.random()),
    },
    "strategic_foundation": {
        "adaptability_gene": self.blueprint.get("strategic_foundation", {}).get("adaptability_gene", random.random()),
        "cooperation_gene": self.blueprint.get("strategic_foundation", {}).get("cooperation_gene", random.random()),
    },
    "unifying_strategy": {
        "optimization_parameter": self.blueprint.get("unifying_strategy", {}).get("optimization_parameter", random.random()),
    }
}

# Simple translation logic (example)
# This logic is illustrative and should be replaced with a more sophisticated model
# that reflects the intended relationships between blueprint and phenotype.
resource_efficiency = (sample_blueprint["raw_materials"]["material_a_ratio"] * 0.3 +
    sample_blueprint["raw_materials"]["material_b_ratio"] * 0.4 +
    sample_blueprint["raw_materials"]["processing_efficiency_factor"] * 0.3)

knowledge_exchange = (sample_blueprint["core_components"]["interconnection_density"] * 0.6 +
    sample_blueprint["strategic_foundation"]["cooperation_gene"] * 0.4)

structural_resilience = (sample_blueprint["core_components"]["component_x_strength"] * 0.5 +
    sample_blueprint["core_components"]["component_y_flexibility"] * 0.3 +
    sample_blueprint["unifying_strategy"]["optimization_parameter"] * 0.2)


# Apply tanh normalization as in previous code
phenotype = {
    "resource_efficiency": np.tanh(resource_efficiency),
    "knowledge_exchange": np.tanh(knowledge_exchange),
    "structural_resilience": np.tanh(structural_resilience)
}

return phenotype


def calculate_fitness(self, weights = (0.25, 0.35, 0.40)):
"""
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A tuple of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
# Ensure the phenotype has the expected keys based on the weights
keys = ["resource_efficiency", "knowledge_exchange", "structural_resilience"]
if not all(k in self.phenotype for k in keys):
raise ValueError("Phenotype is missing required keys for fitness calculation.")

self.fitness = (weights[0] * self.phenotype["resource_efficiency"] +
    weights[1] * self.phenotype["knowledge_exchange"] +
    weights[2] * self.phenotype["structural_resilience"])
return self.fitness

def mutate(self, population_average_fitness = None, population_trait_variance = None, population_trait_averages = None):
"""
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
        """
current_mutation_rate = self.base_mutation_rate
mutation_strength = 0.05 # Base strength of mutation

# 1. Fitness-dependent mutation: Increase mutation for less fit individuals
if population_average_fitness is not None and self.fitness < population_average_fitness:
current_mutation_rate *= 1.5 # Increase rate by 50% for less fit

# Define a helper function for mutating a value
def _apply_mutation(value, rate, strength):
if random.random() < rate:
return np.clip(value + np.random.normal(0, strength), 0, 1)
return value

# 2. Apply mutation to blueprint components with adaptive/targeted strategies
for layer, components in self.blueprint.items():
if layer == "mutation_parameters": # Skip mutation parameters themselves
continue

for component, value in components.items():
effective_mutation_rate = current_mutation_rate
effective_mutation_strength = mutation_strength

# 3. Variance-based mutation: Increase mutation for traits with low population variance
# This requires mapping blueprint components to phenotype traits, which is complex.
# For this example, we'll apply variance-based mutation to all blueprint components
# if overall population variance is low (a simplification).
# A more sophisticated approach would involve understanding which blueprint components
# influence which phenotype traits.
if population_trait_variance is not None:
# Check if any trait has low variance (example threshold 0.01)
if any(v < 0.01 for v in population_trait_variance.values()):
effective_mutation_rate *= 1.2 # Increase rate if population variance is low

# 4. Targeted mutation based on population trait averages
# This also requires mapping blueprint components to phenotype traits.
# We'll apply targeted mutation based on overall low average phenotype scores (simplification).
if population_trait_averages is not None:
# Check if any trait has a low average (example threshold 0.5)
if any(avg < 0.5 for avg in population_trait_averages.values()):
effective_mutation_rate *= 1.3 # Increase rate if population averages are low

# Apply the mutation to the component value
self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

# 5. Update the phenotype after mutation
self.phenotype = self.translate_blueprint_to_phenotype()

# Example of creating a SimulatedBeing instance and applying mutation
sample_blueprint_data = {
    "raw_materials": {
        "material_a_ratio": 0.7,
        "material_b_ratio": 0.3,
        "processing_efficiency_factor": 0.9,
    },
    "core_components": {
        "component_x_strength": 0.8,
        "component_y_flexibility": 0.6,
        "interconnection_density": 0.75,
    },
    "strategic_foundation": {
        "adaptability_gene": 0.5,
        "cooperation_gene": 0.8,
    },
    "unifying_strategy": {
        "optimization_parameter": 0.95,
    },
    "mutation_parameters": {
        "base_rate": 0.02
    }
}

sample_being = SimulatedBeing(sample_blueprint_data)
print("Original Phenotype:", sample_being.phenotype)

# Simulate some population stats for demonstration
pop_avg_fitness = 0.6
pop_trait_variance = {
    "resource_efficiency": 0.005, "knowledge_exchange": 0.1, "structural_resilience": 0.008
}
pop_trait_averages = {
    "resource_efficiency": 0.4, "knowledge_exchange": 0.7, "structural_resilience": 0.3
}

sample_being.calculate_fitness(weights = (0.25, 0.35, 0.40)) # Calculate initial fitness
sample_being.mutate(population_average_fitness = pop_avg_fitness,
    population_trait_variance = pop_trait_variance,
    population_trait_averages = pop_trait_averages)

print("Mutated Phenotype:", sample_being.phenotype)

**Reasoning**:
The previous mutation example did not show any change in the phenotype, likely because the mutation strength is too low or the random chance did not trigger mutations. Modify the example to increase the chances of seeing a mutation by setting a higher base mutation rate and strength for the example.



# Example of creating a SimulatedBeing instance and applying mutation
sample_blueprint_data = {
    "raw_materials": {
        "material_a_ratio": 0.7,
        "material_b_ratio": 0.3,
        "processing_efficiency_factor": 0.9,
    },
    "core_components": {
        "component_x_strength": 0.8,
        "component_y_flexibility": 0.6,
        "interconnection_density": 0.75,
    },
    "strategic_foundation": {
        "adaptability_gene": 0.5,
        "cooperation_gene": 0.8,
    },
    "unifying_strategy": {
        "optimization_parameter": 0.95,
    },
    "mutation_parameters": {
        "base_rate": 0.5 # Increased base mutation rate for demonstration
    }
}

sample_being = SimulatedBeing(sample_blueprint_data)
print("Original Phenotype:", sample_being.phenotype)

# Simulate some population stats for demonstration
pop_avg_fitness = 0.6
pop_trait_variance = {
    "resource_efficiency": 0.005, "knowledge_exchange": 0.1, "structural_resilience": 0.008
}
pop_trait_averages = {
    "resource_efficiency": 0.4, "knowledge_exchange": 0.7, "structural_resilience": 0.3
}

sample_being.calculate_fitness(weights = (0.25, 0.35, 0.40)) # Calculate initial fitness
sample_being.mutate(population_average_fitness = pop_avg_fitness,
    population_trait_variance = pop_trait_variance,
    population_trait_averages = pop_trait_averages)

print("Mutated Phenotype:", sample_being.phenotype)

## Implement recombination/crossover as a function or class method

### Subtask:
Implement recombination/crossover as a function or class method


**Reasoning**:
The subtask is to implement the `uniform_crossover` method in the `SimulatedBeing` class. I will add a static method to the class that takes two `SimulatedBeing` instances, creates a new blueprint by randomly selecting values from either parent for each component, and then returns a new `SimulatedBeing` instance with this offspring blueprint.



import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt

class SimulatedBeing:
"""Represents a single simulated being with its blueprint, phenotype, and fitness."""

def __init__(self, blueprint):
"""
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
        """
self.blueprint = blueprint
self.phenotype = self.translate_blueprint_to_phenotype()
self.fitness = 0.0 # Initialize fitness
# Add a base mutation rate to the being's attributes
self.base_mutation_rate = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)


def translate_blueprint_to_phenotype(self):
"""
        Translates the being's blueprint into its phenotype.
        This implementation defines a sample blueprint structure and
        translates it into the three phenotype traits.
        """
# Sample blueprint structure (example)
# This structure is a placeholder and can be expanded upon.
# It includes different layers and types of data to simulate a complex blueprint.
# Ensure all expected keys are present, using defaults if necessary
blueprint_copy = {
    "raw_materials": self.blueprint.get("raw_materials", {}),
    "core_components": self.blueprint.get("core_components", {}),
    "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
    "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
    "mutation_parameters": self.blueprint.get("mutation_parameters", {})
}

sample_blueprint = {
    "raw_materials": {
        "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
        "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
        "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
    },
    "core_components": {
        "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
        "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
        "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
    },
    "strategic_foundation": {
        "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
        "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
    },
    "unifying_strategy": {
        "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
    },
    "mutation_parameters": {
        "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
    }
}


# Simple translation logic (example)
# This logic is illustrative and should be replaced with a more sophisticated model
# that reflects the intended relationships between blueprint and phenotype.
resource_efficiency = (sample_blueprint["raw_materials"]["material_a_ratio"] * 0.3 +
    sample_blueprint["raw_materials"]["material_b_ratio"] * 0.4 +
    sample_blueprint["raw_materials"]["processing_efficiency_factor"] * 0.3)

knowledge_exchange = (sample_blueprint["core_components"]["interconnection_density"] * 0.6 +
    sample_blueprint["strategic_foundation"]["cooperation_gene"] * 0.4)

structural_resilience = (sample_blueprint["core_components"]["component_x_strength"] * 0.5 +
    sample_blueprint["core_components"]["component_y_flexibility"] * 0.3 +
    sample_blueprint["unifying_strategy"]["optimization_parameter"] * 0.2)


# Apply tanh normalization as in previous code
phenotype = {
    "resource_efficiency": np.tanh(resource_efficiency),
    "knowledge_exchange": np.tanh(knowledge_exchange),
    "structural_resilience": np.tanh(structural_resilience)
}

return phenotype


def calculate_fitness(self, weights = (0.25, 0.35, 0.40)):
"""
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A tuple of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
# Ensure the phenotype has the expected keys based on the weights
keys = ["resource_efficiency", "knowledge_exchange", "structural_resilience"]
if not all(k in self.phenotype for k in keys):
raise ValueError("Phenotype is missing required keys for fitness calculation.")

self.fitness = (weights[0] * self.phenotype["resource_efficiency"] +
    weights[1] * self.phenotype["knowledge_exchange"] +
    weights[2] * self.phenotype["structural_resilience"])
return self.fitness

def mutate(self, population_average_fitness = None, population_trait_variance = None, population_trait_averages = None):
"""
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
        """
current_mutation_rate = self.base_mutation_rate
mutation_strength = 0.05 # Base strength of mutation

# 1. Fitness-dependent mutation: Increase mutation for less fit individuals
if population_average_fitness is not None and self.fitness < population_average_fitness:
current_mutation_rate *= 1.5 # Increase rate by 50% for less fit

# Define a helper function for mutating a value
def _apply_mutation(value, rate, strength):
if random.random() < rate:
# Ensure value is a number before adding random noise
if isinstance(value, (int, float)):
return np.clip(value + np.random.normal(0, strength), 0, 1)
else :
# Handle non-numeric types if necessary, or skip mutation
return value
return value

# 2. Apply mutation to blueprint components with adaptive/targeted strategies
for layer, components in self.blueprint.items():
if layer == "mutation_parameters": # Skip mutation parameters themselves
continue

for component, value in components.items():
effective_mutation_rate = current_mutation_rate
effective_mutation_strength = mutation_strength

# 3. Variance-based mutation: Increase mutation for traits with low population variance
# This requires mapping blueprint components to phenotype traits, which is complex.
# For this example, we'll apply variance-based mutation to all blueprint components
# if overall population variance is low (a simplification).
# A more sophisticated approach would involve understanding which blueprint components
# influence which phenotype traits.
if population_trait_variance is not None:
# Check if any trait has low variance (example threshold 0.01)
if any(v < 0.01 for v in population_trait_variance.values()):
effective_mutation_rate *= 1.2 # Increase rate if population variance is low

# 4. Targeted mutation based on population trait averages
# This also requires mapping blueprint components to phenotype traits.
# We'll apply targeted mutation based on overall low average phenotype scores (simplification).
if population_trait_averages is not None:
# Check if any trait has a low average (example threshold 0.5)
if any(avg < 0.5 for avg in population_trait_averages.values()):
effective_mutation_rate *= 1.3 # Increase rate if population averages are low

# Apply the mutation to the component value
self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

# 5. Update the phenotype after mutation
self.phenotype = self.translate_blueprint_to_phenotype()

@staticmethod
def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing') -> 'SimulatedBeing':
"""
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
offspring_blueprint = {}

# Iterate through each layer of the blueprint
for layer in parent1.blueprint.keys():
offspring_blueprint[layer] = {}
# Iterate through each component within the layer
for component in parent1.blueprint[layer].keys():
# Randomly select the value from either parent
if random.random() < 0.5:
offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
else :
offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

# Create a new SimulatedBeing instance with the offspring blueprint
offspring = SimulatedBeing(offspring_blueprint)
return offspring

# Example Usage (assuming you have two SimulatedBeing instances, parent1 and parent2)
# For demonstration, let's create two sample beings:
parent1_blueprint = {
    "raw_materials": {
        "material_a_ratio": 0.7, "material_b_ratio": 0.3, "processing_efficiency_factor": 0.9
    },
    "core_components": {
        "component_x_strength": 0.8, "component_y_flexibility": 0.6, "interconnection_density": 0.75
    },
    "strategic_foundation": {
        "adaptability_gene": 0.5, "cooperation_gene": 0.8
    },
    "unifying_strategy": {
        "optimization_parameter": 0.95
    },
    "mutation_parameters": {
        "base_rate": 0.01
    }
}

parent2_blueprint = {
    "raw_materials": {
        "material_a_ratio": 0.2, "material_b_ratio": 0.8, "processing_efficiency_factor": 0.4
    },
    "core_components": {
        "component_x_strength": 0.3, "component_y_flexibility": 0.9, "interconnection_density": 0.25
    },
    "strategic_foundation": {
        "adaptability_gene": 0.9, "cooperation_gene": 0.1
    },
    "unifying_strategy": {
        "optimization_parameter": 0.05
    },
    "mutation_parameters": {
        "base_rate": 0.02
    }
}

parent1 = SimulatedBeing(parent1_blueprint)
parent2 = SimulatedBeing(parent2_blueprint)

offspring = SimulatedBeing.uniform_crossover(parent1, parent2)

print("Parent 1 Blueprint:", parent1.blueprint)
print("Parent 2 Blueprint:", parent2.blueprint)
print("Offspring Blueprint:", offspring.blueprint)
print("Offspring Phenotype:", offspring.phenotype)

## Update the evolutionary loop

### Subtask:
Modify the existing `evolve_one_generation` function ( or create a new simulation class) to use the `SimulatedBeing` class and incorporate the new mutation and recombination methods.


**Reasoning**:
I need to integrate the `SimulatedBeing` class and the new mutation and recombination methods into the evolutionary loop. This involves modifying the `evolve_one_generation` function to work with a population of `SimulatedBeing` objects, calculate fitness, select parents, perform crossover, and apply mutation. I will also calculate the required population statistics (average fitness, trait variance, and trait averages) within this function before applying mutation. Finally, I will update the main simulation loop to use the modified `evolve_one_generation`.



import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
import os
import pandas as pd
from typing import Dict, List, Any

@dataclass
class Phenotype:
"""Represents the observable characteristics of a simulated being."""
resource_efficiency: float = 0.0
knowledge_exchange: float = 0.0
structural_resilience: float = 0.0

class PhenotypeTranslator:
"""Translates a blueprint dictionary into a Phenotype object based on defined rules."""

def __init__(self, blueprint_to_phenotype_rules: dict):
"""
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

def translate(self, blueprint: dict) -> Phenotype:
"""
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
phenotype_values = {}

# Apply translation rules to calculate phenotype trait values
for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
calculated_value = 0.0
for rule in rules:
component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
calculated_value += component_value * rule['weight']

# Apply tanh normalization
phenotype_values[phenotype_trait] = np.tanh(calculated_value)

# Create and return a Phenotype object
return Phenotype(
    resource_efficiency = phenotype_values.get("resource_efficiency", 0.0),
    knowledge_exchange = phenotype_values.get("knowledge_exchange", 0.0),
    structural_resilience = phenotype_values.get("structural_resilience", 0.0)
)

@dataclass
class SimulatedBeing:
"""Represents a single simulated being with its blueprint, phenotype, and fitness."""
blueprint: Dict[str, Any]
phenotype_translator: PhenotypeTranslator
phenotype: Phenotype = field(init = False)
fitness: float = field(init = False)
# Add a base mutation rate to the being's attributes
base_mutation_rate: float = field(init = False)


def __post_init__(self):
"""Translates the full blueprint into a phenotype upon creation."""
# Ensure all expected keys are present in the blueprint, using defaults if necessary
# This part can be more sophisticated depending on the desired blueprint structure and defaults
blueprint_copy = {
    "raw_materials": self.blueprint.get("raw_materials", {}),
    "core_components": self.blueprint.get("core_components", {}),
    "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
    "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
    "mutation_parameters": self.blueprint.get("mutation_parameters", {})
}

# Fill in missing blueprint components with random values for consistency
# This ensures the translator always receives a complete blueprint structure
complete_blueprint = {
    "raw_materials": {
        "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
        "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
        "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
    },
    "core_components": {
        "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
        "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
        "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
    },
    "strategic_foundation": {
        "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
        "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
    },
    "unifying_strategy": {
        "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
    },
    "mutation_parameters": {
        "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
    }
}
self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing
self.base_mutation_rate = self.blueprint["mutation_parameters"]["base_rate"]

self.phenotype = self.translate_blueprint_to_phenotype()


def calculate_fitness(self, weights: dict[str, float]) -> float:
"""
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
# Ensure the phenotype has the expected keys based on the weights
keys = list(weights.keys())
if not all(hasattr(self.phenotype, k) for k in keys):
raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {
    keys
}, Has: {
    self.phenotype.__dict__.keys()}")


self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
return self.fitness

def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None, mutation_std: float = 0.05):
"""
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
            mutation_std: The standard deviation of the random noise added during mutation.
        """
current_mutation_rate = self.base_mutation_rate
# mutation_strength = 0.05 # Base strength of mutation - now passed as mutation_std


# 1. Fitness-dependent mutation: Increase mutation for less fit individuals
if population_average_fitness is not None and self.fitness < population_average_fitness:
current_mutation_rate *= 1.5 # Increase rate by 50% for less fit

# Define a helper function for mutating a value
def _apply_mutation(value, rate, strength):
if random.random() < rate:
# Ensure value is a number before adding random noise
if isinstance(value, (int, float)):
return np.clip(value + np.random.normal(0, strength), 0, 1)
else :
# Handle non-numeric types if necessary, or skip mutation
return value
return value

# 2. Apply mutation to blueprint components with adaptive/targeted strategies
for layer, components in self.blueprint.items():
if layer == "mutation_parameters": # Skip mutation parameters themselves
continue

for component, value in components.items():
effective_mutation_rate = current_mutation_rate
effective_mutation_strength = mutation_std # Use passed mutation_std

# 3. Variance-based mutation: Increase mutation for traits with low population variance
# This requires mapping blueprint components to phenotype traits, which is complex.
# For this example, we'll apply variance-based mutation to all blueprint components
# if overall population variance is low (a simplification).
# A more sophisticated approach would involve understanding which blueprint components
# influence which phenotype traits.
if population_trait_variance is not None:
# Check if any trait has low variance (example threshold 0.01)
if any(v < 0.01 for v in population_trait_variance.values()):
effective_mutation_rate *= 1.2 # Increase rate if population variance is low

# 4. Targeted mutation based on population trait averages
# This also requires mapping blueprint components to phenotype traits.
# We'll apply targeted mutation based on overall low average phenotype scores (simplification).
if population_trait_averages is not None:
# Check if any trait has a low average (example threshold 0.5)
if any(avg < 0.5 for avg in population_trait_averages.values()):
effective_mutation_rate *= 1.3 # Increase rate if population averages are low

# Apply the mutation to the component value
self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

# 5. Update the phenotype after mutation
self.phenotype = self.translate_blueprint_to_phenotype()

@staticmethod
def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
"""
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
offspring_blueprint = {}

# Iterate through each layer of the blueprint
for layer in parent1.blueprint.keys():
offspring_blueprint[layer] = {}
# Iterate through each component within the layer
for component in parent1.blueprint[layer].keys():
# Randomly select the value from either parent
if random.random() < 0.5:
offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
else :
offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

# Create a new SimulatedBeing instance with the offspring blueprint and the translator
offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
return offspring

# Define sample translation rules (This should match the blueprint structure)
sample_translation_rules = {
    "resource_efficiency": [{
        "layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3
    },
        {
            "layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4
        },
        {
            "layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3
        },
    ],
    "knowledge_exchange": [{
        "layer": "core_components", "component": "interconnection_density", "weight": 0.6
    },
        {
            "layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4
        },
    ],
    "structural_resilience": [{
        "layer": "core_components", "component": "component_x_strength", "weight": 0.5
    },
        {
            "layer": "core_components", "component": "component_y_flexibility", "weight": 0.3
        },
        {
            "layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2
        },
    ]
}


# Helper function for dynamic fitness (Placeholder - replace with actual logic)
def dynamic_fitness(phenotype: Phenotype, generation: int, env_factor: float) -> float:
"""
    Calculates dynamic fitness based on phenotype, generation, and environmental factor.

    Args:
        phenotype: The Phenotype object of the being.
        generation: The current generation number.
        env_factor: A factor representing environmental pressure.

    Returns:
        The calculated dynamic fitness score.
    """
# Example dynamic fitness: favor resource efficiency early, then structural resilience
if generation < 25:
weights = {
    "resource_efficiency": 0.4 + 0.1 * env_factor,
    "knowledge_exchange": 0.4 - 0.05 * env_factor,
    "structural_resilience": 0.2 + 0.05 * env_factor
} else :
weights = {
    "resource_efficiency": 0.2 - 0.05 * env_factor,
    "knowledge_exchange": 0.2 + 0.05 * env_factor,
    "structural_resilience": 0.6 + 0.1 * env_factor
}

# Ensure weights sum to 1 (optional, depending on desired fitness scaling)
# total_weight = sum(weights.values())
# weights = {k: v / total_weight for k, v in weights.items()}


# Calculate fitness based on phenotype attributes and weights
fitness_score = (weights["resource_efficiency"] * phenotype.resource_efficiency +
    weights["knowledge_exchange"] * phenotype.knowledge_exchange +
    weights["structural_resilience"] * phenotype.structural_resilience)

return fitness_score


# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp = 0.1):
"""
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
# Calculate fitness for each being if not already done
fitnesses = np.array([being.fitness for being in population], dtype = float)

# Numeric stable softmax
scaled = fitnesses / float(temp)
e = np.exp(scaled - np.max(scaled))
probs = e / np.sum(e)
return probs

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
"""
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
if not population:
return {
    "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
}

phenotype_keys = list(population[0].phenotype.__dict__.keys())

generation_stats = {
    "generation": generation,
    "trait_means": {},
    "trait_stds": {}
}

for trait_name in phenotype_keys:
trait_values = [getattr(being.phenotype, trait_name) for being in population]
generation_stats['trait_means'][trait_name] = np.mean(trait_values)
generation_stats['trait_stds'][trait_name] = np.std(trait_values)

generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

return generation_stats


# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
"""
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
# Convert Phenotype object to a dictionary for plotting
phenotype_dict = phenotype_data.__dict__
labels = list(phenotype_dict.keys())
values = list(phenotype_dict.values())

num_vars = len(labels)
# Compute angle each trait goes to in the plot
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

# The plot is a circle, so we need to "complete the loop"
values = values + values[:1]
angles = angles + angles[:1]

fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

# Plot data and fill area
ax.fill(angles, values, color = 'red', alpha = 0.25)
ax.plot(angles, values, color = 'red', linewidth = 2)

# Set the grid and labels
ax.set_yticklabels([]) # Hide radial ticks
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Set title
ax.set_title(title, size = 16, color = 'blue', y = 1.1)

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok = True)

# Save the plot
plt.savefig(os.path.join(save_dir, f" {
    filename
}.png"))
plt.close(fig) # Close the figure to free up memory

def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
"""
    Finds and returns the individual with the highest fitness in the population.
    """
if not population:
return None
# Ensure fitness is calculated before finding the fittest
# for being in population:
#     being.calculate_fitness(...) # Need weights here - will be handled in evolve_population

return max(population, key = lambda being: being.fitness)


# --- The Refined Evolutionary Loop ---
def evolve_population(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_rate: float, mutation_std: float, env_factor: float, phenotype_translator: PhenotypeTranslator) -> List[SimulatedBeing]:
"""
    Simulates one generation of evolution.

    Args:
        population (List[SimulatedBeing]): The current population of beings.
        pop_size (int): The target size of the new population.
        generation (int): The current generation number, used for dynamic fitness.
        selection_temp (float): The temperature for softmax selection. Lower values
                                 lead to more "greedy" selection.
        mutation_rate (float): The base probability of mutation for a gene.
        mutation_std (float): The standard deviation of the random noise added during mutation.
        env_factor (float): A factor representing environmental pressure on certain traits.
        phenotype_translator: An instance of PhenotypeTranslator.


    Returns:
        List[SimulatedBeing]: The new population of beings after one generation of evolution.
    """
if not population:
return []

# 1. Evaluate Fitness for all beings in the population using dynamic fitness
# Define dynamic fitness weights based on the generation number and env_factor
# This logic is now within the dynamic_fitness helper function

for being in population:
being.fitness = dynamic_fitness(being.phenotype, generation, env_factor)

# Get all fitness values to calculate selection probabilities
fitnesses = [b.fitness for b in population]

# Softmax for selection probability. Lower temp leads to more greedy selection.
scaled_fitness = np.array(fitnesses) / float(selection_temp)
# Add a small value to handle cases where all scaled_fitness are the same (can cause issues with exp)
# Or use a more robust softmax implementation if necessary
e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
probs = e / np.sum(e)

# Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


# 2. Select Parents based on fitness probabilities
# Select pop_size * 2 indices as we need two parents per offspring for crossover
parent_indices = np.random.choice(
    range(len(population)),
    size = pop_size, # Select enough parents to create a full population
    p = probs,
    replace = True # Allows a highly-fit parent to be selected multiple times
)

# 3. Create the New Generation through Crossover & Mutation
new_population = []
# We'll create offspring in pairs for crossover, so we need pop_size // 2 pairs
# Ensure pop_size is even or handle the last offspring separately if odd
num_offspring_pairs = pop_size // 2
remaining_offspring = pop_size % 2 # For odd pop_size

# Select enough parents for crossover pairs
crossover_parent_indices = np.random.choice(
    range(len(population)),
    size = num_offspring_pairs * 2,
    p = probs,
    replace = True
)


# Calculate population statistics *before* mutation for adaptive mutation
phenotype_keys = list(population[0].phenotype.__dict__.keys())
population_average_fitness = np.mean(fitnesses)
population_trait_averages = {
    key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
}
population_trait_variance = {
    key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
}
current_pop_stats = {
    "average_fitness": population_average_fitness,
    "trait_averages": population_trait_averages,
    "trait_variance": population_trait_variance
}


for i in range(0, len(crossover_parent_indices), 2):
parent1_index = crossover_parent_indices[i]
parent2_index = crossover_parent_indices[i+1]

parent1 = population[parent1_index]
parent2 = population[parent2_index]

# Perform crossover to create offspring
# Pass the phenotype_translator to the crossover method
offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

# Calculate fitness for the newly created offspring using the current generation's weights
# This ensures self.fitness is available for fitness-dependent mutation
offspring.calculate_fitness(weights = dynamic_fitness(offspring.phenotype, generation, env_factor))


# Apply mutation to the offspring, passing population statistics and mutation_std
offspring.mutate(
    population_average_fitness = current_pop_stats["average_fitness"],
    population_trait_variance = current_pop_stats["trait_variance"],
    population_trait_averages = current_pop_stats["trait_averages"],
    mutation_std = mutation_std # Pass mutation_std
)

new_population.append(offspring)

# If pop_size is odd, select one more parent and create one more offspring (mutation only)
if remaining_offspring > 0:
single_parent_index = np.random.choice(range(len(population)), p = probs)
single_parent = population[single_parent_index]
# Create offspring by cloning and mutating (no crossover)
# This assumes blueprint can be copied and then mutated
single_offspring_blueprint = single_parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints
single_offspring = SimulatedBeing(single_offspring_blueprint, phenotype_translator)

# Calculate fitness for the single offspring
single_offspring.calculate_fitness(weights = dynamic_fitness(single_offspring.phenotype, generation, env_factor))

single_offspring.mutate(
    population_average_fitness = current_pop_stats["average_fitness"],
    population_trait_variance = current_pop_stats["trait_variance"],
    population_trait_averages = current_pop_stats["trait_averages"],
    mutation_std = mutation_std
)
new_population.append(single_offspring)


return new_population


# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
"""
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
if translator is None:
raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

population = []
for _ in range(size):
# Create a sample blueprint for initial population with random values
blueprint = {
    "raw_materials": {
        "material_a_ratio": random.random(),
        "material_b_ratio": random.random(),
        "processing_efficiency_factor": random.random(),
    },
    "core_components": {
        "component_x_strength": random.random(),
        "component_y_flexibility": random.random(),
        "interconnection_density": random.random(),
    },
    "strategic_foundation": {
        "adaptability_gene": random.random(),
        "cooperation_gene": random.random(),
    },
    "unifying_strategy": {
        "optimization_parameter": random.random(),
    },
    "mutation_parameters": {
        "base_rate": 0.01 # Initial base mutation rate
    }
}
# Create a SimulatedBeing instance and pass the translator
population.append(SimulatedBeing(blueprint, translator))
return population

# --- Run the Simulation ---
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Simulation parameters for evolve_population
selection_temp = 0.1
mutation_rate = 0.01 # Base mutation rate (can be overridden by individual being's blueprint)
mutation_std = 0.05 # Standard deviation for mutation noise
env_factor = 0.5 # Example environmental factor


# Create the PhenotypeTranslator instance once
translator = PhenotypeTranslator(sample_translation_rules)

# Generate the initial population using the updated function
population = generate_initial_population(size = pop_size, translator = translator)


# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []


for gen in range(num_generations):
# Evolve the population using the updated function with dynamic weights and the translator
# The dynamic fitness calculation is now handled within evolve_population by calling dynamic_fitness
population = evolve_population(
    population,
    pop_size = pop_size,
    generation = gen,
    selection_temp = selection_temp,
    mutation_rate = mutation_rate, # This is a base rate, individual rates can vary
    mutation_std = mutation_std,
    env_factor = env_factor,
    phenotype_translator = translator # Pass the translator
)

# Calculate and store the average phenotype for the current generation
phenotype_keys = list(population[0].phenotype.__dict__.keys())
avg_phenotype = {
    key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
}
avg_phenotypes_history.append(avg_phenotype)

# Track and store population statistics (fitness is already calculated in evolve_population)
gen_stats = track_population_statistics(population, gen)
population_statistics_history.append(gen_stats)

# Visualize the fittest individual and sampled individuals
fittest_individual = get_fittest_individual(population)
if fittest_individual:
create_radar_plot(fittest_individual.phenotype,
f"Fittest Phenotype - Gen {
    gen
}",
f"fittest_phenotype_gen_ {
    gen
}")

if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
sampled_individual = random.choice(population)
create_radar_plot(sampled_individual.phenotype,
f"Sampled Phenotype - Gen {
    gen
}",
f"sampled_phenotype_gen_ {
    gen
}")


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label = "Resource Efficiency", color = "green")
ax.plot(gens, knowledge_ex, label = "Knowledge Exchange", color = "blue")
ax.plot(gens, structural_res, label = "Structural Resilience", color = "red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()


# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label = "Resource Efficiency Std Dev", color = "green", linestyle = '--')
ax_std.plot(gens, knowledge_ex_std, label = "Knowledge Exchange Std Dev", color = "blue", linestyle = '--')
ax_std.plot(gens, structural_res_std, label = "Structural Resilience Std Dev", color = "red", linestyle = '--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")

## Implement dynamic fitness calculation

### Subtask:
Implement dynamic fitness calculation


**Reasoning**:
Modify the `evolve_one_generation` function to accept dynamic fitness weights and update the fitness calculation within the loop.



import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt

class SimulatedBeing:
"""Represents a single simulated being with its blueprint, phenotype, and fitness."""

def __init__(self, blueprint):
"""
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
        """
self.blueprint = blueprint
self.phenotype = self.translate_blueprint_to_phenotype()
self.fitness = 0.0 # Initialize fitness
# Add a base mutation rate to the being's attributes
self.base_mutation_rate = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)


def translate_blueprint_to_phenotype(self):
"""
        Translates the being's blueprint into its phenotype.
        This implementation defines a sample blueprint structure and
        translates it into the three phenotype traits.
        """
# Sample blueprint structure (example)
# This structure is a placeholder and can be expanded upon.
# It includes different layers and types of data to simulate a complex blueprint.
# Ensure all expected keys are present, using defaults if necessary
blueprint_copy = {
    "raw_materials": self.blueprint.get("raw_materials", {}),
    "core_components": self.blueprint.get("core_components", {}),
    "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
    "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
    "mutation_parameters": self.blueprint.get("mutation_parameters", {})
}

sample_blueprint = {
    "raw_materials": {
        "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
        "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
        "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
    },
    "core_components": {
        "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
        "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
        "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
    },
    "strategic_foundation": {
        "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
        "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
    },
    "unifying_strategy": {
        "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
    },
    "mutation_parameters": {
        "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
    }
}


# Simple translation logic (example)
# This logic is illustrative and should be replaced with a more sophisticated model
# that reflects the intended relationships between blueprint and phenotype.
resource_efficiency = (sample_blueprint["raw_materials"]["material_a_ratio"] * 0.3 +
    sample_blueprint["raw_materials"]["material_b_ratio"] * 0.4 +
    sample_blueprint["raw_materials"]["processing_efficiency_factor"] * 0.3)

knowledge_exchange = (sample_blueprint["core_components"]["interconnection_density"] * 0.6 +
    sample_blueprint["strategic_foundation"]["cooperation_gene"] * 0.4)

structural_resilience = (sample_blueprint["core_components"]["component_x_strength"] * 0.5 +
    sample_blueprint["core_components"]["component_y_flexibility"] * 0.3 +
    sample_blueprint["unifying_strategy"]["optimization_parameter"] * 0.2)


# Apply tanh normalization as in previous code
phenotype = {
    "resource_efficiency": np.tanh(resource_efficiency),
    "knowledge_exchange": np.tanh(knowledge_exchange),
    "structural_resilience": np.tanh(structural_resilience)
}

return phenotype


def calculate_fitness(self, weights = (0.25, 0.35, 0.40)):
"""
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
# Ensure the phenotype has the expected keys based on the weights
keys = list(weights.keys())
if not all(k in self.phenotype for k in keys):
raise ValueError("Phenotype is missing required keys for fitness calculation.")

self.fitness = sum(weights[key] * self.phenotype[key] for key in keys)
return self.fitness

def mutate(self, population_average_fitness = None, population_trait_variance = None, population_trait_averages = None):
"""
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
        """
current_mutation_rate = self.base_mutation_rate
mutation_strength = 0.05 # Base strength of mutation

# 1. Fitness-dependent mutation: Increase mutation for less fit individuals
if population_average_fitness is not None and self.fitness < population_average_fitness:
current_mutation_rate *= 1.5 # Increase rate by 50% for less fit

# Define a helper function for mutating a value
def _apply_mutation(value, rate, strength):
if random.random() < rate:
# Ensure value is a number before adding random noise
if isinstance(value, (int, float)):
return np.clip(value + np.random.normal(0, strength), 0, 1)
else :
# Handle non-numeric types if necessary, or skip mutation
return value
return value

# 2. Apply mutation to blueprint components with adaptive/targeted strategies
for layer, components in self.blueprint.items():
if layer == "mutation_parameters": # Skip mutation parameters themselves
continue

for component, value in components.items():
effective_mutation_rate = current_mutation_rate
effective_mutation_strength = mutation_strength

# 3. Variance-based mutation: Increase mutation for traits with low population variance
# This requires mapping blueprint components to phenotype traits, which is complex.
# For this example, we'll apply variance-based mutation to all blueprint components
# if overall population variance is low (a simplification).
# A more sophisticated approach would involve understanding which blueprint components
# influence which phenotype traits.
if population_trait_variance is not None:
# Check if any trait has low variance (example threshold 0.01)
if any(v < 0.01 for v in population_trait_variance.values()):
effective_mutation_rate *= 1.2 # Increase rate if population variance is low

# 4. Targeted mutation based on population trait averages
# This also requires mapping blueprint components to phenotype traits.
# We'll apply targeted mutation based on overall low average phenotype scores (simplification).
if population_trait_averages is not None:
# Check if any trait has a low average (example threshold 0.5)
if any(avg < 0.5 for avg in population_trait_averages.values()):
effective_mutation_rate *= 1.3 # Increase rate if population averages are low

# Apply the mutation to the component value
self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

# 5. Update the phenotype after mutation
self.phenotype = self.translate_blueprint_to_phenotype()

@staticmethod
def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing') -> 'SimulatedBeing':
"""
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
offspring_blueprint = {}

# Iterate through each layer of the blueprint
for layer in parent1.blueprint.keys():
offspring_blueprint[layer] = {}
# Iterate through each component within the layer
for component in parent1.blueprint[layer].keys():
# Randomly select the value from either parent
if random.random() < 0.5:
offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
else :
offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

# Create a new SimulatedBeing instance with the offspring blueprint
offspring = SimulatedBeing(offspring_blueprint)
return offspring

# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp = 0.1):
"""
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
# Calculate fitness for each being if not already done
fitnesses = np.array([being.fitness for being in population], dtype = float)

# Numeric stable softmax
scaled = fitnesses / float(temp)
e = np.exp(scaled - np.max(scaled))
probs = e / np.sum(e)
return probs

def evolve_one_generation(population: list[SimulatedBeing], pop_size: int, fitness_weights: dict, selection_temp: float = 0.1) -> list[SimulatedBeing]:
"""
    Evolves the population by one generation using selection, crossover, and mutation.

    Args:
        population: The current list of SimulatedBeing instances.
        pop_size: The desired size of the next generation.
        fitness_weights: A dictionary of weights for phenotype traits for fitness calculation.
        selection_temp: The temperature parameter for softmax selection.

    Returns:
        A new list of SimulatedBeing instances representing the next generation.
    """
if not population:
return []

# 1. Calculate Fitness for the current population with dynamic weights
for being in population:
being.calculate_fitness(weights = fitness_weights)

# 2. Calculate Population Statistics for mutation
phenotype_keys = list(population[0].phenotype.keys()) # Get keys from a sample being
population_average_fitness = np.mean([being.fitness for being in population])
population_trait_averages = {
    key: np.mean([being.phenotype[key] for being in population]) for key in phenotype_keys
}
population_trait_variance = {
    key: np.var([being.phenotype[key] for being in population]) for key in phenotype_keys
}


# 3. Calculate Selection Probabilities
selection_probs = selection_probabilities(population, temp = selection_temp)

# Handle potential issues with probabilities (e.g., if all fitnesses are the same)
if np.sum(selection_probs) == 0:
selection_probs = np.ones(len(population)) / len(population) # Assign equal probability


# 4. Select Parents (indices)
# We need to select pairs of parents for crossover.
# Select pop_size * 2 indices as we need two parents per offspring
parent_indices = np.random.choice(
    range(len(population)),
    size = pop_size * 2, # Select enough parents for pop_size offspring
    p = selection_probs,
    replace = True # Allows for one being to be selected multiple times
)

# 5. Create New Generation
new_population = []
for i in range(0, len(parent_indices), 2):
parent1_index = parent_indices[i]
parent2_index = parent_indices[i+1]

parent1 = population[parent1_index]
parent2 = population[parent2_index]

# Perform crossover to create offspring
offspring = SimulatedBeing.uniform_crossover(parent1, parent2)

# Apply mutation to the offspring, passing population statistics
offspring.mutate(
    population_average_fitness = population_average_fitness,
    population_trait_variance = population_trait_variance,
    population_trait_averages = population_trait_averages
)

new_population.append(offspring)

return new_population

# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size = 100):
population = []
for _ in range(size):
# Create a sample blueprint for initial population
blueprint = {
    "raw_materials": {
        "material_a_ratio": random.random(),
        "material_b_ratio": random.random(),
        "processing_efficiency_factor": random.random(),
    },
    "core_components": {
        "component_x_strength": random.random(),
        "component_y_flexibility": random.random(),
        "interconnection_density": random.random(),
    },
    "strategic_foundation": {
        "adaptability_gene": random.random(),
        "cooperation_gene": random.random(),
    },
    "unifying_strategy": {
        "optimization_parameter": random.random(),
    },
    "mutation_parameters": {
        "base_rate": 0.01 # Initial base mutation rate
    }
}
population.append(SimulatedBeing(blueprint))
return population


population = generate_initial_population(size = 100)
num_generations = 50 # Increased generations to see dynamic fitness effects
pop_size = 100 # Keep population size constant

# Store the average phenotypes for plotting
avg_phenotypes_history = []

for gen in range(num_generations):
# Define dynamic fitness weights based on the generation number
if gen < num_generations / 2:
# In the first half, prioritize resource efficiency and knowledge exchange
current_fitness_weights = {
    "resource_efficiency": 0.4,
    "knowledge_exchange": 0.4,
    "structural_resilience": 0.2
} else :
# In the second half, prioritize structural resilience
current_fitness_weights = {
    "resource_efficiency": 0.2,
    "knowledge_exchange": 0.2,
    "structural_resilience": 0.6
}

# Calculate and store the average phenotype for the current generation
phenotype_keys = list(population[0].phenotype.keys())
avg_phenotype = {
    key: np.mean([being.phenotype[key] for being in population]) for key in phenotype_keys
}
avg_phenotypes_history.append(avg_phenotype)

# Evolve the population using the updated function with dynamic weights
population = evolve_one_generation(population, pop_size = pop_size, fitness_weights = current_fitness_weights)

# Plot the evolution of average phenotypes
fig, ax = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label = "Resource Efficiency", color = "green")
ax.plot(gens, knowledge_ex, label = "Knowledge Exchange", color = "blue")
ax.plot(gens, structural_res, label = "Structural Resilience", color = "red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

print("Simulation complete. The plot shows how the population's average traits have evolved under dynamic fitness conditions.")

**Reasoning**:
The previous code successfully implemented dynamic fitness. Now, I will implement tracking and analyzing population statistics over generations.



import pandas as pd

def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
"""
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
if not population:
return {
    "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
}

phenotype_keys = list(population[0].phenotype.keys())

generation_stats = {
    "generation": generation,
    "trait_means": {},
    "trait_stds": {}
}

for trait_name in phenotype_keys:
trait_values = [being.phenotype[trait_name] for being in population]
generation_stats['trait_means'][trait_name] = np.mean(trait_values)
generation_stats['trait_stds'][trait_name] = np.std(trait_values)

generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

return generation_stats

# --- Main Simulation Loop (Modified to track statistics) ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size = 100):
population = []
for _ in range(size):
# Create a sample blueprint for initial population
blueprint = {
    "raw_materials": {
        "material_a_ratio": random.random(),
        "material_b_ratio": random.random(),
        "processing_efficiency_factor": random.random(),
    },
    "core_components": {
        "component_x_strength": random.random(),
        "component_y_flexibility": random.random(),
        "interconnection_density": random.random(),
    },
    "strategic_foundation": {
        "adaptability_gene": random.random(),
        "cooperation_gene": random.random(),
    },
    "unifying_strategy": {
        "optimization_parameter": random.random(),
    },
    "mutation_parameters": {
        "base_rate": 0.01 # Initial base mutation rate
    }
}
population.append(SimulatedBeing(blueprint))
return population

population = generate_initial_population(size = 100)
num_generations = 50
pop_size = 100 # Keep population size constant

# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []


for gen in range(num_generations):
# Define dynamic fitness weights based on the generation number
if gen < num_generations / 2:
# In the first half, prioritize resource efficiency and knowledge exchange
current_fitness_weights = {
    "resource_efficiency": 0.4,
    "knowledge_exchange": 0.4,
    "structural_resilience": 0.2
} else :
# In the second half, prioritize structural resilience
current_fitness_weights = {
    "resource_efficiency": 0.2,
    "knowledge_exchange": 0.2,
    "structural_resilience": 0.6
}

# Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
phenotype_keys = list(population[0].phenotype.keys())
avg_phenotype = {
    key: np.mean([being.phenotype[key] for being in population]) for key in phenotype_keys
}
avg_phenotypes_history.append(avg_phenotype)

# Track and store population statistics
gen_stats = track_population_statistics(population, gen)
population_statistics_history.append(gen_stats)


# Evolve the population using the updated function with dynamic weights
population = evolve_one_generation(population, pop_size = pop_size, fitness_weights = current_fitness_weights)


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label = "Resource Efficiency", color = "green")
ax.plot(gens, knowledge_ex, label = "Knowledge Exchange", color = "blue")
ax.plot(gens, structural_res, label = "Structural Resilience", color = "red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()


# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label = "Resource Efficiency Std Dev", color = "green", linestyle = '--')
ax_std.plot(gens, knowledge_ex_std, label = "Knowledge Exchange Std Dev", color = "blue", linestyle = '--')
ax_std.plot(gens, structural_res_std, label = "Structural Resilience Std Dev", color = "red", linestyle = '--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Population statistics have been tracked and analyzed.")

**Reasoning**:
Implement visualization of individual phenotypes using radar plots, focusing on the fittest individual and a few sampled individuals from different generations.



import matplotlib.pyplot as plt
import numpy as np
import os

def create_radar_plot(phenotype_data: dict, title: str, filename: str, save_dir: str = "dna_evolution_output"):
"""
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A dictionary containing phenotype trait names and values.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
labels = list(phenotype_data.keys())
values = list(phenotype_data.values())

num_vars = len(labels)
# Compute angle each trait goes to in the plot
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

# The plot is a circle, so we need to "complete the loop"
values = values + values[:1]
angles = angles + angles[:1]

fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

# Plot data and fill area
ax.fill(angles, values, color = 'red', alpha = 0.25)
ax.plot(angles, values, color = 'red', linewidth = 2)

# Set the grid and labels
ax.set_yticklabels([]) # Hide radial ticks
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Set title
ax.set_title(title, size = 16, color = 'blue', y = 1.1)

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok = True)

# Save the plot
plt.savefig(os.path.join(save_dir, f" {
    filename
}.png"))
plt.close(fig) # Close the figure to free up memory

def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
"""
    Finds and returns the individual with the highest fitness in the population.
    """
if not population:
return None
return max(population, key = lambda being: being.fitness)


# --- Main Simulation Loop (Modified to include individual visualization) ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size = 100):
population = []
for _ in range(size):
# Create a sample blueprint for initial population
blueprint = {
    "raw_materials": {
        "material_a_ratio": random.random(),
        "material_b_ratio": random.random(),
        "processing_efficiency_factor": random.random(),
    },
    "core_components": {
        "component_x_strength": random.random(),
        "component_y_flexibility": random.random(),
        "interconnection_density": random.random(),
    },
    "strategic_foundation": {
        "adaptability_gene": random.random(),
        "cooperation_gene": random.random(),
    },
    "unifying_strategy": {
        "optimization_parameter": random.random(),
    },
    "mutation_parameters": {
        "base_rate": 0.01 # Initial base mutation rate
    }
}
population.append(SimulatedBeing(blueprint))
return population

population = generate_initial_population(size = 100)
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []


for gen in range(num_generations):
# Define dynamic fitness weights based on the generation number
if gen < num_generations / 2:
# In the first half, prioritize resource efficiency and knowledge exchange
current_fitness_weights = {
    "resource_efficiency": 0.4,
    "knowledge_exchange": 0.4,
    "structural_resilience": 0.2
} else :
# In the second half, prioritize structural resilience
current_fitness_weights = {
    "resource_efficiency": 0.2,
    "knowledge_exchange": 0.2,
    "structural_resilience": 0.6
}

# Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
phenotype_keys = list(population[0].phenotype.keys())
avg_phenotype = {
    key: np.mean([being.phenotype[key] for being in population]) for key in phenotype_keys
}
avg_phenotypes_history.append(avg_phenotype)

# Track and store population statistics
gen_stats = track_population_statistics(population, gen)
population_statistics_history.append(gen_stats)

# Visualize the fittest individual and sampled individuals
fittest_individual = get_fittest_individual(population)
if fittest_individual:
create_radar_plot(fittest_individual.phenotype,
f"Fittest Phenotype - Gen {
    gen
}",
f"fittest_phenotype_gen_ {
    gen
}")

if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
sampled_individual = random.choice(population)
create_radar_plot(sampled_individual.phenotype,
f"Sampled Phenotype - Gen {
    gen
}",
f"sampled_phenotype_gen_ {
    gen
}")


# Evolve the population using the updated function with dynamic weights
population = evolve_one_generation(population, pop_size = pop_size, fitness_weights = current_fitness_weights)


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label = "Resource Efficiency", color = "green")
ax.plot(gens, knowledge_ex, label = "Knowledge Exchange", color = "blue")
ax.plot(gens, structural_res, label = "Structural Resilience", color = "red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()


# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label = "Resource Efficiency Std Dev", color = "green", linestyle = '--')
ax_std.plot(gens, knowledge_ex_std, label = "Knowledge Exchange Std Dev", color = "blue", linestyle = '--')
ax_std.plot(gens, structural_res_std, label = "Structural Resilience Std Dev", color = "red", linestyle = '--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")

## Summary:

### Data Analysis Key Findings

* The `SimulatedBeing` class was successfully created to encapsulate the blueprint, phenotype, and fitness of individual beings.
* A phenotype translation mechanism was implemented within the `SimulatedBeing` class, mapping a sample blueprint structure to the three phenotype traits (resource efficiency, knowledge exchange, and structural resilience).
* An advanced mutation method was added to the `SimulatedBeing` class, incorporating fitness-dependent, variance-based, and targeted mutation strategies based on population statistics.
* A `uniform_crossover` static method was implemented in the `SimulatedBeing` class to perform genetic recombination between two parent beings.
* The main evolutionary loop (`evolve_one_generation` function) was updated to utilize the `SimulatedBeing` class and integrate the new mutation and crossover methods.
* Dynamic fitness calculation was implemented by allowing time-varying fitness weights to be used in the `calculate_fitness` method and the `evolve_one_generation` function.
* A function `track_population_statistics` was created to record key population metrics (mean and standard deviation of traits, average fitness) per generation, and this data was stored in a pandas DataFrame.
* Visualization capabilities were added to generate radar plots of individual phenotypes, specifically for the fittest individual in each generation and sampled individuals at intervals.
* The simulation successfully ran with the integrated features, and the generated plots demonstrate the evolution of average phenotype traits and their standard deviations over generations under dynamic fitness conditions.

### Insights or Next Steps

* The implementation provides a robust framework for simulating evolutionary dynamics with complex genetic structures, adaptive mutation, and dynamic fitness landscapes. Future work could focus on exploring different blueprint structures, translation rules, mutation strategies, and fitness functions to model diverse evolutionary scenarios.
* Further analysis of the collected population statistics can reveal deeper insights into the evolutionary process, such as the impact of mutation rates and selection pressure on genetic diversity and adaptation speed. The individual phenotype visualizations can help understand the emergence of specific traits in the population over time.


# Task
Integrate the provided refactored `SimulatedBeing` class, implement the `PhenotypeTranslator` and `uniform_crossover` methods, update the population generation and evolutionary loop to use these new components, add population statistics tracking and individual visualization, and finally run and analyze the simulation.

## Integrate the refactored `simulatedbeing` class

### Subtask:
Update the notebook with the complete definition of the refactored `SimulatedBeing` class, including the `Phenotype` dataclass and the `__post_init__` and `mutate` methods.


**Reasoning**:
The subtask is to provide the complete definition of the refactored `SimulatedBeing` class, including the `Phenotype` dataclass and the specified methods. I will define the `Phenotype` dataclass and the `SimulatedBeing` class with the `__init__`, `__post_init__`, `translate_blueprint_to_phenotype`, `calculate_fitness`, and `mutate` methods as described in the instructions.



import random
import numpy as np
from dataclasses import dataclass, field

@dataclass
class Phenotype:
"""Represents the observable characteristics of a simulated being."""
resource_efficiency: float = 0.0
knowledge_exchange: float = 0.0
structural_resilience: float = 0.0

class SimulatedBeing:
"""Represents a single simulated being with its blueprint, phenotype, and fitness."""

def __init__(self, blueprint: dict):
"""
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
        """
self.blueprint = blueprint
self.phenotype: Phenotype = None # Will be set in __post_init__
self.fitness: float = 0.0 # Initialize fitness
# Add a base mutation rate to the being's attributes
self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)

def __post_init__(self):
"""Calls translate_blueprint_to_phenotype after initialization."""
self.phenotype = self.translate_blueprint_to_phenotype()

def translate_blueprint_to_phenotype(self) -> Phenotype:
"""
        Translates the being's blueprint into its phenotype.
        This implementation defines a sample blueprint structure and
        translates it into the three phenotype traits.
        """
# Sample blueprint structure (example)
# This structure is a placeholder and can be expanded upon.
# It includes different layers and types of data to simulate a complex blueprint.
# Ensure all expected keys are present, using defaults if necessary
blueprint_copy = {
    "raw_materials": self.blueprint.get("raw_materials", {}),
    "core_components": self.blueprint.get("core_components", {}),
    "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
    "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
    "mutation_parameters": self.blueprint.get("mutation_parameters", {})
}

# Fill in missing blueprint components with random values for consistency
sample_blueprint = {
    "raw_materials": {
        "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
        "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
        "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
    },
    "core_components": {
        "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
        "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
        "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
    },
    "strategic_foundation": {
        "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
        "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
    },
    "unifying_strategy": {
        "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
    },
    "mutation_parameters": {
        "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
    }
}
self.blueprint = sample_blueprint # Update the being's blueprint with defaults if any were missing

# Simple translation logic (example)
# This logic is illustrative and should be replaced with a more sophisticated model
# that reflects the intended relationships between blueprint and phenotype.
resource_efficiency = (sample_blueprint["raw_materials"]["material_a_ratio"] * 0.3 +
    sample_blueprint["raw_materials"]["material_b_ratio"] * 0.4 +
    sample_blueprint["raw_materials"]["processing_efficiency_factor"] * 0.3)

knowledge_exchange = (sample_blueprint["core_components"]["interconnection_density"] * 0.6 +
    sample_blueprint["strategic_foundation"]["cooperation_gene"] * 0.4)

structural_resilience = (sample_blueprint["core_components"]["component_x_strength"] * 0.5 +
    sample_blueprint["core_components"]["component_y_flexibility"] * 0.3 +
    sample_blueprint["unifying_strategy"]["optimization_parameter"] * 0.2)


# Apply tanh normalization as in previous code
phenotype = Phenotype(
    resource_efficiency = np.tanh(resource_efficiency),
    knowledge_exchange = np.tanh(knowledge_exchange),
    structural_resilience = np.tanh(structural_resilience)
)

return phenotype


def calculate_fitness(self, weights: dict[str, float]) -> float:
"""
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
# Ensure the phenotype has the expected keys based on the weights
keys = list(weights.keys())
if not all(hasattr(self.phenotype, k) for k in keys):
raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {
    keys
}, Has: {
    self.phenotype.__dict__.keys()}")


self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
return self.fitness

def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None):
"""
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
        """
current_mutation_rate = self.base_mutation_rate
mutation_strength = 0.05 # Base strength of mutation

# 1. Fitness-dependent mutation: Increase mutation for less fit individuals
if population_average_fitness is not None and self.fitness < population_average_fitness:
current_mutation_rate *= 1.5 # Increase rate by 50% for less fit

# Define a helper function for mutating a value
def _apply_mutation(value, rate, strength):
if random.random() < rate:
# Ensure value is a number before adding random noise
if isinstance(value, (int, float)):
return np.clip(value + np.random.normal(0, strength), 0, 1)
else :
# Handle non-numeric types if necessary, or skip mutation
return value
return value

# 2. Apply mutation to blueprint components with adaptive/targeted strategies
for layer, components in self.blueprint.items():
if layer == "mutation_parameters": # Skip mutation parameters themselves
continue

for component, value in components.items():
effective_mutation_rate = current_mutation_rate
effective_mutation_strength = mutation_strength

# 3. Variance-based mutation: Increase mutation for traits with low population variance
# This requires mapping blueprint components to phenotype traits, which is complex.
# For this example, we'll apply variance-based mutation to all blueprint components
# if overall population variance is low (a simplification).
# A more sophisticated approach would involve understanding which blueprint components
# influence which phenotype traits.
if population_trait_variance is not None:
# Check if any trait has low variance (example threshold 0.01)
if any(v < 0.01 for v in population_trait_variance.values()):
effective_mutation_rate *= 1.2 # Increase rate if population variance is low

# 4. Targeted mutation based on population trait averages
# This also requires mapping blueprint components to phenotype traits.
# We'll apply targeted mutation based on overall low average phenotype scores (simplification).
if population_trait_averages is not None:
# Check if any trait has a low average (example threshold 0.5)
if any(avg < 0.5 for avg in population_trait_averages.values()):
effective_mutation_rate *= 1.3 # Increase rate if population averages are low

# Apply the mutation to the component value
self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

# 5. Update the phenotype after mutation
self.phenotype = self.translate_blueprint_to_phenotype()

@staticmethod
def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing') -> 'SimulatedBeing':
"""
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
offspring_blueprint = {}

# Iterate through each layer of the blueprint
for layer in parent1.blueprint.keys():
offspring_blueprint[layer] = {}
# Iterate through each component within the layer
for component in parent1.blueprint[layer].keys():
# Randomly select the value from either parent
if random.random() < 0.5:
offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
else :
offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

# Create a new SimulatedBeing instance with the offspring blueprint
offspring = SimulatedBeing(offspring_blueprint)
return offspring


## Define the `phenotypetranslator` class

### Subtask:
Define the `PhenotypeTranslator` class, which is used by the `SimulatedBeing` class to translate the blueprint into a phenotype.


**Reasoning**:
Define the `PhenotypeTranslator` class with an `__init__` method to store the translation rules and a `translate` method to apply these rules to a blueprint and return a `Phenotype` object with normalized values.



from dataclasses import dataclass
import numpy as np
import random

@dataclass
class Phenotype:
"""Represents the observable characteristics of a simulated being."""
resource_efficiency: float = 0.0
knowledge_exchange: float = 0.0
structural_resilience: float = 0.0

class PhenotypeTranslator:
"""Translates a blueprint dictionary into a Phenotype object based on defined rules."""

def __init__(self, blueprint_to_phenotype_rules: dict):
"""
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

def translate(self, blueprint: dict) -> Phenotype:
"""
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
phenotype_values = {}

# Apply translation rules to calculate phenotype trait values
for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
calculated_value = 0.0
for rule in rules:
component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
calculated_value += component_value * rule['weight']

# Apply tanh normalization
phenotype_values[phenotype_trait] = np.tanh(calculated_value)

# Create and return a Phenotype object
return Phenotype(
    resource_efficiency = phenotype_values.get("resource_efficiency", 0.0),
    knowledge_exchange = phenotype_values.get("knowledge_exchange", 0.0),
    structural_resilience = phenotype_values.get("structural_resilience", 0.0)
)

# Define sample translation rules (This should match the blueprint structure)
sample_translation_rules = {
    "resource_efficiency": [{
        "layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3
    },
        {
            "layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4
        },
        {
            "layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3
        },
    ],
    "knowledge_exchange": [{
        "layer": "core_components", "component": "interconnection_density", "weight": 0.6
    },
        {
            "layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4
        },
    ],
    "structural_resilience": [{
        "layer": "core_components", "component": "component_x_strength", "weight": 0.5
    },
        {
            "layer": "core_components", "component": "component_y_flexibility", "weight": 0.3
        },
        {
            "layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2
        },
    ]
}

# Example Usage:
# Create a sample blueprint
sample_blueprint_data = {
    "raw_materials": {
        "material_a_ratio": random.random(),
        "material_b_ratio": random.random(),
        "processing_efficiency_factor": random.random(),
    },
    "core_components": {
        "component_x_strength": random.random(),
        "component_y_flexibility": random.random(),
        "interconnection_density": random.random(),
    },
    "strategic_foundation": {
        "adaptability_gene": random.random(),
        "cooperation_gene": random.random(),
    },
    "unifying_strategy": {
        "optimization_parameter": random.random(),
    },
    "mutation_parameters": {
        "base_rate": 0.01
    }
}

# Create a translator instance
translator = PhenotypeTranslator(sample_translation_rules)

# Translate the blueprint
sample_phenotype = translator.translate(sample_blueprint_data)

print("Sample Blueprint:", sample_blueprint_data)
print("Translated Phenotype:", sample_phenotype)

**Reasoning**:
Update the `SimulatedBeing` class to use the `PhenotypeTranslator` for blueprint to phenotype translation.



from dataclasses import dataclass, field
import numpy as np
import random

@dataclass
class Phenotype:
"""Represents the observable characteristics of a simulated being."""
resource_efficiency: float = 0.0
knowledge_exchange: float = 0.0
structural_resilience: float = 0.0

class PhenotypeTranslator:
"""Translates a blueprint dictionary into a Phenotype object based on defined rules."""

def __init__(self, blueprint_to_phenotype_rules: dict):
"""
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

def translate(self, blueprint: dict) -> Phenotype:
"""
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
phenotype_values = {}

# Apply translation rules to calculate phenotype trait values
for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
calculated_value = 0.0
for rule in rules:
component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
calculated_value += component_value * rule['weight']

# Apply tanh normalization
phenotype_values[phenotype_trait] = np.tanh(calculated_value)

# Create and return a Phenotype object
return Phenotype(
    resource_efficiency = phenotype_values.get("resource_efficiency", 0.0),
    knowledge_exchange = phenotype_values.get("knowledge_exchange", 0.0),
    structural_resilience = phenotype_values.get("structural_resilience", 0.0)
)

class SimulatedBeing:
"""Represents a single simulated being with its blueprint, phenotype, and fitness."""

def __init__(self, blueprint: dict, phenotype_translator: PhenotypeTranslator):
"""
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
self.blueprint = blueprint
self.phenotype_translator = phenotype_translator
self.phenotype: Phenotype = self.translate_blueprint_to_phenotype()
self.fitness: float = 0.0 # Initialize fitness
# Add a base mutation rate to the being's attributes
self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)

def translate_blueprint_to_phenotype(self) -> Phenotype:
"""
        Translates the being's blueprint into its phenotype using the provided translator.
        """
# Ensure all expected keys are present in the blueprint, using defaults if necessary
# This part can be more sophisticated depending on the desired blueprint structure and defaults
blueprint_copy = {
    "raw_materials": self.blueprint.get("raw_materials", {}),
    "core_components": self.blueprint.get("core_components", {}),
    "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
    "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
    "mutation_parameters": self.blueprint.get("mutation_parameters", {})
}

# Fill in missing blueprint components with random values for consistency
# This ensures the translator always receives a complete blueprint structure
complete_blueprint = {
    "raw_materials": {
        "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
        "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
        "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
    },
    "core_components": {
        "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
        "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
        "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
    },
    "strategic_foundation": {
        "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
        "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
    },
    "unifying_strategy": {
        "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
    },
    "mutation_parameters": {
        "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
    }
}
self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


return self.phenotype_translator.translate(self.blueprint)


def calculate_fitness(self, weights: dict[str, float]) -> float:
"""
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
# Ensure the phenotype has the expected keys based on the weights
keys = list(weights.keys())
if not all(hasattr(self.phenotype, k) for k in keys):
raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {
    keys
}, Has: {
    self.phenotype.__dict__.keys()}")


self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
return self.fitness

def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None):
"""
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
        """
current_mutation_rate = self.base_mutation_rate
mutation_strength = 0.05 # Base strength of mutation

# 1. Fitness-dependent mutation: Increase mutation for less fit individuals
if population_average_fitness is not None and self.fitness < population_average_fitness:
current_mutation_rate *= 1.5 # Increase rate by 50% for less fit

# Define a helper function for mutating a value
def _apply_mutation(value, rate, strength):
if random.random() < rate:
# Ensure value is a number before adding random noise
if isinstance(value, (int, float)):
return np.clip(value + np.random.normal(0, strength), 0, 1)
else :
# Handle non-numeric types if necessary, or skip mutation
return value
return value

# 2. Apply mutation to blueprint components with adaptive/targeted strategies
for layer, components in self.blueprint.items():
if layer == "mutation_parameters": # Skip mutation parameters themselves
continue

for component, value in components.items():
effective_mutation_rate = current_mutation_rate
effective_mutation_strength = mutation_strength

# 3. Variance-based mutation: Increase mutation for traits with low population variance
# This requires mapping blueprint components to phenotype traits, which is complex.
# For this example, we'll apply variance-based mutation to all blueprint components
# if overall population variance is low (a simplification).
# A more sophisticated approach would involve understanding which blueprint components
# influence which phenotype traits.
if population_trait_variance is not None:
# Check if any trait has low variance (example threshold 0.01)
if any(v < 0.01 for v in population_trait_variance.values()):
effective_mutation_rate *= 1.2 # Increase rate if population variance is low

# 4. Targeted mutation based on population trait averages
# This also requires mapping blueprint components to phenotype traits.
# We'll apply targeted mutation based on overall low average phenotype scores (simplification).
if population_trait_averages is not None:
# Check if any trait has a low average (example threshold 0.5)
if any(avg < 0.5 for avg in population_trait_averages.values()):
effective_mutation_rate *= 1.3 # Increase rate if population averages are low

# Apply the mutation to the component value
self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

# 5. Update the phenotype after mutation
self.phenotype = self.translate_blueprint_to_phenotype()

@staticmethod
def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
"""
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
offspring_blueprint = {}

# Iterate through each layer of the blueprint
for layer in parent1.blueprint.keys():
offspring_blueprint[layer] = {}
# Iterate through each component within the layer
for component in parent1.blueprint[layer].keys():
# Randomly select the value from either parent
if random.random() < 0.5:
offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
else :
offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

# Create a new SimulatedBeing instance with the offspring blueprint and the translator
offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
return offspring

# Define sample translation rules (This should match the blueprint structure)
sample_translation_rules = {
    "resource_efficiency": [{
        "layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3
    },
        {
            "layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4
        },
        {
            "layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3
        },
    ],
    "knowledge_exchange": [{
        "layer": "core_components", "component": "interconnection_density", "weight": 0.6
    },
        {
            "layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4
        },
    ],
    "structural_resilience": [{
        "layer": "core_components", "component": "component_x_strength", "weight": 0.5
    },
        {
            "layer": "core_components", "component": "component_y_flexibility", "weight": 0.3
        },
        {
            "layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2
        },
    ]
}

# Example Usage:
# Create a translator instance
translator = PhenotypeTranslator(sample_translation_rules)

# Create a sample blueprint
sample_blueprint_data = {
    "raw_materials": {
        "material_a_ratio": random.random(),
        "material_b_ratio": random.random(),
        "processing_efficiency_factor": random.random(),
    },
    "core_components": {
        "component_x_strength": random.random(),
        "component_y_flexibility": random.random(),
        "interconnection_density": random.random(),
    },
    "strategic_foundation": {
        "adaptability_gene": random.random(),
        "cooperation_gene": random.random(),
    },
    "unifying_strategy": {
        "optimization_parameter": random.random(),
    },
    "mutation_parameters": {
        "base_rate": 0.01
    }
}

# Create a SimulatedBeing instance using the translator
sample_being = SimulatedBeing(sample_blueprint_data, translator)

print("Sample Blueprint:", sample_being.blueprint)
print("Translated Phenotype:", sample_being.phenotype)

## Update population generation

### Subtask:
Modify the `generate_initial_population` function to create instances of the new `SimulatedBeing` class with appropriate initial blueprints.


**Reasoning**:
Update the `generate_initial_population` function to create a list of `SimulatedBeing` objects using the `PhenotypeTranslator`.



import random
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import os
import pandas as pd

@dataclass
class Phenotype:
"""Represents the observable characteristics of a simulated being."""
resource_efficiency: float = 0.0
knowledge_exchange: float = 0.0
structural_resilience: float = 0.0

class PhenotypeTranslator:
"""Translates a blueprint dictionary into a Phenotype object based on defined rules."""

def __init__(self, blueprint_to_phenotype_rules: dict):
"""
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

def translate(self, blueprint: dict) -> Phenotype:
"""
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
phenotype_values = {}

# Apply translation rules to calculate phenotype trait values
for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
calculated_value = 0.0
for rule in rules:
component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
calculated_value += component_value * rule['weight']

# Apply tanh normalization
phenotype_values[phenotype_trait] = np.tanh(calculated_value)

# Create and return a Phenotype object
return Phenotype(
    resource_efficiency = phenotype_values.get("resource_efficiency", 0.0),
    knowledge_exchange = phenotype_values.get("knowledge_exchange", 0.0),
    structural_resilience = phenotype_values.get("structural_resilience", 0.0)
)

class SimulatedBeing:
"""Represents a single simulated being with its blueprint, phenotype, and fitness."""

def __init__(self, blueprint: dict, phenotype_translator: PhenotypeTranslator):
"""
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
self.blueprint = blueprint
self.phenotype_translator = phenotype_translator
self.phenotype: Phenotype = self.translate_blueprint_to_phenotype()
self.fitness: float = 0.0 # Initialize fitness
# Add a base mutation rate to the being's attributes
self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)

def translate_blueprint_to_phenotype(self) -> Phenotype:
"""
        Translates the being's blueprint into its phenotype using the provided translator.
        """
# Ensure all expected keys are present in the blueprint, using defaults if necessary
# This part can be more sophisticated depending on the desired blueprint structure and defaults
blueprint_copy = {
    "raw_materials": self.blueprint.get("raw_materials", {}),
    "core_components": self.blueprint.get("core_components", {}),
    "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
    "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
    "mutation_parameters": self.blueprint.get("mutation_parameters", {})
}

# Fill in missing blueprint components with random values for consistency
# This ensures the translator always receives a complete blueprint structure
complete_blueprint = {
    "raw_materials": {
        "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
        "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
        "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
    },
    "core_components": {
        "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
        "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
        "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
    },
    "strategic_foundation": {
        "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
        "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
    },
    "unifying_strategy": {
        "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
    },
    "mutation_parameters": {
        "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
    }
}
self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


return self.phenotype_translator.translate(self.blueprint)


def calculate_fitness(self, weights: dict[str, float]) -> float:
"""
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
# Ensure the phenotype has the expected keys based on the weights
keys = list(weights.keys())
if not all(hasattr(self.phenotype, k) for k in keys):
raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {
    keys
}, Has: {
    self.phenotype.__dict__.keys()}")


self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
return self.fitness

def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None):
"""
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
        """
current_mutation_rate = self.base_mutation_rate
mutation_strength = 0.05 # Base strength of mutation

# 1. Fitness-dependent mutation: Increase mutation for less fit individuals
if population_average_fitness is not None and self.fitness < population_average_fitness:
current_mutation_rate *= 1.5 # Increase rate by 50% for less fit

# Define a helper function for mutating a value
def _apply_mutation(value, rate, strength):
if random.random() < rate:
# Ensure value is a number before adding random noise
if isinstance(value, (int, float)):
return np.clip(value + np.random.normal(0, strength), 0, 1)
else :
# Handle non-numeric types if necessary, or skip mutation
return value
return value

# 2. Apply mutation to blueprint components with adaptive/targeted strategies
for layer, components in self.blueprint.items():
if layer == "mutation_parameters": # Skip mutation parameters themselves
continue

for component, value in components.items():
effective_mutation_rate = current_mutation_rate
effective_mutation_strength = mutation_strength

# 3. Variance-based mutation: Increase mutation for traits with low population variance
# This requires mapping blueprint components to phenotype traits, which is complex.
# For this example, we'll apply variance-based mutation to all blueprint components
# if overall population variance is low (a simplification).
# A more sophisticated approach would involve understanding which blueprint components
# influence which phenotype traits.
if population_trait_variance is not None:
# Check if any trait has low variance (example threshold 0.01)
if any(v < 0.01 for v in population_trait_variance.values()):
effective_mutation_rate *= 1.2 # Increase rate if population variance is low

# 4. Targeted mutation based on population trait averages
# This also requires mapping blueprint components to phenotype traits.
# We'll apply targeted mutation based on overall low average phenotype scores (simplification).
if population_trait_averages is not None:
# Check if any trait has a low average (example threshold 0.5)
if any(avg < 0.5 for avg in population_trait_averages.values()):
effective_mutation_rate *= 1.3 # Increase rate if population averages are low

# Apply the mutation to the component value
self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

# 5. Update the phenotype after mutation
self.phenotype = self.translate_blueprint_to_phenotype()

@staticmethod
def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
"""
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
offspring_blueprint = {}

# Iterate through each layer of the blueprint
for layer in parent1.blueprint.keys():
offspring_blueprint[layer] = {}
# Iterate through each component within the layer
for component in parent1.blueprint[layer].keys():
# Randomly select the value from either parent
if random.random() < 0.5:
offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
else :
offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

# Create a new SimulatedBeing instance with the offspring blueprint and the translator
offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
return offspring

# Define sample translation rules (This should match the blueprint structure)
sample_translation_rules = {
    "resource_efficiency": [{
        "layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3
    },
        {
            "layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4
        },
        {
            "layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3
        },
    ],
    "knowledge_exchange": [{
        "layer": "core_components", "component": "interconnection_density", "weight": 0.6
    },
        {
            "layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4
        },
    ],
    "structural_resilience": [{
        "layer": "core_components", "component": "component_x_strength", "weight": 0.5
    },
        {
            "layer": "core_components", "component": "component_y_flexibility", "weight": 0.3
        },
        {
            "layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2
        },
    ]
}


# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp = 0.1):
"""
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
# Calculate fitness for each being if not already done
fitnesses = np.array([being.fitness for being in population], dtype = float)

# Numeric stable softmax
scaled = fitnesses / float(temp)
e = np.exp(scaled - np.max(scaled))
probs = e / np.sum(e)
return probs

def evolve_one_generation(population: list[SimulatedBeing], pop_size: int, fitness_weights: dict, phenotype_translator: PhenotypeTranslator, selection_temp: float = 0.1) -> list[SimulatedBeing]:
"""
    Evolves the population by one generation using selection, crossover, and mutation.

    Args:
        population: The current list of SimulatedBeing instances.
        pop_size: The desired size of the next generation.
        fitness_weights: A dictionary of weights for phenotype traits for fitness calculation.
        phenotype_translator: An instance of PhenotypeTranslator.
        selection_temp: The temperature parameter for softmax selection.

    Returns:
        A new list of SimulatedBeing instances representing the next generation.
    """
if not population:
return []

# 1. Calculate Fitness for the current population with dynamic weights
for being in population:
being.calculate_fitness(weights = fitness_weights)

# 2. Calculate Population Statistics for mutation
phenotype_keys = list(population[0].phenotype.__dict__.keys()) # Get keys from a sample being's phenotype
population_average_fitness = np.mean([being.fitness for being in population])
population_trait_averages = {
    key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
}
population_trait_variance = {
    key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
}


# 3. Calculate Selection Probabilities
selection_probs = selection_probabilities(population, temp = selection_temp)

# Handle potential issues with probabilities (e.g., if all fitnesses are the same)
if np.sum(selection_probs) == 0:
selection_probs = np.ones(len(population)) / len(population) # Assign equal probability


# 4. Select Parents (indices)
# We need to select pairs of parents for crossover.
# Select pop_size * 2 indices as we need two parents per offspring
parent_indices = np.random.choice(
    range(len(population)),
    size = pop_size * 2, # Select enough parents for pop_size offspring
    p = selection_probs,
    replace = True # Allows for one being to be selected multiple times
)

# 5. Create New Generation
new_population = []
for i in range(0, len(parent_indices), 2):
parent1_index = parent_indices[i]
parent2_index = parent_indices[i+1]

parent1 = population[parent1_index]
parent2 = population[parent2_index]

# Perform crossover to create offspring, passing the translator
offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

# Apply mutation to the offspring, passing population statistics
offspring.mutate(
    population_average_fitness = population_average_fitness,
    population_trait_variance = population_trait_variance,
    population_trait_averages = population_trait_averages
)

new_population.append(offspring)

return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
"""
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
if not population:
return {
    "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
}

phenotype_keys = list(population[0].phenotype.__dict__.keys())

generation_stats = {
    "generation": generation,
    "trait_means": {},
    "trait_stds": {}
}

for trait_name in phenotype_keys:
trait_values = [getattr(being.phenotype, trait_name) for being in population]
generation_stats['trait_means'][trait_name] = np.mean(trait_values)
generation_stats['trait_stds'][trait_name] = np.std(trait_values)

generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

return generation_stats


# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
"""
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
# Convert Phenotype object to a dictionary for plotting
phenotype_dict = phenotype_data.__dict__
labels = list(phenotype_dict.keys())
values = list(phenotype_dict.values())

num_vars = len(labels)
# Compute angle each trait goes to in the plot
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

# The plot is a circle, so we need to "complete the loop"
values = values + values[:1]
angles = angles + angles[:1]

fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

# Plot data and fill area
ax.fill(angles, values, color = 'red', alpha = 0.25)
ax.plot(angles, values, color = 'red', linewidth = 2)

# Set the grid and labels
ax.set_yticklabels([]) # Hide radial ticks
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Set title
ax.set_title(title, size = 16, color = 'blue', y = 1.1)

# Ensure the save directory exists
os.makedirs(save_dir, exist_ok = True)

# Save the plot
plt.savefig(os.path.join(save_dir, f" {
    filename
}.png"))
plt.close(fig) # Close the figure to free up memory

def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
"""
    Finds and returns the individual with the highest fitness in the population.
    """
if not population:
return None
return max(population, key = lambda being: being.fitness)


# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
"""
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
if translator is None:
raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

population = []
for _ in range(size):
# Create a sample blueprint for initial population with random values
blueprint = {
    "raw_materials": {
        "material_a_ratio": random.random(),
        "material_b_ratio": random.random(),
        "processing_efficiency_factor": random.random(),
    },
    "core_components": {
        "component_x_strength": random.random(),
        "component_y_flexibility": random.random(),
        "interconnection_density": random.random(),
    },
    "strategic_foundation": {
        "adaptability_gene": random.random(),
        "cooperation_gene": random.random(),
    },
    "unifying_strategy": {
        "optimization_parameter": random.random(),
    },
    "mutation_parameters": {
        "base_rate": 0.01 # Initial base mutation rate
    }
}
# Create a SimulatedBeing instance and pass the translator
population.append(SimulatedBeing(blueprint, translator))
return population

# --- Run the Simulation ---
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Create the PhenotypeTranslator instance once
translator = PhenotypeTranslator(sample_translation_rules)

# Generate the initial population using the updated function
population = generate_initial_population(size = pop_size, translator = translator)


# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []


for gen in range(num_generations):
# Define dynamic fitness weights based on the generation number
if gen < num_generations / 2:
# In the first half, prioritize resource efficiency and knowledge exchange
current_fitness_weights = {
    "resource_efficiency": 0.4,
    "knowledge_exchange": 0.4,
    "structural_resilience": 0.2
} else :
# In the second half, prioritize structural resilience
current_fitness_weights = {
    "resource_efficiency": 0.2,
    "knowledge_exchange": 0.2,
    "structural_resilience": 0.6
}

# Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
phenotype_keys = list(population[0].phenotype.__dict__.keys())
avg_phenotype = {
    key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
}
avg_phenotypes_history.append(avg_phenotype)

# Track and store population statistics
gen_stats = track_population_statistics(population, gen)
population_statistics_history.append(gen_stats)

# Visualize the fittest individual and sampled individuals
fittest_individual = get_fittest_individual(population)
if fittest_individual:
create_radar_plot(fittest_individual.phenotype,
f"Fittest Phenotype - Gen {
    gen
}",
f"fittest_phenotype_gen_ {
    gen
}")

if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
sampled_individual = random.choice(population)
create_radar_plot(sampled_individual.phenotype,
f"Sampled Phenotype - Gen {
    gen
}",
f"sampled_phenotype_gen_ {
    gen
}")


# Evolve the population using the updated function with dynamic weights and the translator
population = evolve_one_generation(population, pop_size = pop_size, fitness_weights = current_fitness_weights, phenotype_translator = translator)


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label = "Resource Efficiency", color = "green")
ax.plot(gens, knowledge_ex, label = "Knowledge Exchange", color = "blue")
ax.plot(gens, structural_res, label = "Structural Resilience", color = "red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()


# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize = (8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label = "Resource Efficiency Std Dev", color = "green", linestyle = '--')
ax_std.plot(gens, knowledge_ex_std, label = "Knowledge Exchange Std Dev", color = "blue", linestyle = '--')
ax_std.plot(gens, structural_res_std, label = "Structural Resilience Std Dev", color = "red", linestyle = '--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")

## Summary:

### Data Analysis Key Findings

* The `PhenotypeTranslator` class was successfully defined to translate blueprint dictionaries into `Phenotype` objects based on defined rules, incorporating tanh normalization.
* The `SimulatedBeing` class was refactored to include a `PhenotypeTranslator` instance for blueprint-to-phenotype translation.
* The `generate_initial_population` function was updated to create `SimulatedBeing` instances, providing each with a random initial blueprint and the `PhenotypeTranslator` instance.
* The simulation loop was successfully modified to use the updated population generation and the `PhenotypeTranslator` instance during offspring creation via uniform crossover.
* Population statistics, including average phenotype traits, trait standard deviations, and average fitness, were tracked and stored across generations.
* Visualizations, such as plots of the evolution of average phenotype traits and standard deviations, and radar plots of fittest and sampled individuals, were successfully generated.

### Insights or Next Steps

* The successful integration of the `PhenotypeTranslator` and the refactored `SimulatedBeing` class provides a more modular and extensible framework for the simulation.
* Analyzing the generated statistics and plots will allow for a deeper understanding of how the dynamic fitness weights influence the evolutionary trajectory of the population's phenotype traits and genetic diversity.


# All the refinements have been integrated into the code. The SimulatedBeing class now operates with a full, nested blueprint. A dedicated PhenotypeTranslator class maps this blueprint to the final phenotype traits, and the mutation logic is now adaptive, targeting less-fit individuals and low-variance traits. The updated evolutionary loop also includes two-offspring crossover, a dynamic fitness function, and comprehensive statistical tracking. All plots are now automatically saved to the specified directory.
# The Refined Evolutionary Loop
# The evolve_population function is now the core of the simulation. It orchestrates the entire process for one generation, from fitness evaluation to creating the next generation. It also calculates and passes population statistics to the mutation logic.
# def evolve_population(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_rate: float, mutation_std: float, env_factor: float, stats: Dict):
# 1. Evaluate Fitness for all beings in the population
# for being in population:
# being.fitness = dynamic_fitness(being.phenotype, generation, env_factor)

# # Get all fitness values to calculate selection probabilities
# fitnesses = [b.fitness for b in population]

# # Softmax for selection probability. Lower temp leads to more greedy selection.
# scaled_fitness = np.array(fitnesses) / float(selection_temp)
# e = np.exp(scaled_fitness - np.max(scaled_fitness))
# probs = e / np.sum(e)

# # 2. Select Parents based on fitness probabilities
# parent_indices = np.random.choice(
# range(len(population)),
# size=pop_size, # Select enough parents to create a full population
# p=probs,
# replace=True # Allows a highly-fit parent to be selected multiple times
# )

# # 3. Create the New Generation through Crossover & Mutation
# new_population = []
# # Create offspring in pairs
# for i in range(0, pop_size, 2):
# parent1 = population[parent_indices[i]]
# parent2 = populatio

# Task
Summarize the implemented features and the insights gained from running the advanced simulation.

## Summary: data analysis key findings

### Subtask:
Summarize the key data analysis findings from the implemented evolutionary simulation.


## Summary:

### Data Analysis Key Findings

* The simulation was successfully implemented and run.
* The data needed for analysis is available in a dataframe named `stats_df`.

### Insights or Next Steps

* Proceed with analyzing the `stats_df` to extract specific findings from the evolutionary simulation.# Before calling a utility function that expects dictionaries
# Assuming 'population' is a list of SimulatedBeing objects
phenotype_dicts = [being.phenotype.__dict__ for being in population]

# Now, you can pass this list to the utility functions
normalized_phenotypes = percentile_normalize(phenotype_dicts)
selection_probs = selection_probabilities(normalized_phenotypes)

import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
from pyvis.network import Network
import os
import numpy as np
from scipy.stats import rankdata

# Layer classes
class JOT:
    def __init__(self, timing_bias=1.0, procedural_awareness=1.0):
        self.timing_bias = timing_bias
        self.procedural_awareness = procedural_awareness

class TLO:
    def __init__(self, objective_framework=1.0, cognitive_scaffolding=1.0):
        self.objective_framework = objective_framework
        self.cognitive_scaffolding = cognitive_scaffolding

class Layer770:
    def __init__(self, rotation_chance=0.3):
        self.state = "ON"
        if random.random() < rotation_chance:
            self.state = "OFF"

class OJT:
    def __init__(self, learning_speed=1.0, skill_acquisition_rate=1.0):
        self.learning_speed = learning_speed
        self.skill_acquisition_rate = skill_acquisition_rate

class GEF:
    def __init__(self, resource_abundance=1.0, competitive_pressure=1.0):
        self.resource_abundance = resource_abundance
        self.competitive_pressure = competitive_pressure

class EOU:
    def __init__(self, base_efficiency=1.0, knowledge_receptivity=1.0):
        self.base_efficiency = base_efficiency
        self.knowledge_receptivity = knowledge_receptivity

class RER:
    def __init__(self, reflex_speed=1.0, error_correction=1.0):
        self.reflex_speed = reflex_speed
        self.error_correction = error_correction

class TOL:
    def __init__(self, strategic_planning=1.0, decision_bias=1.0):
        self.strategic_planning = strategic_planning
        self.decision_bias = decision_bias

# Phenotype Calculation & Influence Mapping
def calculate_phenotype(layers):
    # Retrieve influence scores from each layer
    jot = layers['jot']
    tlo = layers['tlo']
    layer770 = layers['770']
    ojt = layers['ojt']
    gef = layers['gef']
    eou = layers['eou']
    rer = layers['rer']
    tol = layers['tol']

    # Synergy function to model non-linear interactions
    synergy_factor = (rer.reflex_speed * tol.strategic_planning) / 2.0
    if layer770.state == "OFF":
        synergy_factor *= 0.5  # 770 OFF state reduces synergy

    # Resource Efficiency (weighted sum with synergy)
    resource_efficiency = (
        (ojt.learning_speed * 0.2) +
        (eou.base_efficiency * 0.4) +
        (gef.resource_abundance * 0.2) +
        (tol.strategic_planning * 0.2)
    ) * synergy_factor

    # Knowledge Exchange (focused on JOT, TLO, OJT, and TOL)
    knowledge_exchange = (
        (jot.timing_bias * 0.1) +
        (tlo.objective_framework * 0.2) +
        (ojt.skill_acquisition_rate * 0.4) +
        (tol.decision_bias * 0.3)
    )

    # Structural Resilience (influenced by EOU and GEF, modulated by RER)
    structural_resilience = (
        (eou.base_efficiency * 0.5) +
        (gef.competitive_pressure * 0.3) +
        (rer.error_correction * 0.2)
    )

    return {
        "ResourceEfficiency": resource_efficiency,
        "KnowledgeExchange": knowledge_exchange,
        "StructuralResilience": structural_resilience
    }

# Utility functions (added to resolve NameError)
def percentile_normalize(population_phenotypes, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience")):
    """
    population_phenotypes: list of dict-like phenotypes (each has keys in keys)
    returns: list of dicts with percentile-normalized values [0..1]
    """
    if not population_phenotypes:
        return []

    # Build arrays for each phenotype key
    arrays = {
        k: np.array([p[k] for p in population_phenotypes], dtype = float) for k in keys
    }

    normalized = []
    num_beings = len(population_phenotypes)
    for i in range(num_beings):
        entry = {}
        for k in keys:
            ranks = rankdata(arrays[k], method = "average")
            # Convert to 0..1 percentile
            entry[k] = (ranks[i] - 1) / (num_beings - 1) if num_beings > 1 else 1.0
        normalized.append(entry)
    return normalized

def phenotype_fitness(p, weights = (0.25, 0.35, 0.40)):
    # p: dict with keys resource_efficiency, knowledge_exchange, structural_resilience
    # Ensure keys match the phenotype dictionary keys
    keys = ["ResourceEfficiency", "KnowledgeExchange", "StructuralResilience"]
    if not all(k in p for k in keys):
        raise ValueError(f"Phenotype dictionary is missing required keys for fitness calculation. Expected: {keys}, Has: {p.keys()}")

    return (weights[0] * p["ResourceEfficiency"] +
            weights[1] * p["KnowledgeExchange"] +
            weights[2] * p["StructuralResilience"])


def selection_probabilities(population_phenotypes, fitness_func = phenotype_fitness, temp = 0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    fitnesses = np.array([fitness_func(p) for p in population_phenotypes], dtype = float)
    # Numeric stable softmax
    scaled = fitnesses / float(temp)
    e = np.exp(scaled - np.max(scaled))
    probs = e / np.sum(e)
    return probs


# Step 2: Interactive Visualization
def create_interactive_map(influence_data):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    net.barnes_hut()

    # Define nodes and edges
    nodes = ["JOT", "TLO", "770", "OJT", "GEF", "EOU", "RER", "TOL"]
    phenotype_nodes = ["ResourceEfficiency", "KnowledgeExchange", "StructuralResilience"]

    # Add nodes
    for node in nodes:
        net.add_node(node, title=node)
    for p_node in phenotype_nodes:
        net.add_node(p_node, title=p_node, color='red')

    # Add edges with weights
    edges = [("JOT", "TLO"), ("TLO", "770"), ("770", "OJT"), ("OJT", "GEF"),
             ("GEF", "EOU"), ("EOU", "RER"), ("RER", "TOL")]

    for edge in edges:
        net.add_edge(edge[0], edge[1])

    # Add edges from TOL to phenotypes
    net.add_edge("TOL", "ResourceEfficiency")
    net.add_edge("TOL", "KnowledgeExchange")
    net.add_edge("TOL", "StructuralResilience")

    # Add edges from other layers to phenotypes based on calculate_phenotype logic
    # Resource Efficiency influences: OJT, EOU, GEF, TOL (through synergy)
    net.add_edge("OJT", "ResourceEfficiency", weight=0.2)
    net.add_edge("EOU", "ResourceEfficiency", weight=0.4)
    net.add_edge("GEF", "ResourceEfficiency", weight=0.2)
    # TOL influence on Resource Efficiency is through synergy, not a direct weighted sum here for simplicity in visualization

    # Knowledge Exchange influences: JOT, TLO, OJT, TOL
    net.add_edge("JOT", "KnowledgeExchange", weight=0.1)
    net.add_edge("TLO", "KnowledgeExchange", weight=0.2)
    net.add_edge("OJT", "KnowledgeExchange", weight=0.4)
    net.add_edge("TOL", "KnowledgeExchange", weight=0.3)

    # Structural Resilience influences: EOU, GEF, RER
    net.add_edge("EOU", "StructuralResilience", weight=0.5)
    net.add_edge("GEF", "StructuralResilience", weight=0.3)
    net.add_edge("RER", "StructuralResilience", weight=0.2)


    net.show("evolutionary_pipeline.html")

# Example usage
layers_instance = {
    'jot': JOT(), 'tlo': TLO(), '770': Layer770(), 'ojt': OJT(),
    'gef': GEF(), 'eou': EOU(), 'rer': RER(), 'tol': TOL()
}
phenotypes = calculate_phenotype(layers_instance)
create_interactive_map(phenotypes)

# Before calling a utility function that expects dictionaries
# Assuming 'population' is a list of SimulatedBeing objects
# phenotype_dicts = [being.phenotype.__dict__ for being in population] # This line is commented out as 'population' is not defined here

# Now, you can pass this list to the utility functions
# normalized_phenotypes = percentile_normalize(phenotype_dicts) # This line is commented out
# selection_probs = selection_probabilities(normalized_phenotypes) # This line is commented out



!pip install pyvis

import networkx as nx
import matplotlib.pyplot as plt
import random
import pandas as pd
from pyvis.network import Network
import os
import numpy as np
from scipy.stats import rankdata

# Layer classes
class JOT:
    def __init__(self, timing_bias=1.0, procedural_awareness=1.0):
        self.timing_bias = timing_bias
        self.procedural_awareness = procedural_awareness

class TLO:
    def __init__(self, objective_framework=1.0, cognitive_scaffolding=1.0):
        self.objective_framework = objective_framework
        self.cognitive_scaffolding = cognitive_scaffolding

class Layer770:
    def __init__(self, rotation_chance=0.3):
        self.state = "ON"
        if random.random() < rotation_chance:
            self.state = "OFF"

class OJT:
    def __init__(self, learning_speed=1.0, skill_acquisition_rate=1.0):
        self.learning_speed = learning_speed
        self.skill_acquisition_rate = skill_acquisition_rate

class GEF:
    def __init__(self, resource_abundance=1.0, competitive_pressure=1.0):
        self.resource_abundance = resource_abundance
        self.competitive_pressure = competitive_pressure

class EOU:
    def __init__(self, base_efficiency=1.0, knowledge_receptivity=1.0):
        self.base_efficiency = base_efficiency
        self.knowledge_receptivity = knowledge_receptivity

class RER:
    def __init__(self, reflex_speed=1.0, error_correction=1.0):
        self.reflex_speed = reflex_speed
        self.error_correction = error_correction

class TOL:
    def __init__(self, strategic_planning=1.0, decision_bias=1.0):
        self.strategic_planning = strategic_planning
        self.decision_bias = decision_bias

# Phenotype Calculation & Influence Mapping
def calculate_phenotype(layers):
    # Retrieve influence scores from each layer
    jot = layers['jot']
    tlo = layers['tlo']
    layer770 = layers['770']
    ojt = layers['ojt']
    gef = layers['gef']
    eou = layers['eou']
    rer = layers['rer']
    tol = layers['tol']

    # Synergy function to model non-linear interactions
    synergy_factor = (rer.reflex_speed * tol.strategic_planning) / 2.0
    if layer770.state == "OFF":
        synergy_factor *= 0.5  # 770 OFF state reduces synergy

    # Resource Efficiency (weighted sum with synergy)
    resource_efficiency = (
        (ojt.learning_speed * 0.2) +
        (eou.base_efficiency * 0.4) +
        (gef.resource_abundance * 0.2) +
        (tol.strategic_planning * 0.2)
    ) * synergy_factor

    # Knowledge Exchange (focused on JOT, TLO, OJT, and TOL)
    knowledge_exchange = (
        (jot.timing_bias * 0.1) +
        (tlo.objective_framework * 0.2) +
        (ojt.skill_acquisition_rate * 0.4) +
        (tol.decision_bias * 0.3)
    )

    # Structural Resilience (influenced by EOU and GEF, modulated by RER)
    structural_resilience = (
        (eou.base_efficiency * 0.5) +
        (gef.competitive_pressure * 0.3) +
        (rer.error_correction * 0.2)
    )

    return {
        "ResourceEfficiency": resource_efficiency,
        "KnowledgeExchange": knowledge_exchange,
        "StructuralResilience": structural_resilience
    }

# Utility functions (added to resolve NameError)
def percentile_normalize(population_phenotypes, keys = ("resource_efficiency", "knowledge_exchange", "structural_resilience")):
    """
    population_phenotypes: list of dict-like phenotypes (each has keys in keys)
    returns: list of dicts with percentile-normalized values [0..1]
    """
    if not population_phenotypes:
        return []

    # Build arrays for each phenotype key
    arrays = {
        k: np.array([p[k] for p in population_phenotypes], dtype = float) for k in keys
    }

    normalized = []
    num_beings = len(population_phenotypes)
    for i in range(num_beings):
        entry = {}
        for k in keys:
            ranks = rankdata(arrays[k], method = "average")
            # Convert to 0..1 percentile
            entry[k] = (ranks[i] - 1) / (num_beings - 1) if num_beings > 1 else 1.0
        normalized.append(entry)
    return normalized

def phenotype_fitness(p, weights = (0.25, 0.35, 0.40)):
    # p: dict with keys resource_efficiency, knowledge_exchange, structural_resilience
    # Ensure keys match the phenotype dictionary keys
    keys = ["ResourceEfficiency", "KnowledgeExchange", "StructuralResilience"]
    if not all(k in p for k in keys):
        raise ValueError(f"Phenotype dictionary is missing required keys for fitness calculation. Expected: {keys}, Has: {p.keys()}")

    return (weights[0] * p["ResourceEfficiency"] +
            weights[1] * p["KnowledgeExchange"] +
            weights[2] * p["StructuralResilience"])


def selection_probabilities(population_phenotypes, fitness_func = phenotype_fitness, temp = 0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    fitnesses = np.array([fitness_func(p) for p in population_phenotypes], dtype = float)
    # Numeric stable softmax
    scaled = fitnesses / float(temp)
    e = np.exp(scaled - np.max(scaled))
    probs = e / np.sum(e)
    return probs


# Step 2: Interactive Visualization
def create_interactive_map(influence_data):
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
    net.barnes_hut()

    # Define nodes and edges
    nodes = ["JOT", "TLO", "770", "OJT", "GEF", "EOU", "RER", "TOL"]
    phenotype_nodes = ["ResourceEfficiency", "KnowledgeExchange", "StructuralResilience"]

    # Add nodes
    for node in nodes:
        net.add_node(node, title=node)
    for p_node in phenotype_nodes:
        net.add_node(p_node, title=p_node, color='red')

    # Add edges with weights
    edges = [("JOT", "TLO"), ("TLO", "770"), ("770", "OJT"), ("OJT", "GEF"),
             ("GEF", "EOU"), ("EOU", "RER"), ("RER", "TOL")]

    for edge in edges:
        net.add_edge(edge[0], edge[1])

    # Add edges from TOL to phenotypes
    net.add_edge("TOL", "ResourceEfficiency")
    net.add_edge("TOL", "KnowledgeExchange")
    net.add_edge("TOL", "StructuralResilience")

    # Add edges from other layers to phenotypes based on calculate_phenotype logic
    # Resource Efficiency influences: OJT, EOU, GEF, TOL (through synergy)
    net.add_edge("OJT", "ResourceEfficiency", weight=0.2)
    net.add_edge("EOU", "ResourceEfficiency", weight=0.4)
    net.add_edge("GEF", "ResourceEfficiency", weight=0.2)
    # TOL influence on Resource Efficiency is through synergy, not a direct weighted sum here for simplicity in visualization

    # Knowledge Exchange influences: JOT, TLO, OJT, TOL
    net.add_edge("JOT", "KnowledgeExchange", weight=0.1)
    net.add_edge("TLO", "KnowledgeExchange", weight=0.2)
    net.add_edge("OJT", "KnowledgeExchange", weight=0.4)
    net.add_edge("TOL", "KnowledgeExchange", weight=0.3)

    # Structural Resilience influences: EOU, GEF, RER
    net.add_edge("EOU", "StructuralResilience", weight=0.5)
    net.add_edge("GEF", "StructuralResilience", weight=0.3)
    net.add_edge("RER", "StructuralResilience", weight=0.2)


    net.show("evolutionary_pipeline.html")

# Example usage
layers_instance = {
    'jot': JOT(), 'tlo': TLO(), '770': Layer770(), 'ojt': OJT(),
    'gef': GEF(), 'eou': EOU(), 'rer': RER(), 'tol': TOL()
}
phenotypes = calculate_phenotype(layers_instance)
create_interactive_map(phenotypes)

# Before calling a utility function that expects dictionaries
# Assuming 'population' is a list of SimulatedBeing objects
# phenotype_dicts = [being.phenotype.__dict__ for being in population] # This line is commented out as 'population' is not defined here

# Now, you can pass this list to the utility functions
# normalized_phenotypes = percentile_normalize(phenotype_dicts) # This line is commented out
# selection_probs = selection_probabilities(normalized_phenotypes) # This line is commented out

# Task
Integrate the `SimulatedBeing` class, implement advanced mutation and crossover, update the evolutionary loop, add dynamic fitness, statistics tracking, and visualization for individual beings into the provided code.

## Integrate the refactored `simulatedbeing` class

### Subtask:
Update the notebook with the complete definition of the refactored `SimulatedBeing` class, including the `Phenotype` dataclass and the `__post_init__` and `mutate` methods.


**Reasoning**:
Define the Phenotype dataclass and the SimulatedBeing class with the specified methods, including `__init__`, `__post_init__`, `translate_blueprint_to_phenotype`, `calculate_fitness`, and `mutate`.



from dataclasses import dataclass, field
import numpy as np
import random

@dataclass
class Phenotype:
    """Represents the observable characteristics of a simulated being."""
    resource_efficiency: float = 0.0
    knowledge_exchange: float = 0.0
    structural_resilience: float = 0.0

class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    def __init__(self, blueprint: dict):
        """
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
        """
        self.blueprint = blueprint
        self.phenotype: Phenotype = None # Will be set in __post_init__
        self.fitness: float = 0.0 # Initialize fitness
        # Add a base mutation rate to the being's attributes
        self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)
        self.__post_init__() # Call post_init explicitly as __post_init__ is for dataclasses


    def __post_init__(self):
        """Calls translate_blueprint_to_phenotype after initialization."""
        self.phenotype = self.translate_blueprint_to_phenotype()

    def translate_blueprint_to_phenotype(self) -> Phenotype:
        """
        Translates the being's blueprint into its phenotype.
        This implementation defines a sample blueprint structure and
        translates it into the three phenotype traits.
        """
        # Sample blueprint structure (example)
        # This structure is a placeholder and can be expanded upon.
        # It includes different layers and types of data to simulate a complex blueprint.
        # Ensure all expected keys are present, using defaults if necessary
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        sample_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = sample_blueprint # Update the being's blueprint with defaults if any were missing


        # Simple translation logic (example)
        # This logic is illustrative and should be replaced with a more sophisticated model
        # that reflects the intended relationships between blueprint and phenotype.
        resource_efficiency = (sample_blueprint["raw_materials"]["material_a_ratio"] * 0.3 +
                               sample_blueprint["raw_materials"]["material_b_ratio"] * 0.4 +
                               sample_blueprint["raw_materials"]["processing_efficiency_factor"] * 0.3)

        knowledge_exchange = (sample_blueprint["core_components"]["interconnection_density"] * 0.6 +
                              sample_blueprint["strategic_foundation"]["cooperation_gene"] * 0.4)

        structural_resilience = (sample_blueprint["core_components"]["component_x_strength"] * 0.5 +
                                 sample_blueprint["core_components"]["component_y_flexibility"] * 0.3 +
                                 sample_blueprint["unifying_strategy"]["optimization_parameter"] * 0.2)

        # Apply tanh normalization as in previous code
        phenotype = Phenotype(
            resource_efficiency=np.tanh(resource_efficiency),
            knowledge_exchange=np.tanh(knowledge_exchange),
            structural_resilience=np.tanh(structural_resilience)
        )

        return phenotype


    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
        """
        current_mutation_rate = self.base_mutation_rate
        mutation_strength = 0.05 # Base strength of mutation

        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value, rate, strength):
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_strength

                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (a simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages are low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing') -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint
        offspring = SimulatedBeing(offspring_blueprint)
        return offspring

**Reasoning**:
Define the PhenotypeTranslator class with the specified methods to handle blueprint to phenotype translation.



class PhenotypeTranslator:
    """Translates a blueprint dictionary into a Phenotype object based on defined rules."""

    def __init__(self, blueprint_to_phenotype_rules: dict):
        """
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
        self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

    def translate(self, blueprint: dict) -> Phenotype:
        """
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
        phenotype_values = {}

        # Apply translation rules to calculate phenotype trait values
        for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
            calculated_value = 0.0
            for rule in rules:
                component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
                calculated_value += component_value * rule['weight']

            # Apply tanh normalization
            phenotype_values[phenotype_trait] = np.tanh(calculated_value)

        # Create and return a Phenotype object
        return Phenotype(
            resource_efficiency=phenotype_values.get("resource_efficiency", 0.0),
            knowledge_exchange=phenotype_values.get("knowledge_exchange", 0.0),
            structural_resilience=phenotype_values.get("structural_resilience", 0.0)
        )

# Define sample translation rules (This should match the blueprint structure)
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

**Reasoning**:
Update the `SimulatedBeing` class to use the `PhenotypeTranslator` for blueprint to phenotype translation.



class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    def __init__(self, blueprint: dict, phenotype_translator: PhenotypeTranslator):
        """
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
        self.blueprint = blueprint
        self.phenotype_translator = phenotype_translator
        self.phenotype: Phenotype = self.translate_blueprint_to_phenotype()
        self.fitness: float = 0.0 # Initialize fitness
        # Add a base mutation rate to the being's attributes
        self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)

    def translate_blueprint_to_phenotype(self) -> Phenotype:
        """
        Translates the being's blueprint into its phenotype using the provided translator.
        """
        # Ensure all expected keys are present in the blueprint, using defaults if necessary
        # This part can be more sophisticated depending on the desired blueprint structure and defaults
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        # This ensures the translator always receives a complete blueprint structure
        complete_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


        return self.phenotype_translator.translate(self.blueprint)


    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
        """
        current_mutation_rate = self.base_mutation_rate
        mutation_strength = 0.05 # Base strength of mutation

        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value, rate, strength):
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_strength

                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (a simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages are low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint and the translator
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
        return offspring

**Reasoning**:
Modify the `generate_initial_population` function to create instances of the new `SimulatedBeing` class, providing each with a random initial blueprint and the `PhenotypeTranslator`.



# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp = 0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    # Calculate fitness for each being if not already done
    fitnesses = np.array([being.fitness for being in population], dtype = float)

    # Numeric stable softmax
    scaled = fitnesses / float(temp)
    e = np.exp(scaled - np.max(scaled))
    probs = e / np.sum(e)
    return probs

def evolve_one_generation(population: list[SimulatedBeing], pop_size: int, fitness_weights: dict, phenotype_translator: PhenotypeTranslator, selection_temp: float = 0.1) -> list[SimulatedBeing]:
    """
    Evolves the population by one generation using selection, crossover, and mutation.

    Args:
        population: The current list of SimulatedBeing instances.
        pop_size: The desired size of the next generation.
        fitness_weights: A dictionary of weights for phenotype traits for fitness calculation.
        phenotype_translator: An instance of PhenotypeTranslator.
        selection_temp: The temperature parameter for softmax selection.

    Returns:
        A new list of SimulatedBeing instances representing the next generation.
    """
    if not population:
        return []

    # 1. Calculate Fitness for the current population with dynamic weights
    for being in population:
        being.calculate_fitness(weights = fitness_weights)

    # 2. Calculate Population Statistics for mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys()) # Get keys from a sample being's phenotype
    population_average_fitness = np.mean([being.fitness for being in population])
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }


    # 3. Calculate Selection Probabilities
    selection_probs = selection_probabilities(population, temp = selection_temp)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same)
    if np.sum(selection_probs) == 0:
        selection_probs = np.ones(len(population)) / len(population) # Assign equal probability


    # 4. Select Parents (indices)
    # We need to select pairs of parents for crossover.
    # Select pop_size * 2 indices as we need two parents per offspring
    parent_indices = np.random.choice(
        range(len(population)),
        size = pop_size, # Select enough parents for pop_size offspring
        p = selection_probs,
        replace = True # Allows for one being to be selected multiple times
    )

    # 5. Create New Generation
    new_population = []
    # We'll create offspring by selecting parents and applying mutation/crossover
    # For simplicity in this loop, we'll select parents with replacement
    # and apply mutation. Crossover will be integrated in the next step.
    for i in range(pop_size):
        # Select a parent based on probabilities
        parent_index = np.random.choice(range(len(population)), p=selection_probs)
        parent = population[parent_index]

        # Create offspring blueprint (simple copy for now, crossover later)
        offspring_blueprint = parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints


        # Create new SimulatedBeing instance for offspring
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)


        # Apply mutation to the offspring, passing population statistics
        offspring.mutate(
            population_average_fitness = population_average_fitness,
            population_trait_variance = population_trait_variance,
            population_trait_averages = population_trait_averages
        )

        new_population.append(offspring)


    return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
        }


    phenotype_keys = list(population[0].phenotype.__dict__.keys())

    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

    return generation_stats

# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    # Convert Phenotype object to a dictionary for plotting
    phenotype_dict = phenotype_data.__dict__
    labels = list(phenotype_dict.keys())
    values = list(phenotype_dict.values())

    num_vars = len(labels)
    # Compute angle each trait goes to in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

    # Plot data and fill area
    ax.fill(angles, values, color = 'red', alpha = 0.25)
    ax.plot(angles, values, color = 'red', linewidth = 2)

    # Set the grid and labels
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set title
    ax.set_title(title, size = 16, color = 'blue', y = 1.1)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f" {filename}.png"))
    plt.close(fig) # Close the figure to free up memory


def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key = lambda being: being.fitness)


# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

## Update the evolutionary loop

### Subtask:
Modify the existing `evolve_one_generation` function ( or create a new simulation class) to use the `SimulatedBeing` class and incorporate the new mutation and recombination methods.


**Reasoning**:
Modify the `evolve_one_generation` function to correctly implement the parent selection, crossover, and mutation steps using the `SimulatedBeing` class and the `uniform_crossover` method, ensuring population statistics are calculated and passed to the `mutate` method.



from typing import List, Dict, Any
import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataclasses import dataclass, field

# Assuming Phenotype, PhenotypeTranslator, and SimulatedBeing classes are already defined and available
# Assuming track_population_statistics, create_radar_plot, get_fittest_individual are already defined and available
# Assuming sample_translation_rules is already defined and available

# Helper function for dynamic fitness (Placeholder - replace with actual logic)
def dynamic_fitness(phenotype: Phenotype, generation: int, env_factor: float) -> float:
    """
    Calculates dynamic fitness based on phenotype, generation, and environmental factor.

    Args:
        phenotype: The Phenotype object of the being.
        generation: The current generation number.
        env_factor: A factor representing environmental pressure.

    Returns:
        The calculated dynamic fitness score.
    """
    # Example dynamic fitness: favor resource efficiency early, then structural resilience
    if generation < 25:
        weights = {
            "resource_efficiency": 0.4 + 0.1 * env_factor,
            "knowledge_exchange": 0.4 - 0.05 * env_factor,
            "structural_resilience": 0.2 + 0.05 * env_factor
        }
    else:
        weights = {
            "resource_efficiency": 0.2 - 0.05 * env_factor,
            "knowledge_exchange": 0.2 + 0.05 * env_factor,
            "structural_resilience": 0.6 + 0.1 * env_factor
        }

    # Ensure weights sum to 1 (optional, depending on desired fitness scaling)
    # total_weight = sum(weights.values())
    # weights = {k: v / total_weight for k, v in weights.items()}

    # Calculate fitness based on phenotype attributes and weights
    fitness_score = (weights["resource_efficiency"] * phenotype.resource_efficiency +
                     weights["knowledge_exchange"] * phenotype.knowledge_exchange +
                     weights["structural_resilience"] * phenotype.structural_resilience)

    return fitness_score


# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp=0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    # Calculate fitness for each being if not already done
    fitnesses = np.array([being.fitness for being in population], dtype=float)

    # Numeric stable softmax
    scaled = fitnesses / float(temp)
    e = np.exp(scaled - np.max(scaled) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    return probs


def evolve_one_generation(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_std: float, env_factor: float, phenotype_translator: PhenotypeTranslator) -> List[SimulatedBeing]:
    """
    Simulates one generation of evolution.

    Args:
        population (List[SimulatedBeing]): The current population of beings.
        pop_size (int): The target size of the new population.
        generation (int): The current generation number, used for dynamic fitness.
        selection_temp (float): The temperature for softmax selection. Lower values
                                 lead to more "greedy" selection.
        mutation_std (float): The standard deviation of the random noise added during mutation.
        env_factor (float): A factor representing environmental pressure on certain traits.
        phenotype_translator: An instance of PhenotypeTranslator.


    Returns:
        List[SimulatedBeing]: The new population of beings after one generation of evolution.
    """
    if not population:
        return []

    # 1. Evaluate Fitness for all beings in the population using dynamic fitness
    # Define dynamic fitness weights based on the generation number and env_factor
    # This logic is now within the dynamic_fitness helper function

    for being in population:
        being.fitness = dynamic_fitness(being.phenotype, generation, env_factor)

    # Get all fitness values to calculate selection probabilities
    fitnesses = [b.fitness for b in population]

    # Softmax for selection probability. Lower temp leads to more greedy selection.
    scaled_fitness = np.array(fitnesses) / float(selection_temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    # 2. Select Parents based on fitness probabilities
    # Select pop_size * 2 indices as we need two parents per offspring for crossover
    parent_indices = np.random.choice(
        range(len(population)),
        size=pop_size, # Select enough parents to create a full population
        p=probs,
        replace=True # Allows a highly-fit parent to be selected multiple times
    )


    # 3. Create the New Generation through Crossover & Mutation
    new_population = []
    # We'll create offspring in pairs for crossover, so we need pop_size // 2 pairs
    # Ensure pop_size is even or handle the last offspring separately if odd
    num_offspring_pairs = pop_size // 2
    remaining_offspring = pop_size % 2 # For odd pop_size

    # Select enough parents for crossover pairs
    crossover_parent_indices = np.random.choice(
        range(len(population)),
        size=num_offspring_pairs * 2,
        p=probs,
        replace=True
    )


    # Calculate population statistics *before* mutation for adaptive mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    population_average_fitness = np.mean(fitnesses)
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    current_pop_stats = {
        "average_fitness": population_average_fitness,
        "trait_averages": population_trait_averages,
        "trait_variance": population_trait_variance
    }


    for i in range(0, len(crossover_parent_indices), 2):
        parent1_index = crossover_parent_indices[i]
        parent2_index = crossover_parent_indices[i+1]

        parent1 = population[parent1_index]
        parent2 = population[parent2_index]

        # Perform crossover to create offspring
        # Pass the phenotype_translator to the crossover method
        offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

        # Calculate fitness for the newly created offspring using the current generation's weights
        # This ensures self.fitness is available for fitness-dependent mutation
        offspring.calculate_fitness(weights=dynamic_fitness(offspring.phenotype, generation, env_factor))

        # Apply mutation to the offspring, passing population statistics and mutation_std
        offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std # Pass mutation_std
        )

        new_population.append(offspring)

    # If pop_size is odd, select one more parent and create one more offspring (mutation only)
    if remaining_offspring > 0:
        single_parent_index = np.random.choice(range(len(population)), p=probs)
        single_parent = population[single_parent_index]
        # Create offspring by cloning and mutating (no crossover)
        # This assumes blueprint can be copied and then mutated
        single_offspring_blueprint = single_parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints
        single_offspring = SimulatedBeing(single_offspring_blueprint, phenotype_translator)

        # Calculate fitness for the single offspring
        single_offspring.calculate_fitness(weights=dynamic_fitness(single_offspring.phenotype, generation, env_factor))

        single_offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std
        )
        new_population.append(single_offspring)


    return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
        }

    phenotype_keys = list(population[0].phenotype.__dict__.keys())

    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

    return generation_stats

# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    # Convert Phenotype object to a dictionary for plotting
    phenotype_dict = phenotype_data.__dict__
    labels = list(phenotype_dict.keys())
    values = list(phenotype_dict.values())

    num_vars = len(labels)
    # Compute angle each trait goes to in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

    # Plot data and fill area
    ax.fill(angles, values, color = 'red', alpha = 0.25)
    ax.plot(angles, values, color = 'red', linewidth = 2)

    # Set the grid and labels
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set title
    ax.set_title(title, size = 16, color = 'blue', y = 1.1)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f" {filename}.png"))
    plt.close(fig) # Close the figure to free up memory


def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key = lambda being: being.fitness)

# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

# --- Run the Simulation ---
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Simulation parameters for evolve_population
selection_temp = 0.1
mutation_std = 0.05 # Standard deviation for mutation noise
env_factor = 0.5 # Example environmental factor

# Create the PhenotypeTranslator instance once
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

translator = PhenotypeTranslator(sample_translation_rules)

# Generate the initial population using the updated function
population = generate_initial_population(size=pop_size, translator=translator)

# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []

for gen in range(num_generations):
    # Evolve the population using the updated function with dynamic weights and the translator
    # The dynamic fitness calculation is now handled within evolve_population by calling dynamic_fitness
    population = evolve_one_generation(
        population,
        pop_size=pop_size,
        generation=gen,
        selection_temp=selection_temp,
        mutation_std=mutation_std,
        env_factor=env_factor,
        phenotype_translator=translator # Pass the translator
    )

    # Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    avg_phenotype = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    avg_phenotypes_history.append(avg_phenotype)

    # Track and store population statistics (fitness is already calculated in evolve_population)
    gen_stats = track_population_statistics(population, gen)
    population_statistics_history.append(gen_stats)

    # Visualize the fittest individual and sampled individuals
    fittest_individual = get_fittest_individual(population)
    if fittest_individual:
        create_radar_plot(fittest_individual.phenotype,
                          f"Fittest Phenotype - Gen {gen}",
                          f"fittest_phenotype_gen_{gen}")

    if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
        sampled_individual = random.choice(population)
        create_radar_plot(sampled_individual.phenotype,
                          f"Sampled Phenotype - Gen {gen}",
                          f"sampled_phenotype_gen_{gen}")


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label="Resource Efficiency", color="green")
ax.plot(gens, knowledge_ex, label="Knowledge Exchange", color="blue")
ax.plot(gens, structural_res, label="Structural Resilience", color="red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label="Resource Efficiency Std Dev", color="green", linestyle='--')
ax_std.plot(gens, knowledge_ex_std, label="Knowledge Exchange Std Dev", color="blue", linestyle='--')
ax_std.plot(gens, structural_res_std, label="Structural Resilience Std Dev", color="red", linestyle='--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")

**Reasoning**:
The error "AttributeError: 'numpy.float64' object has no attribute 'keys'" indicates that the `dynamic_fitness` function is returning a float instead of a dictionary of weights when called within `offspring.calculate_fitness`. I need to fix the `dynamic_fitness` function to return the calculated weights dictionary.



from typing import List, Dict, Any
import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataclasses import dataclass, field

# Assuming Phenotype, PhenotypeTranslator, and SimulatedBeing classes are already defined and available
# Assuming track_population_statistics, create_radar_plot, get_fittest_individual are already defined and available
# Assuming sample_translation_rules is already defined and available

# Helper function for dynamic fitness (Placeholder - replace with actual logic)
def dynamic_fitness(phenotype: Phenotype, generation: int, env_factor: float) -> Dict[str, float]:
    """
    Calculates dynamic fitness weights based on phenotype, generation, and environmental factor.

    Args:
        phenotype: The Phenotype object of the being.
        generation: The current generation number.
        env_factor: A factor representing environmental pressure.

    Returns:
        A dictionary of weights for each phenotype trait.
    """
    # Example dynamic fitness: favor resource efficiency early, then structural resilience
    if generation < 25:
        weights = {
            "resource_efficiency": 0.4 + 0.1 * env_factor,
            "knowledge_exchange": 0.4 - 0.05 * env_factor,
            "structural_resilience": 0.2 + 0.05 * env_factor
        }
    else:
        weights = {
            "resource_efficiency": 0.2 - 0.05 * env_factor,
            "knowledge_exchange": 0.2 + 0.05 * env_factor,
            "structural_resilience": 0.6 + 0.1 * env_factor
        }

    # Ensure weights sum to 1 (optional, depending on desired fitness scaling)
    # total_weight = sum(weights.values())
    # weights = {k: v / total_weight for k, v in weights.items()}

    return weights


# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp=0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    # Calculate fitness for each being if not already done
    fitnesses = np.array([being.fitness for being in population], dtype=float)

    # Numeric stable softmax
    scaled = fitnesses / float(temp)
    e = np.exp(scaled - np.max(scaled) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    return probs


def evolve_one_generation(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_std: float, env_factor: float, phenotype_translator: PhenotypeTranslator) -> List[SimulatedBeing]:
    """
    Simulates one generation of evolution.

    Args:
        population (List[SimulatedBeing]): The current population of beings.
        pop_size (int): The target size of the new population.
        generation (int): The current generation number, used for dynamic fitness.
        selection_temp (float): The temperature for softmax selection. Lower values
                                 lead to more "greedy" selection.
        mutation_std (float): The standard deviation of the random noise added during mutation.
        env_factor (float): A factor representing environmental pressure on certain traits.
        phenotype_translator: An instance of PhenotypeTranslator.


    Returns:
        List[SimulatedBeing]: The new population of beings after one generation of evolution.
    """
    if not population:
        return []

    # 1. Evaluate Fitness for all beings in the population using dynamic fitness
    # Define dynamic fitness weights based on the generation number and env_factor
    # This logic is now within the dynamic_fitness helper function

    for being in population:
        # Call dynamic_fitness to get the weights for this generation
        current_fitness_weights = dynamic_fitness(being.phenotype, generation, env_factor)
        being.fitness = being.calculate_fitness(weights=current_fitness_weights)


    # Get all fitness values to calculate selection probabilities
    fitnesses = [b.fitness for b in population]

    # Softmax for selection probability. Lower temp leads to more greedy selection.
    scaled_fitness = np.array(fitnesses) / float(selection_temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    # 2. Select Parents based on fitness probabilities
    # Select pop_size * 2 indices as we need two parents per offspring for crossover
    parent_indices = np.random.choice(
        range(len(population)),
        size=pop_size, # Select enough parents to create a full population
        p=probs,
        replace=True # Allows a highly-fit parent to be selected multiple times
    )


    # 3. Create the New Generation through Crossover & Mutation
    new_population = []
    # We'll create offspring in pairs for crossover, so we need pop_size // 2 pairs
    # Ensure pop_size is even or handle the last offspring separately if odd
    num_offspring_pairs = pop_size // 2
    remaining_offspring = pop_size % 2 # For odd pop_size

    # Select enough parents for crossover pairs
    crossover_parent_indices = np.random.choice(
        range(len(population)),
        size=num_offspring_pairs * 2,
        p=probs,
        replace=True
    )


    # Calculate population statistics *before* mutation for adaptive mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    population_average_fitness = np.mean(fitnesses)
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    current_pop_stats = {
        "average_fitness": population_average_fitness,
        "trait_averages": population_trait_averages,
        "trait_variance": population_trait_variance
    }


    for i in range(0, len(crossover_parent_indices), 2):
        parent1_index = crossover_parent_indices[i]
        parent2_index = crossover_parent_indices[i+1]

        parent1 = population[parent1_index]
        parent2 = population[parent2_index]

        # Perform crossover to create offspring
        # Pass the phenotype_translator to the crossover method
        offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

        # Calculate fitness for the newly created offspring using the current generation's weights
        # This ensures self.fitness is available for fitness-dependent mutation
        # Call dynamic_fitness to get the weights for the offspring's fitness calculation
        offspring_fitness_weights = dynamic_fitness(offspring.phenotype, generation, env_factor)
        offspring.calculate_fitness(weights=offspring_fitness_weights)

        # Apply mutation to the offspring, passing population statistics and mutation_std
        offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std # Pass mutation_std
        )

        new_population.append(offspring)

    # If pop_size is odd, select one more parent and create one more offspring (mutation only)
    if remaining_offspring > 0:
        single_parent_index = np.random.choice(range(len(population)), p=probs)
        single_parent = population[single_parent_index]
        # Create offspring by cloning and mutating (no crossover)
        # This assumes blueprint can be copied and then mutated
        single_offspring_blueprint = single_parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints
        single_offspring = SimulatedBeing(single_offspring_blueprint, phenotype_translator)

        # Calculate fitness for the single offspring
        single_offspring_fitness_weights = dynamic_fitness(single_offspring.phenotype, generation, env_factor)
        single_offspring.calculate_fitness(weights=single_offspring_fitness_weights)

        single_offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std
        )
        new_population.append(single_offspring)


    return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
        }

    phenotype_keys = list(population[0].phenotype.__dict__.keys())

    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

    return generation_stats

# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    # Convert Phenotype object to a dictionary for plotting
    phenotype_dict = phenotype_data.__dict__
    labels = list(phenotype_dict.keys())
    values = list(phenotype_dict.values())

    num_vars = len(labels)
    # Compute angle each trait goes to in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

    # Plot data and fill area
    ax.fill(angles, values, color = 'red', alpha = 0.25)
    ax.plot(angles, values, color = 'red', linewidth = 2)

    # Set the grid and labels
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set title
    ax.set_title(title, size = 16, color = 'blue', y = 1.1)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f" {filename}.png"))
    plt.close(fig) # Close the figure to free up memory


def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key = lambda being: being.fitness)

# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

# --- Run the Simulation ---
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Simulation parameters for evolve_population
selection_temp = 0.1
mutation_std = 0.05 # Standard deviation for mutation noise
env_factor = 0.5 # Example environmental factor

# Create the PhenotypeTranslator instance once
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

translator = PhenotypeTranslator(sample_translation_rules)

# Generate the initial population using the updated function
population = generate_initial_population(size=pop_size, translator=translator)

# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []

for gen in range(num_generations):
    # Evolve the population using the updated function with dynamic weights and the translator
    # The dynamic fitness calculation is now handled within evolve_population by calling dynamic_fitness
    population = evolve_one_generation(
        population,
        pop_size=pop_size,
        generation=gen,
        selection_temp=selection_temp,
        mutation_std=mutation_std,
        env_factor=env_factor,
        phenotype_translator=translator # Pass the translator
    )

    # Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    avg_phenotype = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    avg_phenotypes_history.append(avg_phenotype)

    # Track and store population statistics (fitness is already calculated in evolve_population)
    gen_stats = track_population_statistics(population, gen)
    population_statistics_history.append(gen_stats)

    # Visualize the fittest individual and sampled individuals
    fittest_individual = get_fittest_individual(population)
    if fittest_individual:
        create_radar_plot(fittest_individual.phenotype,
                          f"Fittest Phenotype - Gen {gen}",
                          f"fittest_phenotype_gen_{gen}")

    if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
        sampled_individual = random.choice(population)
        create_radar_plot(sampled_individual.phenotype,
                          f"Sampled Phenotype - Gen {gen}",
                          f"sampled_phenotype_gen_{gen}")


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label="Resource Efficiency", color="green")
ax.plot(gens, knowledge_ex, label="Knowledge Exchange", color="blue")
ax.plot(gens, structural_res, label="Structural Resilience", color="red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label="Resource Efficiency Std Dev", color="green", linestyle='--')
ax_std.plot(gens, knowledge_ex_std, label="Knowledge Exchange Std Dev", color="blue", linestyle='--')
ax_std.plot(gens, structural_res_std, label="Structural Resilience Std Dev", color="red", linestyle='--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")

**Reasoning**:
The error "TypeError: SimulatedBeing.mutate() got an unexpected keyword argument 'mutation_std'" indicates that the `mutate` method in the `SimulatedBeing` class does not accept the `mutation_std` argument. I need to modify the `mutate` method in the `SimulatedBeing` class to accept `mutation_std`.



from typing import List, Dict, Any
import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Phenotype:
    """Represents the observable characteristics of a simulated being."""
    resource_efficiency: float = 0.0
    knowledge_exchange: float = 0.0
    structural_resilience: float = 0.0

class PhenotypeTranslator:
    """Translates a blueprint dictionary into a Phenotype object based on defined rules."""

    def __init__(self, blueprint_to_phenotype_rules: dict):
        """
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
        self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

    def translate(self, blueprint: dict) -> Phenotype:
        """
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
        phenotype_values = {}

        # Apply translation rules to calculate phenotype trait values
        for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
            calculated_value = 0.0
            for rule in rules:
                component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
                calculated_value += component_value * rule['weight']

            # Apply tanh normalization
            phenotype_values[phenotype_trait] = np.tanh(calculated_value)

        # Create and return a Phenotype object
        return Phenotype(
            resource_efficiency=phenotype_values.get("resource_efficiency", 0.0),
            knowledge_exchange=phenotype_values.get("knowledge_exchange", 0.0),
            structural_resilience=phenotype_values.get("structural_resilience", 0.0)
        )

class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    def __init__(self, blueprint: dict, phenotype_translator: PhenotypeTranslator):
        """
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
        self.blueprint = blueprint
        self.phenotype_translator = phenotype_translator
        self.phenotype: Phenotype = self.translate_blueprint_to_phenotype()
        self.fitness: float = 0.0 # Initialize fitness
        # Add a base mutation rate to the being's attributes
        self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)


    def translate_blueprint_to_phenotype(self) -> Phenotype:
        """
        Translates the being's blueprint into its phenotype using the provided translator.
        """
        # Ensure all expected keys are present in the blueprint, using defaults if necessary
        # This part can be more sophisticated depending on the desired blueprint structure and defaults
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        # This ensures the translator always receives a complete blueprint structure
        complete_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


        return self.phenotype_translator.translate(self.blueprint)


    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None, mutation_std: float = 0.05):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
            mutation_std: The standard deviation of the random noise added during mutation.
        """
        current_mutation_rate = self.base_mutation_rate
        # mutation_strength = 0.05 # Base strength of mutation - now passed as mutation_std

        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value, rate, strength):
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_std # Use passed mutation_std

                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (a simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages are low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint and the translator
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
        return offspring

# Helper function for dynamic fitness (Placeholder - replace with actual logic)
def dynamic_fitness(phenotype: Phenotype, generation: int, env_factor: float) -> Dict[str, float]:
    """
    Calculates dynamic fitness weights based on phenotype, generation, and environmental factor.

    Args:
        phenotype: The Phenotype object of the being.
        generation: The current generation number.
        env_factor: A factor representing environmental pressure.

    Returns:
        A dictionary of weights for each phenotype trait.
    """
    # Example dynamic fitness: favor resource efficiency early, then structural resilience
    if generation < 25:
        weights = {
            "resource_efficiency": 0.4 + 0.1 * env_factor,
            "knowledge_exchange": 0.4 - 0.05 * env_factor,
            "structural_resilience": 0.2 + 0.05 * env_factor
        }
    else:
        weights = {
            "resource_efficiency": 0.2 - 0.05 * env_factor,
            "knowledge_exchange": 0.2 + 0.05 * env_factor,
            "structural_resilience": 0.6 + 0.1 * env_factor
        }

    # Ensure weights sum to 1 (optional, depending on desired fitness scaling)
    # total_weight = sum(weights.values())
    # weights = {k: v / total_weight for k, v in weights.items()}

    return weights

# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp=0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    # Calculate fitness for each being if not already done
    fitnesses = np.array([being.fitness for being in population], dtype=float)

    # Numeric stable softmax
    scaled = fitnesses / float(temp)
    e = np.exp(scaled - np.max(scaled) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    return probs


def evolve_one_generation(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_std: float, env_factor: float, phenotype_translator: PhenotypeTranslator) -> List[SimulatedBeing]:
    """
    Simulates one generation of evolution.

    Args:
        population (List[SimulatedBeing]): The current population of beings.
        pop_size (int): The target size of the new population.
        generation (int): The current generation number, used for dynamic fitness.
        selection_temp (float): The temperature for softmax selection. Lower values
                                 lead to more "greedy" selection.
        mutation_std (float): The standard deviation of the random noise added during mutation.
        env_factor (float): A factor representing environmental pressure on certain traits.
        phenotype_translator: An instance of PhenotypeTranslator.


    Returns:
        List[SimulatedBeing]: The new population of beings after one generation of evolution.
    """
    if not population:
        return []

    # 1. Evaluate Fitness for all beings in the population using dynamic fitness
    # Define dynamic fitness weights based on the generation number and env_factor
    # This logic is now within the dynamic_fitness helper function

    for being in population:
        # Call dynamic_fitness to get the weights for this generation
        current_fitness_weights = dynamic_fitness(being.phenotype, generation, env_factor)
        being.fitness = being.calculate_fitness(weights=current_fitness_weights)


    # Get all fitness values to calculate selection probabilities
    fitnesses = [b.fitness for b in population]

    # Softmax for selection probability. Lower temp leads to more greedy selection.
    scaled_fitness = np.array(fitnesses) / float(selection_temp)
    e = np.exp(scaled - np.max(scaled) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    # 2. Select Parents based on fitness probabilities
    # Select pop_size * 2 indices as we need two parents per offspring for crossover
    parent_indices = np.random.choice(
        range(len(population)),
        size=pop_size, # Select enough parents to create a full population
        p=probs,
        replace=True # Allows a highly-fit parent to be selected multiple times
    )


    # 3. Create the New Generation through Crossover & Mutation
    new_population = []
    # We'll create offspring in pairs for crossover, so we need pop_size // 2 pairs
    # Ensure pop_size is even or handle the last offspring separately if odd
    num_offspring_pairs = pop_size // 2
    remaining_offspring = pop_size % 2 # For odd pop_size

    # Select enough parents for crossover pairs
    crossover_parent_indices = np.random.choice(
        range(len(population)),
        size=num_offspring_pairs * 2,
        p=probs,
        replace=True
    )


    # Calculate population statistics *before* mutation for adaptive mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    population_average_fitness = np.mean(fitnesses)
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    current_pop_stats = {
        "average_fitness": population_average_fitness,
        "trait_averages": population_trait_averages,
        "trait_variance": population_trait_variance
    }


    for i in range(0, len(crossover_parent_indices), 2):
        parent1_index = crossover_parent_indices[i]
        parent2_index = crossover_parent_indices[i+1]

        parent1 = population[parent1_index]
        parent2 = population[parent2_index]

        # Perform crossover to create offspring
        # Pass the phenotype_translator to the crossover method
        offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

        # Calculate fitness for the newly created offspring using the current generation's weights
        # This ensures self.fitness is available for fitness-dependent mutation
        # Call dynamic_fitness to get the weights for the offspring's fitness calculation
        offspring_fitness_weights = dynamic_fitness(offspring.phenotype, generation, env_factor)
        offspring.calculate_fitness(weights=offspring_fitness_weights)

        # Apply mutation to the offspring, passing population statistics and mutation_std
        offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std # Pass mutation_std
        )

        new_population.append(offspring)

    # If pop_size is odd, select one more parent and create one more offspring (mutation only)
    if remaining_offspring > 0:
        single_parent_index = np.random.choice(range(len(population)), p=probs)
        single_parent = population[single_parent_index]
        # Create offspring by cloning and mutating (no crossover)
        # This assumes blueprint can be copied and then mutated
        single_offspring_blueprint = single_parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints
        single_offspring = SimulatedBeing(single_offspring_blueprint, phenotype_translator)

        # Calculate fitness for the single offspring
        single_offspring_fitness_weights = dynamic_fitness(single_offspring.phenotype, generation, env_factor)
        single_offspring.calculate_fitness(weights=single_offspring_fitness_weights)

        single_offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std
        )
        new_population.append(single_offspring)


    return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
        }

    phenotype_keys = list(population[0].phenotype.__dict__.keys())

    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

    return generation_stats

# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    # Convert Phenotype object to a dictionary for plotting
    phenotype_dict = phenotype_data.__dict__
    labels = list(phenotype_dict.keys())
    values = list(phenotype_dict.values())

    num_vars = len(labels)
    # Compute angle each trait goes to in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

    # Plot data and fill area
    ax.fill(angles, values, color = 'red', alpha = 0.25)
    ax.plot(angles, values, color = 'red', linewidth = 2)

    # Set the grid and labels
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set title
    ax.set_title(title, size = 16, color = 'blue', y = 1.1)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f" {filename}.png"))
    plt.close(fig) # Close the figure to free up memory


def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key = lambda being: being.fitness)

# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

# --- Run the Simulation ---
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Simulation parameters for evolve_population
selection_temp = 0.1
mutation_std = 0.05 # Standard deviation for mutation noise
env_factor = 0.5 # Example environmental factor

# Create the PhenotypeTranslator instance once
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

translator = PhenotypeTranslator(sample_translation_rules)

# Generate the initial population using the updated function
population = generate_initial_population(size=pop_size, translator=translator)

# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []

for gen in range(num_generations):
    # Evolve the population using the updated function with dynamic weights and the translator
    # The dynamic fitness calculation is now handled within evolve_population by calling dynamic_fitness
    population = evolve_one_generation(
        population,
        pop_size=pop_size,
        generation=gen,
        selection_temp=selection_temp,
        mutation_std=mutation_std,
        env_factor=env_factor,
        phenotype_translator=translator # Pass the translator
    )

    # Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    avg_phenotype = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    avg_phenotypes_history.append(avg_phenotype)

    # Track and store population statistics (fitness is already calculated in evolve_population)
    gen_stats = track_population_statistics(population, gen)
    population_statistics_history.append(gen_stats)

    # Visualize the fittest individual and sampled individuals
    fittest_individual = get_fittest_individual(population)
    if fittest_individual:
        create_radar_plot(fittest_individual.phenotype,
                          f"Fittest Phenotype - Gen {gen}",
                          f"fittest_phenotype_gen_{gen}")

    if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
        sampled_individual = random.choice(population)
        create_radar_plot(sampled_individual.phenotype,
                          f"Sampled Phenotype - Gen {gen}",
                          f"sampled_phenotype_gen_{gen}")


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label="Resource Efficiency", color="green")
ax.plot(gens, knowledge_ex, label="Knowledge Exchange", color="blue")
ax.plot(gens, structural_res, label="Structural Resilience", color="red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label="Resource Efficiency Std Dev", color="green", linestyle='--')
ax_std.plot(gens, knowledge_ex_std, label="Knowledge Exchange Std Dev", color="blue", linestyle='--')
ax_std.plot(gens, structural_res_std, label="Structural Resilience Std Dev", color="red", linestyle='--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")

**Reasoning**:
The error "NameError: name 'scaled' is not defined" in the `evolve_one_generation` function's selection probability calculation is because the variable is named `scaled_fitness` but `scaled` is used in the `np.exp` call. I need to correct the variable name to `scaled_fitness` in the `np.exp` call within the `evolve_one_generation` function.



from typing import List, Dict, Any
import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Phenotype:
    """Represents the observable characteristics of a simulated being."""
    resource_efficiency: float = 0.0
    knowledge_exchange: float = 0.0
    structural_resilience: float = 0.0

class PhenotypeTranslator:
    """Translates a blueprint dictionary into a Phenotype object based on defined rules."""

    def __init__(self, blueprint_to_phenotype_rules: dict):
        """
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
        self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

    def translate(self, blueprint: dict) -> Phenotype:
        """
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
        phenotype_values = {}

        # Apply translation rules to calculate phenotype trait values
        for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
            calculated_value = 0.0
            for rule in rules:
                component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
                calculated_value += component_value * rule['weight']

            # Apply tanh normalization
            phenotype_values[phenotype_trait] = np.tanh(calculated_value)

        # Create and return a Phenotype object
        return Phenotype(
            resource_efficiency=phenotype_values.get("resource_efficiency", 0.0),
            knowledge_exchange=phenotype_values.get("knowledge_exchange", 0.0),
            structural_resilience=phenotype_values.get("structural_resilience", 0.0)
        )

class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    def __init__(self, blueprint: dict, phenotype_translator: PhenotypeTranslator):
        """
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
        self.blueprint = blueprint
        self.phenotype_translator = phenotype_translator
        self.phenotype: Phenotype = self.translate_blueprint_to_phenotype()
        self.fitness: float = 0.0 # Initialize fitness
        # Add a base mutation rate to the being's attributes
        self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)


    def translate_blueprint_to_phenotype(self) -> Phenotype:
        """
        Translates the being's blueprint into its phenotype using the provided translator.
        """
        # Ensure all expected keys are present in the blueprint, using defaults if necessary
        # This part can be more sophisticated depending on the desired blueprint structure and defaults
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        # This ensures the translator always receives a complete blueprint structure
        complete_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


        return self.phenotype_translator.translate(self.blueprint)


    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None, mutation_std: float = 0.05):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
            mutation_std: The standard deviation of the random noise added during mutation.
        """
        current_mutation_rate = self.base_mutation_rate
        # mutation_strength = 0.05 # Base strength of mutation - now passed as mutation_std

        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value, rate, strength):
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_std # Use passed mutation_std

                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (a simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages are low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint and the translator
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
        return offspring

# Helper function for dynamic fitness (Placeholder - replace with actual logic)
def dynamic_fitness(phenotype: Phenotype, generation: int, env_factor: float) -> Dict[str, float]:
    """
    Calculates dynamic fitness weights based on phenotype, generation, and environmental factor.

    Args:
        phenotype: The Phenotype object of the being.
        generation: The current generation number.
        env_factor: A factor representing environmental pressure.

    Returns:
        A dictionary of weights for each phenotype trait.
    """
    # Example dynamic fitness: favor resource efficiency early, then structural resilience
    if generation < 25:
        weights = {
            "resource_efficiency": 0.4 + 0.1 * env_factor,
            "knowledge_exchange": 0.4 - 0.05 * env_factor,
            "structural_resilience": 0.2 + 0.05 * env_factor
        }
    else:
        weights = {
            "resource_efficiency": 0.2 - 0.05 * env_factor,
            "knowledge_exchange": 0.2 + 0.05 * env_factor,
            "structural_resilience": 0.6 + 0.1 * env_factor
        }

    # Ensure weights sum to 1 (optional, depending on desired fitness scaling)
    # total_weight = sum(weights.values())
    # weights = {k: v / total_weight for k, v in weights.items()}

    return weights

# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp=0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    # Calculate fitness for each being if not already done
    fitnesses = np.array([being.fitness for being in population], dtype=float)

    # Numeric stable softmax
    scaled_fitness = fitnesses / float(temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    return probs


def evolve_one_generation(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_std: float, env_factor: float, phenotype_translator: PhenotypeTranslator) -> List[SimulatedBeing]:
    """
    Simulates one generation of evolution.

    Args:
        population (List[SimulatedBeing]): The current population of beings.
        pop_size (int): The target size of the new population.
        generation (int): The current generation number, used for dynamic fitness.
        selection_temp (float): The temperature for softmax selection. Lower values
                                 lead to more "greedy" selection.
        mutation_std (float): The standard deviation of the random noise added during mutation.
        env_factor (float): A factor representing environmental pressure on certain traits.
        phenotype_translator: An instance of PhenotypeTranslator.


    Returns:
        List[SimulatedBeing]: The new population of beings after one generation of evolution.
    """
    if not population:
        return []

    # 1. Evaluate Fitness for all beings in the population using dynamic fitness
    # Define dynamic fitness weights based on the generation number and env_factor
    # This logic is now within the dynamic_fitness helper function

    for being in population:
        # Call dynamic_fitness to get the weights for this generation
        current_fitness_weights = dynamic_fitness(being.phenotype, generation, env_factor)
        being.fitness = being.calculate_fitness(weights=current_fitness_weights)


    # Get all fitness values to calculate selection probabilities
    fitnesses = [b.fitness for b in population]

    # Softmax for selection probability. Lower temp leads to more greedy selection.
    scaled_fitness = np.array(fitnesses) / float(selection_temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    # 2. Select Parents based on fitness probabilities
    # Select pop_size * 2 indices as we need two parents per offspring for crossover
    parent_indices = np.random.choice(
        range(len(population)),
        size=pop_size, # Select enough parents to create a full population
        p=probs,
        replace=True # Allows a highly-fit parent to be selected multiple times
    )


    # 3. Create the New Generation through Crossover & Mutation
    new_population = []
    # We'll create offspring in pairs for crossover, so we need pop_size // 2 pairs
    # Ensure pop_size is even or handle the last offspring separately if odd
    num_offspring_pairs = pop_size // 2
    remaining_offspring = pop_size % 2 # For odd pop_size

    # Select enough parents for crossover pairs
    crossover_parent_indices = np.random.choice(
        range(len(population)),
        size=num_offspring_pairs * 2,
        p=probs,
        replace=True
    )


    # Calculate population statistics *before* mutation for adaptive mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    population_average_fitness = np.mean(fitnesses)
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    current_pop_stats = {
        "average_fitness": population_average_fitness,
        "trait_averages": population_trait_averages,
        "trait_variance": population_trait_variance
    }


    for i in range(0, len(crossover_parent_indices), 2):
        parent1_index = crossover_parent_indices[i]
        parent2_index = crossover_parent_indices[i+1]

        parent1 = population[parent1_index]
        parent2 = population[parent2_index]

        # Perform crossover to create offspring
        # Pass the phenotype_translator to the crossover method
        offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

        # Calculate fitness for the newly created offspring using the current generation's weights
        # This ensures self.fitness is available for fitness-dependent mutation
        # Call dynamic_fitness to get the weights for the offspring's fitness calculation
        offspring_fitness_weights = dynamic_fitness(offspring.phenotype, generation, env_factor)
        offspring.calculate_fitness(weights=offspring_fitness_weights)

        # Apply mutation to the offspring, passing population statistics and mutation_std
        offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std # Pass mutation_std
        )

        new_population.append(offspring)

    # If pop_size is odd, select one more parent and create one more offspring (mutation only)
    if remaining_offspring > 0:
        single_parent_index = np.random.choice(range(len(population)), p=probs)
        single_parent = population[single_parent_index]
        # Create offspring by cloning and mutating (no crossover)
        # This assumes blueprint can be copied and then mutated
        single_offspring_blueprint = single_parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints
        single_offspring = SimulatedBeing(single_offspring_blueprint, phenotype_translator)

        # Calculate fitness for the single offspring
        single_offspring_fitness_weights = dynamic_fitness(single_offspring.phenotype, generation, env_factor)
        single_offspring.calculate_fitness(weights=single_offspring_fitness_weights)

        single_offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std
        )
        new_population.append(single_offspring)


    return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
        }

    phenotype_keys = list(population[0].phenotype.__dict__.keys())

    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

    return generation_stats

# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    # Convert Phenotype object to a dictionary for plotting
    phenotype_dict = phenotype_data.__dict__
    labels = list(phenotype_dict.keys())
    values = list(phenotype_dict.values())

    num_vars = len(labels)
    # Compute angle each trait goes to in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

    # Plot data and fill area
    ax.fill(angles, values, color = 'red', alpha = 0.25)
    ax.plot(angles, values, color = 'red', linewidth = 2)

    # Set the grid and labels
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set title
    ax.set_title(title, size = 16, color = 'blue', y = 1.1)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f" {filename}.png"))
    plt.close(fig) # Close the figure to free up memory


def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key = lambda being: being.fitness)

# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

# --- Run the Simulation ---
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Simulation parameters for evolve_population
selection_temp = 0.1
mutation_std = 0.05 # Standard deviation for mutation noise
env_factor = 0.5 # Example environmental factor

# Create the PhenotypeTranslator instance once
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

translator = PhenotypeTranslator(sample_translation_rules)

# Generate the initial population using the updated function
population = generate_initial_population(size=pop_size, translator=translator)

# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []

for gen in range(num_generations):
    # Evolve the population using the updated function with dynamic weights and the translator
    # The dynamic fitness calculation is now handled within evolve_population by calling dynamic_fitness
    population = evolve_one_generation(
        population,
        pop_size=pop_size,
        generation=gen,
        selection_temp=selection_temp,
        mutation_std=mutation_std,
        env_factor=env_factor,
        phenotype_translator=translator # Pass the translator
    )

    # Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    avg_phenotype = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    avg_phenotypes_history.append(avg_phenotype)

    # Track and store population statistics (fitness is already calculated in evolve_population)
    gen_stats = track_population_statistics(population, gen)
    population_statistics_history.append(gen_stats)

    # Visualize the fittest individual and sampled individuals
    fittest_individual = get_fittest_individual(population)
    if fittest_individual:
        create_radar_plot(fittest_individual.phenotype,
                          f"Fittest Phenotype - Gen {gen}",
                          f"fittest_phenotype_gen_{gen}")

    if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
        sampled_individual = random.choice(population)
        create_radar_plot(sampled_individual.phenotype,
                          f"Sampled Phenotype - Gen {gen}",
                          f"sampled_phenotype_gen_{gen}")


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label="Resource Efficiency", color="green")
ax.plot(gens, knowledge_ex, label="Knowledge Exchange", color="blue")
ax.plot(gens, structural_res, label="Structural Resilience", color="red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label="Resource Efficiency Std Dev", color="green", linestyle='--')
ax_std.plot(gens, knowledge_ex_std, label="Knowledge Exchange Std Dev", color="blue", linestyle='--')
ax_std.plot(gens, structural_res_std, label="Structural Resilience Std Dev", color="red", linestyle='--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")

## Summary:

### Data Analysis Key Findings

*   The initial attempt to run the evolutionary simulation resulted in an `AttributeError` because the `dynamic_fitness` function was incorrectly returning a single float value instead of a dictionary of fitness weights as expected by the `calculate_fitness` method.
*   After correcting the return type of `dynamic_fitness`, a `TypeError` occurred in the `SimulatedBeing.mutate()` method because the `evolve_one_generation` function was attempting to pass a `mutation_std` argument that the method's signature did not initially accept.
*   Resolving the `TypeError` led to a `NameError` in the selection probability calculation within `evolve_one_generation`, where an undefined variable `scaled` was used instead of the correct variable `scaled_fitness`.

### Insights or Next Steps

*   Thorough type checking and signature matching between function calls and definitions are crucial to avoid runtime errors, especially when integrating new components or modifying existing functions.
*   Detailed error messages, such as `AttributeError`, `TypeError`, and `NameError`, provide specific clues about the nature and location of issues, facilitating targeted debugging.


import random
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Dict, List, Any

@dataclass
class Phenotype:
    """Represents the observable characteristics of a simulated being."""
    resource_efficiency: float = 0.0
    knowledge_exchange: float = 0.0
    structural_resilience: float = 0.0

class PhenotypeTranslator:
    """Translates a blueprint dictionary into a Phenotype object based on defined rules."""

    def __init__(self, blueprint_to_phenotype_rules: Dict[str, List[Dict[str, Any]]]):
        """
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
        self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

    def translate(self, blueprint: Dict[str, Any]) -> Phenotype:
        """
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
        phenotype_values = {}

        # Apply translation rules to calculate phenotype trait values
        for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
            calculated_value = 0.0
            for rule in rules:
                component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
                calculated_value += component_value * rule['weight']

            # Apply tanh normalization
            phenotype_values[phenotype_trait] = np.tanh(calculated_value)

        # Create and return a Phenotype object
        return Phenotype(
            resource_efficiency = phenotype_values.get("resource_efficiency", 0.0),
            knowledge_exchange = phenotype_values.get("knowledge_exchange", 0.0),
            structural_resilience = phenotype_values.get("structural_resilience", 0.0)
        )

@dataclass
class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    blueprint: Dict[str, Any]
    phenotype_translator: PhenotypeTranslator
    phenotype: Phenotype = field(init = False)
    fitness: float = field(init = False)
    base_mutation_rate: float = field(init = False)

    def __post_init__(self):
        """Translates the full blueprint into a phenotype upon creation."""

        # Ensure all expected keys are present in the blueprint, using defaults if necessary
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        complete_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing
        self.base_mutation_rate = self.blueprint["mutation_parameters"]["base_rate"]

        self.phenotype = self.translate_blueprint_to_phenotype()


    def translate_blueprint_to_phenotype(self) -> Phenotype:
        """
        Translates the being's blueprint into its phenotype using the provided translator.
        """

        return self.phenotype_translator.translate(self.blueprint)


    def calculate_fitness(self, weights: Dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: Dict[str, float] = None, population_trait_averages: Dict[str, float] = None, mutation_std: float = 0.05):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
            mutation_std: The standard deviation of the random noise added during mutation.
        """
        current_mutation_rate = self.base_mutation_rate
        # mutation_strength = 0.05 # Base strength of mutation - now passed as mutation_std


        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value: float, rate: float, strength: float) -> float:
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_std # Use passed mutation_std


                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (a simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages are low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint and the translator
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
        return offspring

# Define sample translation rules (This should match the blueprint structure)
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> List[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

def evolve_population(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_rate: float, mutation_std: float, env_factor: float, phenotype_translator: PhenotypeTranslator) -> List[SimulatedBeing]:
    """
    Simulates one generation of evolution.

    Args:
        population: The current list of SimulatedBeing instances.
        pop_size: The desired size of the next generation.
        generation: The current generation number, used for dynamic fitness.
        selection_temp: The temperature for softmax selection. Lower values lead to more "greedy" selection.
        mutation_rate: The base probability of mutation for a gene.
        mutation_std: The standard deviation of the random noise added during mutation.
        env_factor: A factor representing environmental pressure on certain traits.
        phenotype_translator: An instance of PhenotypeTranslator.

    Returns:
        A list of SimulatedBeing instances representing the next generation.
    """

    # 1. Evaluate Fitness for all beings in the population
    for being in population:
        being.calculate_fitness(weights=dynamic_fitness(being.phenotype, generation, env_factor))

    # 2. Calculate Population Statistics for mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    fitnesses = [being.fitness for being in population]
    population_average_fitness = np.mean(fitnesses)
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }

    # 3. Calculate Selection Probabilities
    selection_probs = selection_probabilities(population, temp=selection_temp)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same)
    if np.sum(selection_probs) == 0:
        selection_probs = np.ones(len(population)) / len(population)

    # 4. Select Parents (indices)
    # We need to select pairs of parents for crossover.
    # Select pop_size / 2 pairs
    num_pairs = pop_size // 2
    parent_indices = np.random.choice(
        range(len(population)),
        size=num_pairs * 2,  # Select enough parents to create 'num_pairs' offspring
        p=selection_probs,
        replace=True  # Allows for one being to be selected multiple times
    )

    # 5. Create the New Generation through Crossover & Mutation
    new_population = []
    for i in range(0, pop_size, 2):
        if i + 1 < pop_size:  # Make sure we have two valid parents
            parent1 = population[parent_indices[i]]
            parent2 = population[parent_indices[i+1]]

            # Perform crossover to create offspring, passing the translator
            offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

            # Apply mutation to the offspring, passing population statistics
            offspring.mutate(
                population_average_fitness=population_average_fitness,
                population_trait_variance=population_trait_variance,
                population_trait_averages=population_trait_averages,
                mutation_std=mutation_std  # Pass mutation_std
            )
            new_population.append(offspring)
        else:  # Handle odd population size, mutate last individual
            parent = population[parent_indices[i]]
            offspring = SimulatedBeing(blueprint=parent.blueprint, phenotype_translator=phenotype_translator)
            offspring.mutate(
                population_average_fitness=population_average_fitness,
                population_trait_variance=population_trait_variance,
                population_trait_averages=population_trait_averages,
                mutation_std=mutation_std
            )
            new_population.append(offspring)
            break  # exit loop if we have only 1 individual left


    return new_population

# Helper function for dynamic fitness (Placeholder - replace with actual logic)
def dynamic_fitness(phenotype: Phenotype, generation: int, env_factor: float) -> float:
    """
    Calculates dynamic fitness based on phenotype, generation, and environmental factor.

    Args:
        phenotype: The Phenotype object of the being.
        generation: The current generation number.
        env_factor: A factor representing environmental pressure.

    Returns:
        The calculated dynamic fitness score.
    """
    # Example dynamic fitness: favor resource efficiency early, then structural resilience
    if generation < 25:
        weights = {
            "resource_efficiency": 0.4 + 0.1 * env_factor,
            "knowledge_exchange": 0.4 - 0.05 * env_factor,
            "structural_resilience": 0.2 + 0.05 * env_factor
        }
    else:
        weights = {
            "resource_efficiency": 0.2 - 0.05 * env_factor,
            "knowledge_exchange": 0.2 + 0.05 * env_factor,
            "structural_resilience": 0.6 + 0.1 * env_factor
        }

    # Calculate fitness based on phenotype attributes and weights
    fitness_score = (weights["resource_efficiency"] * phenotype.resource_efficiency +
                     weights["knowledge_exchange"] * phenotype.knowledge_exchange +
                     weights["structural_resilience"] * phenotype.structural_resilience)

    return fitness_score

# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: List[SimulatedBeing], temp: float = 0.1) -> np.ndarray:
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    fitnesses = np.array([being.fitness for being in population])
    scaled = fitnesses / float(temp)
    e = np.exp(scaled - np.max(scaled))
    probs = e / np.sum(e)
    return probs

# Helper function to track population statistics
def track_population_statistics(population: List[SimulatedBeing], generation: int) -> Dict[str, Any]:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation,
            "trait_means": {},
            "trait_stds": {},
            "average_fitness": 0.0
        }

    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])
    return generation_stats

def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output") -> None:
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

    labels = list(phenotype_data.__dict__.keys())
    values = list(phenotype_data.__dict__.values())

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    ax.set_yticklabels([])  # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, size=16, color='blue', y=1.1)

    plt.savefig(os.path.join(save_dir, f"{filename}.png"))
    plt.close()

def get_fittest_individual(population: List[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key=lambda being: being.fitness)

# Helper function to generate basic statistics about the traits.

def print_stats(generation: int, stats : Dict[str, float]):
  print(f"\n----- Generation {generation} Statistics -----")
  print(f"Average Fitness: {stats['average_fitness']:.4f}")
  print("Trait Means:")
  for trait, mean in stats['trait_means'].items():
    print(f"  {trait}: {mean:.4f}")
  print("Trait Standard Deviations:")
  for trait, std in stats['trait_stds'].items():
    print(f"  {trait}: {std:.4f}")
# Test the functions
def visualize_traits(avg_phenotypes_history, stats_df, num_generations):
  fig, axes = plt.subplots(2, 1, figsize=(10, 10))
  gens = range(num_generations)

  resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
  knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
  structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

  ax = axes[0] # average phenotype
  ax.plot(gens, resource_eff, label="Resource Efficiency", color="green")
  ax.plot(gens, knowledge_ex, label="Knowledge Exchange", color="blue")
  ax.plot(gens, structural_res, label="Structural Resilience", color="red")
  ax.set_title("Evolution of Average Phenotype Traits Over Generations")
  ax.set_xlabel("Generation")
  ax.set_ylabel("Average Trait Value")
  ax.legend()
  ax.grid(True)
  fig.tight_layout()

  # Plot the evolution of trait standard deviations
  ax_std = axes[1]
  resource_eff_std = stats_df["trait_stds"].apply(lambda x: x["resource_efficiency"])
  knowledge_ex_std = stats_df["trait_stds"].apply(lambda x: x["knowledge_exchange"])
  structural_res_std = stats_df["trait_stds"].apply(lambda x: x["structural_resilience"])

  ax_std.plot(gens, resource_eff_std, label="Resource Efficiency Std Dev", color="green", linestyle="--")
  ax_std.plot(gens, knowledge_ex_std, label="Knowledge Exchange Std Dev", color="blue", linestyle="--")
  ax_std.plot(gens, structural_res_std, label="Structural Resilience Std Dev", color="red", linestyle="--")
  ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
  ax_std.set_xlabel("Generation")
  ax_std.set_ylabel("Standard Deviation")
  ax_std.legend()
  ax_std.grid(True)

  fig.tight_layout()
  plt.show()

# Main Simulation Loop
def run_simulation(num_generations, pop_size, translator, initial_population = None):
  """Runs the evolutionary simulation.
  """
  # Parameters
  sampling_interval = 10
  selection_temp = 0.1
  mutation_rate = 0.01
  mutation_std = 0.05 # use the same mutation_std with the mutate function
  env_factor = 0.5
  if initial_population is None:
        population = generate_initial_population(size = pop_size, translator = translator)
  else:
        population = initial_population
  # Data structures to store historical data
  avg_phenotypes_history = []
  population_statistics_history = []

  for generation in range(num_generations):
    # Evolve the population using the updated function
    population = evolve_population(
        population = population,
        pop_size = pop_size,
        generation = generation,
        selection_temp = selection_temp,
        mutation_rate = mutation_rate,
        mutation_std = mutation_std,
        env_factor = env_factor,
        phenotype_translator = translator
    )

    # Track population statistics
    gen_stats = track_population_statistics(population, generation)
    population_statistics_history.append(gen_stats)

    # Printing basic stats
    print_stats(generation, gen_stats)

    # Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    avg_phenotype = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    avg_phenotypes_history.append(avg_phenotype)


    # Visualize the fittest individual and sampled individuals
    fittest_individual = get_fittest_individual(population)
    if fittest_individual:
      create_radar_plot(fittest_individual.phenotype,
                        f"Fittest Phenotype - Gen {generation}",
                        f"fittest_phenotype_gen_{generation}")

    if generation % sampling_interval == 0 or generation == num_generations - 1: # Sample at intervals and the final generation
      sampled_individual = random.choice(population)
      create_radar_plot(sampled_individual.phenotype,
                        f"Sampled Phenotype - Gen {generation}",
                        f"sampled_phenotype_gen_{generation}")
  # Convert the pop stats history to a pandas dataframe.
  stats_df = pd.DataFrame(population_statistics_history)
  visualize_traits(avg_phenotypes_history, stats_df, num_generations)
  return population, stats_df

# Simulation setup
num_generations = 50
pop_size = 100
translator = PhenotypeTranslator(sample_translation_rules)

# Run simulation
final_population, stats_df = run_simulation(num_generations, pop_size, translator)

print(stats_df.head())

## Define the `phenotypetranslator` class

### Subtask:
Define the `PhenotypeTranslator` class, which is used by the `SimulatedBeing` class to translate the blueprint into a phenotype.

**Reasoning**:
Define the `PhenotypeTranslator` class with an `__init__` method to store the translation rules and a `translate` method to apply these rules to a blueprint and return a `Phenotype` object with normalized values.

from dataclasses import dataclass
import numpy as np
import random

@dataclass
class Phenotype:
    """Represents the observable characteristics of a simulated being."""
    resource_efficiency: float = 0.0
    knowledge_exchange: float = 0.0
    structural_resilience: float = 0.0

class PhenotypeTranslator:
    """Translates a blueprint dictionary into a Phenotype object based on defined rules."""

    def __init__(self, blueprint_to_phenotype_rules: dict):
        """
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
        self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

    def translate(self, blueprint: dict) -> Phenotype:
        """
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
        phenotype_values = {}

        # Apply translation rules to calculate phenotype trait values
        for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
            calculated_value = 0.0
            for rule in rules:
                component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
                calculated_value += component_value * rule['weight']

            # Apply tanh normalization
            phenotype_values[phenotype_trait] = np.tanh(calculated_value)

        # Create and return a Phenotype object
        return Phenotype(
            resource_efficiency=phenotype_values.get("resource_efficiency", 0.0),
            knowledge_exchange=phenotype_values.get("knowledge_exchange", 0.0),
            structural_resilience=phenotype_values.get("structural_resilience", 0.0)
        )

# Define sample translation rules (This should match the blueprint structure)
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

# Example Usage:
# Create a sample blueprint
sample_blueprint_data = {
    "raw_materials": {
        "material_a_ratio": random.random(),
        "material_b_ratio": random.random(),
        "processing_efficiency_factor": random.random(),
    },
    "core_components": {
        "component_x_strength": random.random(),
        "component_y_flexibility": random.random(),
        "interconnection_density": random.random(),
    },
    "strategic_foundation": {
        "adaptability_gene": random.random(),
        "cooperation_gene": random.random(),
    },
    "unifying_strategy": {
        "optimization_parameter": random.random(),
    },
    "mutation_parameters": {
        "base_rate": 0.01
    }
}

# Create a translator instance
translator = PhenotypeTranslator(sample_translation_rules)

# Translate the blueprint
sample_phenotype = translator.translate(sample_blueprint_data)

print("Sample Blueprint:", sample_blueprint_data)
print("Translated Phenotype:", sample_phenotype)

## Update the `SimulatedBeing` class to use the `PhenotypeTranslator`

### Subtask:
Update the `SimulatedBeing` class to use the `PhenotypeTranslator` for blueprint to phenotype translation.

**Reasoning**:
Update the `SimulatedBeing` class to include a `PhenotypeTranslator` instance and use it in the `translate_blueprint_to_phenotype` method.

from dataclasses import dataclass, field
import numpy as np
import random

@dataclass
class Phenotype:
    """Represents the observable characteristics of a simulated being."""
    resource_efficiency: float = 0.0
    knowledge_exchange: float = 0.0
    structural_resilience: float = 0.0

class PhenotypeTranslator:
    """Translates a blueprint dictionary into a Phenotype object based on defined rules."""

    def __init__(self, blueprint_to_phenotype_rules: dict):
        """
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
        self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

    def translate(self, blueprint: dict) -> Phenotype:
        """
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
        phenotype_values = {}

        # Apply translation rules to calculate phenotype trait values
        for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
            calculated_value = 0.0
            for rule in rules:
                component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
                calculated_value += component_value * rule['weight']

            # Apply tanh normalization
            phenotype_values[phenotype_trait] = np.tanh(calculated_value)

        # Create and return a Phenotype object
        return Phenotype(
            resource_efficiency=phenotype_values.get("resource_efficiency", 0.0),
            knowledge_exchange=phenotype_values.get("knowledge_exchange", 0.0),
            structural_resilience=phenotype_values.get("structural_resilience", 0.0)
        )

class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    def __init__(self, blueprint: dict, phenotype_translator: PhenotypeTranslator):
        """
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
        self.blueprint = blueprint
        self.phenotype_translator = phenotype_translator
        self.phenotype: Phenotype = self.translate_blueprint_to_phenotype()
        self.fitness: float = 0.0 # Initialize fitness
        # Add a base mutation rate to the being's attributes
        self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)

    def translate_blueprint_to_phenotype(self) -> Phenotype:
        """
        Translates the being's blueprint into its phenotype using the provided translator.
        """
        # Ensure all expected keys are present in the blueprint, using defaults if necessary
        # This part can be more sophisticated depending on the desired blueprint structure and defaults
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        # This ensures the translator always receives a complete blueprint structure
        complete_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


        return self.phenotype_translator.translate(self.blueprint)


    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
        """
        current_mutation_rate = self.base_mutation_rate
        mutation_strength = 0.05 # Base strength of mutation

        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value, rate, strength):
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_strength

                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages are low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint and the translator
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
        return offspring

## Implement recombination/crossover as a function or class method

### Subtask:
Implement recombination/crossover as a function or class method

**Reasoning**:
The subtask is to implement the `uniform_crossover` method in the `SimulatedBeing` class. I will add a static method to the class that takes two `SimulatedBeing` instances, creates a new blueprint by randomly selecting values from either parent for each component, and then returns a new `SimulatedBeing` instance with this offspring blueprint.

import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt

class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    def __init__(self, blueprint: dict, phenotype_translator: 'PhenotypeTranslator'):
        """
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
        self.blueprint = blueprint
        self.phenotype_translator = phenotype_translator
        self.phenotype = self.translate_blueprint_to_phenotype()
        self.fitness = 0.0 # Initialize fitness
        # Add a base mutation rate to the being's attributes
        self.base_mutation_rate = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)


    def translate_blueprint_to_phenotype(self) -> 'Phenotype':
        """
        Translates the being's blueprint into its phenotype using the provided translator.
        """
        # Ensure all expected keys are present in the blueprint, using defaults if necessary
        # This part can be more sophisticated depending on the desired blueprint structure and defaults
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        # This ensures the translator always receives a complete blueprint structure
        complete_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


        return self.phenotype_translator.translate(self.blueprint)

    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
        """
        current_mutation_rate = self.base_mutation_rate
        mutation_strength = 0.05 # Base strength of mutation

        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value, rate, strength):
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_strength

                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (a simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages are low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: 'PhenotypeTranslator') -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint and the translator
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
        return offspring

# Example Usage (assuming you have two SimulatedBeing instances, parent1 and parent2)
# For demonstration, let's create two sample beings:
# Define sample translation rules (This should match the blueprint structure)
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

# Create a translator instance
translator = PhenotypeTranslator(sample_translation_rules)

parent1_blueprint = {
    "raw_materials": {"material_a_ratio": 0.7, "material_b_ratio": 0.3, "processing_efficiency_factor": 0.9},
    "core_components": {"component_x_strength": 0.8, "component_y_flexibility": 0.6, "interconnection_density": 0.75},
    "strategic_foundation": {"adaptability_gene": 0.5, "cooperation_gene": 0.8},
    "unifying_strategy": {"optimization_parameter": 0.95},
    "mutation_parameters": {"base_rate": 0.01}
}

parent2_blueprint = {
    "raw_materials": {"material_a_ratio": 0.2, "material_b_ratio": 0.8, "processing_efficiency_factor": 0.4},
    "core_components": {"component_x_strength": 0.3, "component_y_flexibility": 0.9, "interconnection_density": 0.25},
    "strategic_foundation": {"adaptability_gene": 0.9, "cooperation_gene": 0.1},
    "unifying_strategy": {"optimization_parameter": 0.05},
    "mutation_parameters": {"base_rate": 0.02}
}

parent1 = SimulatedBeing(parent1_blueprint, translator)
parent2 = SimulatedBeing(parent2_blueprint, translator)

offspring = SimulatedBeing.uniform_crossover(parent1, parent2, translator)

print("Parent 1 Blueprint:", parent1.blueprint)
print("Parent 2 Blueprint:", parent2.blueprint)
print("Offspring Blueprint:", offspring.blueprint)
print("Offspring Phenotype:", offspring.phenotype)

## Update population generation

### Subtask:
Modify the existing `generate_initial_population` function to create instances of the new `SimulatedBeing` class, providing each with a random initial blueprint and the `PhenotypeTranslator`.

**Reasoning**:
Modify the `generate_initial_population` function to create instances of the new `SimulatedBeing` class, providing each with a random initial blueprint and the `PhenotypeTranslator`.

# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp = 0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    # Calculate fitness for each being if not already done
    fitnesses = np.array([being.fitness for being in population], dtype = float)

    # Numeric stable softmax
    scaled = fitnesses / float(temp)
    e = np.exp(scaled - np.max(scaled))
    probs = e / np.sum(e)
    return probs

def evolve_one_generation(population: list[SimulatedBeing], pop_size: int, fitness_weights: dict, phenotype_translator: PhenotypeTranslator, selection_temp: float = 0.1) -> list[SimulatedBeing]:
    """
    Evolves the population by one generation using selection, crossover, and mutation.

    Args:
        population: The current list of SimulatedBeing instances.
        pop_size: The desired size of the next generation.
        fitness_weights: A dictionary of weights for each phenotype trait.
        phenotype_translator: An instance of PhenotypeTranslator.
        selection_temp: The temperature parameter for softmax selection.

    Returns:
        A new list of SimulatedBeing instances representing the next generation.
    """
    if not population:
        return []

    # 1. Calculate Fitness for the current population
    for being in population:
        being.calculate_fitness(weights = fitness_weights)

    # 2. Calculate Population Statistics for mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys()) # Get keys from a sample being's phenotype
    population_average_fitness = np.mean([being.fitness for being in population])
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }


    # 3. Calculate Selection Probabilities
    selection_probs = selection_probabilities(population, temp = selection_temp)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same)
    if np.sum(selection_probs) == 0:
        selection_probs = np.ones(len(population)) / len(population) # Assign equal probability


    # 4. Select Parents (indices)
    # We need to select pairs of parents for crossover.
    # Select pop_size * 2 indices as we need two parents per offspring
    parent_indices = np.random.choice(
        range(len(population)),
        size = pop_size, # Select enough parents for pop_size offspring
        p = selection_probs,
        replace = True # Allows for one being to be selected multiple times
    )

    # 5. Create New Generation
    new_population = []
    # We'll create offspring by selecting parents and applying mutation/crossover
    # For simplicity in this loop, we'll select parents with replacement
    # and apply mutation. Crossover will be integrated in the next step.
    for i in range(pop_size):
        # Select a parent based on probabilities
        parent_index = np.random.choice(range(len(population)), p=selection_probs)
        parent = population[parent_index]

        # Create offspring blueprint (simple copy for now, crossover later)
        offspring_blueprint = parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints


        # Create new SimulatedBeing instance for offspring
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)


        # Apply mutation to the offspring, passing population statistics
        offspring.mutate(
            population_average_fitness = population_average_fitness,
            population_trait_variance = population_trait_variance,
            population_trait_averages = population_trait_averages
        )

        new_population.append(offspring)


    return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
        }


    phenotype_keys = list(population[0].phenotype.__dict__.keys())

    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

    return generation_stats

# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    # Convert Phenotype object to a dictionary for plotting
    phenotype_dict = phenotype_data.__dict__
    labels = list(phenotype_dict.keys())
    values = list(phenotype_dict.values())

    num_vars = len(labels)
    # Compute angle each trait goes to in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

    # Plot data and fill area
    ax.fill(angles, values, color = 'red', alpha = 0.25)
    ax.plot(angles, values, color = 'red', linewidth = 2)

    # Set the grid and labels
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set title
    ax.set_title(title, size = 16, color = 'blue', y = 1.1)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f" {filename}.png"))
    plt.close(fig) # Close the figure to free up memory


def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key = lambda being: being.fitness)


# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

## Update the evolutionary loop

### Subtask:
Modify the existing `evolve_one_generation` function ( or create a new simulation class) to use the `SimulatedBeing` class and incorporate the new mutation and recombination methods.

**Reasoning**:
Modify the `evolve_one_generation` function to correctly implement the parent selection, crossover, and mutation steps using the `SimulatedBeing` class and the `uniform_crossover` method, ensuring population statistics are calculated and passed to the `mutate` method.

from typing import List, Dict, Any
import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Phenotype:
    """Represents the observable characteristics of a simulated being."""
    resource_efficiency: float = 0.0
    knowledge_exchange: float = 0.0
    structural_resilience: float = 0.0

class PhenotypeTranslator:
    """Translates a blueprint dictionary into a Phenotype object based on defined rules."""

    def __init__(self, blueprint_to_phenotype_rules: dict):
        """
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
        self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

    def translate(self, blueprint: dict) -> Phenotype:
        """
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
        phenotype_values = {}

        # Apply translation rules to calculate phenotype trait values
        for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
            calculated_value = 0.0
            for rule in rules:
                component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
                calculated_value += component_value * rule['weight']

            # Apply tanh normalization
            phenotype_values[phenotype_trait] = np.tanh(calculated_value)

        # Create and return a Phenotype object
        return Phenotype(
            resource_efficiency=phenotype_values.get("resource_efficiency", 0.0),
            knowledge_exchange=phenotype_values.get("knowledge_exchange", 0.0),
            structural_resilience=phenotype_values.get("structural_resilience", 0.0)
        )

class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    def __init__(self, blueprint: dict, phenotype_translator: PhenotypeTranslator):
        """
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
        self.blueprint = blueprint
        self.phenotype_translator = phenotype_translator
        self.phenotype: Phenotype = self.translate_blueprint_to_phenotype()
        self.fitness: float = 0.0 # Initialize fitness
        # Add a base mutation rate to the being's attributes
        self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)


    def translate_blueprint_to_phenotype(self) -> Phenotype:
        """
        Translates the being's blueprint into its phenotype using the provided translator.
        """
        # Ensure all expected keys are present in the blueprint, using defaults if necessary
        # This part can be more sophisticated depending on the desired blueprint structure and defaults
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        # This ensures the translator always receives a complete blueprint structure
        complete_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


        return self.phenotype_translator.translate(self.blueprint)


    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None, mutation_std: float = 0.05):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
            mutation_std: The standard deviation of the random noise added during mutation.
        """
        current_mutation_rate = self.base_mutation_rate
        # mutation_strength = 0.05 # Base strength of mutation - now passed as mutation_std

        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value, rate, strength):
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_std # Use passed mutation_std

                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages are low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint and the translator
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
        return offspring

# Helper function for dynamic fitness (Placeholder - replace with actual logic)
def dynamic_fitness(phenotype: Phenotype, generation: int, env_factor: float) -> Dict[str, float]:
    """
    Calculates dynamic fitness weights based on phenotype, generation, and environmental factor.

    Args:
        phenotype: The Phenotype object of the being.
        generation: The current generation number.
        env_factor: A factor representing environmental pressure.

    Returns:
        A dictionary of weights for each phenotype trait.
    """
    # Example dynamic fitness: favor resource efficiency early, then structural resilience
    if generation < 25:
        weights = {
            "resource_efficiency": 0.4 + 0.1 * env_factor,
            "knowledge_exchange": 0.4 - 0.05 * env_factor,
            "structural_resilience": 0.2 + 0.05 * env_factor
        }
    else:
        weights = {
            "resource_efficiency": 0.2 - 0.05 * env_factor,
            "knowledge_exchange": 0.2 + 0.05 * env_factor,
            "structural_resilience": 0.6 + 0.1 * env_factor
        }

    # Ensure weights sum to 1 (optional, depending on desired fitness scaling)
    # total_weight = sum(weights.values())
    # weights = {k: v / total_weight for k, v in weights.items()}

    return weights

# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp=0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    # Calculate fitness for each being if not already done
    fitnesses = np.array([being.fitness for being in population], dtype=float)

    # Numeric stable softmax
    scaled_fitness = np.array(fitnesses) / float(temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    return probs


def evolve_one_generation(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_std: float, env_factor: float, phenotype_translator: PhenotypeTranslator) -> List[SimulatedBeing]:
    """
    Simulates one generation of evolution.

    Args:
        population (List[SimulatedBeing]): The current population of beings.
        pop_size (int): The target size of the new population.
        generation (int): The current generation number, used for dynamic fitness.
        selection_temp (float): The temperature for softmax selection. Lower values
                                 lead to more "greedy" selection.
        mutation_std (float): The standard deviation of the random noise added during mutation.
        env_factor (float): A factor representing environmental pressure on certain traits.
        phenotype_translator: An instance of PhenotypeTranslator.


    Returns:
        List[SimulatedBeing]: The new population of beings after one generation of evolution.
    """
    if not population:
        return []

    # 1. Evaluate Fitness for all beings in the population using dynamic fitness
    # Define dynamic fitness weights based on the generation number and env_factor
    # This logic is now within the dynamic_fitness helper function

    for being in population:
        # Call dynamic_fitness to get the weights for this generation
        current_fitness_weights = dynamic_fitness(being.phenotype, generation, env_factor)
        being.fitness = being.calculate_fitness(weights=current_fitness_weights)


    # Get all fitness values to calculate selection probabilities
    fitnesses = [b.fitness for b in population]

    # Softmax for selection probability. Lower temp leads to more greedy selection.
    scaled_fitness = np.array(fitnesses) / float(selection_temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    # 2. Select Parents based on fitness probabilities
    # Select pop_size * 2 indices as we need two parents per offspring for crossover
    parent_indices = np.random.choice(
        range(len(population)),
        size=pop_size, # Select enough parents to create a full population
        p=probs,
        replace=True # Allows a highly-fit parent to be selected multiple times
    )


    # 3. Create the New Generation through Crossover & Mutation
    new_population = []
    # We'll create offspring in pairs for crossover, so we need pop_size // 2 pairs
    # Ensure pop_size is even or handle the last offspring separately if odd
    num_offspring_pairs = pop_size // 2
    remaining_offspring = pop_size % 2 # For odd pop_size

    # Select enough parents for crossover pairs
    crossover_parent_indices = np.random.choice(
        range(len(population)),
        size=num_offspring_pairs * 2,
        p=probs,
        replace=True
    )


    # Calculate population statistics *before* mutation for adaptive mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    population_average_fitness = np.mean(fitnesses)
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    current_pop_stats = {
        "average_fitness": population_average_fitness,
        "trait_averages": population_trait_averages,
        "trait_variance": population_trait_variance
    }


    for i in range(0, len(crossover_parent_indices), 2):
        parent1_index = crossover_parent_indices[i]
        parent2_index = crossover_parent_indices[i+1]

        parent1 = population[parent1_index]
        parent2 = population[parent2_index]

        # Perform crossover to create offspring
        # Pass the phenotype_translator to the crossover method
        offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

        # Calculate fitness for the newly created offspring using the current generation's weights
        # This ensures self.fitness is available for fitness-dependent mutation
        # Call dynamic_fitness to get the weights for the offspring's fitness calculation
        offspring_fitness_weights = dynamic_fitness(offspring.phenotype, generation, env_factor)
        offspring.calculate_fitness(weights=offspring_fitness_weights)

        # Apply mutation to the offspring, passing population statistics and mutation_std
        offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std # Pass mutation_std
        )

        new_population.append(offspring)

    # If pop_size is odd, select one more parent and create one more offspring (mutation only)
    if remaining_offspring > 0:
        single_parent_index = np.random.choice(range(len(population)), p=probs)
        single_parent = population[single_parent_index]
        # Create offspring by cloning and mutating (no crossover)
        # This assumes blueprint can be copied and then mutated
        single_offspring_blueprint = single_parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints
        single_offspring = SimulatedBeing(single_offspring_blueprint, phenotype_translator)

        # Calculate fitness for the single offspring
        single_offspring_fitness_weights = dynamic_fitness(single_offspring.phenotype, generation, env_factor)
        single_offspring.calculate_fitness(weights=single_offspring_fitness_weights)

        single_offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std
        )
        new_population.append(single_offspring)


    return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
        }

    phenotype_keys = list(population[0].phenotype.__dict__.keys())

    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

    return generation_stats

# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    # Convert Phenotype object to a dictionary for plotting
    phenotype_dict = phenotype_data.__dict__
    labels = list(phenotype_dict.keys())
    values = list(phenotype_dict.values())

    num_vars = len(labels)
    # Compute angle each trait goes to in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

    # Plot data and fill area
    ax.fill(angles, values, color = 'red', alpha = 0.25)
    ax.plot(angles, values, color = 'red', linewidth = 2)

    # Set the grid and labels
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set title
    ax.set_title(title, size = 16, color = 'blue', y = 1.1)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f" {filename}.png"))
    plt.close(fig) # Close the figure to free up memory


def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key = lambda being: being.fitness)

# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

# --- Run the Simulation ---
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Simulation parameters for evolve_population
selection_temp = 0.1
mutation_std = 0.05 # Standard deviation for mutation noise
env_factor = 0.5 # Example environmental factor

# Create the PhenotypeTranslator instance once
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

translator = PhenotypeTranslator(sample_translation_rules)

# Generate the initial population using the updated function
population = generate_initial_population(size=pop_size, translator=translator)

# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []

for gen in range(num_generations):
    # Evolve the population using the updated function with dynamic weights and the translator
    # The dynamic fitness calculation is now handled within evolve_population by calling dynamic_fitness
    population = evolve_one_generation(
        population,
        pop_size=pop_size,
        generation=gen,
        selection_temp=selection_temp,
        mutation_std=mutation_std,
        env_factor=env_factor,
        phenotype_translator=translator # Pass the translator
    )

    # Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    avg_phenotype = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    avg_phenotypes_history.append(avg_phenotype)

    # Track and store population statistics (fitness is already calculated in evolve_population)
    gen_stats = track_population_statistics(population, gen)
    population_statistics_history.append(gen_stats)

    # Visualize the fittest individual and sampled individuals
    fittest_individual = get_fittest_individual(population)
    if fittest_individual:
        create_radar_plot(fittest_individual.phenotype,
                          f"Fittest Phenotype - Gen {gen}",
                          f"fittest_phenotype_gen_{gen}")

    if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
        sampled_individual = random.choice(population)
        create_radar_plot(sampled_individual.phenotype,
                          f"Sampled Phenotype - Gen {gen}",
                          f"sampled_phenotype_gen_{gen}")


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label="Resource Efficiency", color="green")
ax.plot(gens, knowledge_ex, label="Knowledge Exchange", color="blue")
ax.plot(gens, structural_res, label="Structural Resilience", color="red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label="Resource Efficiency Std Dev", color="green", linestyle='--')
ax_std.plot(gens, knowledge_ex_std, label="Knowledge Exchange Std Dev", color="blue", linestyle='--')
ax_std.plot(gens, structural_res_std, label="Structural Resilience Std Dev", color="red", linestyle='--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")

## Implement dynamic fitness calculation

### Subtask:
Modify the `evolve_one_generation` function to accept dynamic fitness weights and update the fitness calculation within the loop.

**Reasoning**:
Modify the `evolve_one_generation` function to calculate dynamic fitness weights using the `dynamic_fitness` helper function and then use these weights when calling the `calculate_fitness` method for each `SimulatedBeing` instance.

from typing import List, Dict, Any
import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Phenotype:
    """Represents the observable characteristics of a simulated being."""
    resource_efficiency: float = 0.0
    knowledge_exchange: float = 0.0
    structural_resilience: float = 0.0

class PhenotypeTranslator:
    """Translates a blueprint dictionary into a Phenotype object based on defined rules."""

    def __init__(self, blueprint_to_phenotype_rules: dict):
        """
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
        self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

    def translate(self, blueprint: dict) -> Phenotype:
        """
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
        phenotype_values = {}

        # Apply translation rules to calculate phenotype trait values
        for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
            calculated_value = 0.0
            for rule in rules:
                component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
                calculated_value += component_value * rule['weight']

            # Apply tanh normalization
            phenotype_values[phenotype_trait] = np.tanh(calculated_value)

        # Create and return a Phenotype object
        return Phenotype(
            resource_efficiency=phenotype_values.get("resource_efficiency", 0.0),
            knowledge_exchange=phenotype_values.get("knowledge_exchange", 0.0),
            structural_resilience=phenotype_values.get("structural_resilience", 0.0)
        )

class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    def __init__(self, blueprint: dict, phenotype_translator: PhenotypeTranslator):
        """
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
        self.blueprint = blueprint
        self.phenotype_translator = phenotype_translator
        self.phenotype: Phenotype = self.translate_blueprint_to_phenotype()
        self.fitness: float = 0.0 # Initialize fitness
        # Add a base mutation rate to the being's attributes
        self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)


    def translate_blueprint_to_phenotype(self) -> Phenotype:
        """
        Translates the being's blueprint into its phenotype using the provided translator.
        """
        # Ensure all expected keys are present in the blueprint, using defaults if necessary
        # This part can be more sophisticated depending on the desired blueprint structure and defaults
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        # This ensures the translator always receives a complete blueprint structure
        complete_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


        return self.phenotype_translator.translate(self.blueprint)


    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None, mutation_std: float = 0.05):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
            mutation_std: The standard deviation of the random noise added during mutation.
        """
        current_mutation_rate = self.base_mutation_rate
        # mutation_strength = 0.05 # Base strength of mutation - now passed as mutation_std

        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value, rate, strength):
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_std # Use passed mutation_std

                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages are low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint and the translator
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
        return offspring

# Helper function for dynamic fitness (Placeholder - replace with actual logic)
def dynamic_fitness(phenotype: Phenotype, generation: int, env_factor: float) -> Dict[str, float]:
    """
    Calculates dynamic fitness weights based on phenotype, generation, and environmental factor.

    Args:
        phenotype: The Phenotype object of the being.
        generation: The current generation number.
        env_factor: A factor representing environmental pressure.

    Returns:
        A dictionary of weights for each phenotype trait.
    """
    # Example dynamic fitness: favor resource efficiency early, then structural resilience
    if generation < 25:
        weights = {
            "resource_efficiency": 0.4 + 0.1 * env_factor,
            "knowledge_exchange": 0.4 - 0.05 * env_factor,
            "structural_resilience": 0.2 + 0.05 * env_factor
        }
    else:
        weights = {
            "resource_efficiency": 0.2 - 0.05 * env_factor,
            "knowledge_exchange": 0.2 + 0.05 * env_factor,
            "structural_resilience": 0.6 + 0.1 * env_factor
        }

    # Ensure weights sum to 1 (optional, depending on desired fitness scaling)
    # total_weight = sum(weights.values())
    # weights = {k: v / total_weight for k, v in weights.items()}

    return weights

# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp=0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    # Calculate fitness for each being if not already done
    fitnesses = np.array([being.fitness for being in population], dtype=float)

    # Numeric stable softmax
    scaled_fitness = np.array(fitnesses) / float(temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    return probs


def evolve_one_generation(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_std: float, env_factor: float, phenotype_translator: PhenotypeTranslator) -> List[SimulatedBeing]:
    """
    Simulates one generation of evolution.

    Args:
        population (List[SimulatedBeing]): The current population of beings.
        pop_size (int): The target size of the new population.
        generation (int): The current generation number, used for dynamic fitness.
        selection_temp (float): The temperature for softmax selection. Lower values
                                 lead to more "greedy" selection.
        mutation_std (float): The standard deviation of the random noise added during mutation.
        env_factor (float): A factor representing environmental pressure on certain traits.
        phenotype_translator: An instance of PhenotypeTranslator.


    Returns:
        List[SimulatedBeing]: The new population of beings after one generation of evolution.
    """
    if not population:
        return []

    # 1. Evaluate Fitness for all beings in the population using dynamic fitness
    # Define dynamic fitness weights based on the generation number and env_factor
    # This logic is now within the dynamic_fitness helper function

    for being in population:
        # Call dynamic_fitness to get the weights for this generation
        current_fitness_weights = dynamic_fitness(being.phenotype, generation, env_factor)
        being.fitness = being.calculate_fitness(weights=current_fitness_weights)


    # Get all fitness values to calculate selection probabilities
    fitnesses = [b.fitness for b in population]

    # Softmax for selection probability. Lower temp leads to more greedy selection.
    scaled_fitness = np.array(fitnesses) / float(selection_temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    # 2. Select Parents based on fitness probabilities
    # Select pop_size * 2 indices as we need two parents per offspring for crossover
    parent_indices = np.random.choice(
        range(len(population)),
        size=pop_size, # Select enough parents to create a full population
        p=probs,
        replace=True # Allows a highly-fit parent to be selected multiple times
    )


    # 3. Create the New Generation through Crossover & Mutation
    new_population = []
    # We'll create offspring in pairs for crossover, so we need pop_size // 2 pairs
    # Ensure pop_size is even or handle the last offspring separately if odd
    num_offspring_pairs = pop_size // 2
    remaining_offspring = pop_size % 2 # For odd pop_size

    # Select enough parents for crossover pairs
    crossover_parent_indices = np.random.choice(
        range(len(population)),
        size=num_offspring_pairs * 2,
        p=probs,
        replace=True
    )


    # Calculate population statistics *before* mutation for adaptive mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    population_average_fitness = np.mean(fitnesses)
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    current_pop_stats = {
        "average_fitness": population_average_fitness,
        "trait_averages": population_trait_averages,
        "trait_variance": population_trait_variance
    }


    for i in range(0, len(crossover_parent_indices), 2):
        parent1_index = crossover_parent_indices[i]
        parent2_index = crossover_parent_indices[i+1]

        parent1 = population[parent1_index]
        parent2 = population[parent2_index]

        # Perform crossover to create offspring
        # Pass the phenotype_translator to the crossover method
        offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

        # Calculate fitness for the newly created offspring using the current generation's weights
        # This ensures self.fitness is available for fitness-dependent mutation
        # Call dynamic_fitness to get the weights for the offspring's fitness calculation
        offspring_fitness_weights = dynamic_fitness(offspring.phenotype, generation, env_factor)
        offspring.calculate_fitness(weights=offspring_fitness_weights)

        # Apply mutation to the offspring, passing population statistics and mutation_std
        offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std # Pass mutation_std
        )

        new_population.append(offspring)

    # If pop_size is odd, select one more parent and create one more offspring (mutation only)
    if remaining_offspring > 0:
        single_parent_index = np.random.choice(range(len(population)), p=probs)
        single_parent = population[single_parent_index]
        # Create offspring by cloning and mutating (no crossover)
        # This assumes blueprint can be copied and then mutated
        single_offspring_blueprint = single_parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints
        single_offspring = SimulatedBeing(single_offspring_blueprint, phenotype_translator)

        # Calculate fitness for the single offspring
        single_offspring_fitness_weights = dynamic_fitness(single_offspring.phenotype, generation, env_factor)
        single_offspring.calculate_fitness(weights=single_offspring_fitness_weights)

        single_offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std
        )
        new_population.append(single_offspring)


    return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
        }

    phenotype_keys = list(population[0].phenotype.__dict__.keys())

    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

    return generation_stats

# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    # Convert Phenotype object to a dictionary for plotting
    phenotype_dict = phenotype_data.__dict__
    labels = list(phenotype_dict.keys())
    values = list(phenotype_dict.values())

    num_vars = len(labels)
    # Compute angle each trait goes to in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

    # Plot data and fill area
    ax.fill(angles, values, color = 'red', alpha = 0.25)
    ax.plot(angles, values, color = 'red', linewidth = 2)

    # Set the grid and labels
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set title
    ax.set_title(title, size = 16, color = 'blue', y = 1.1)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f" {filename}.png"))
    plt.close(fig) # Close the figure to free up memory


def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key = lambda being: being.fitness)

# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

# --- Run the Simulation ---
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Simulation parameters for evolve_population
selection_temp = 0.1
mutation_std = 0.05 # Standard deviation for mutation noise
env_factor = 0.5 # Example environmental factor

# Create the PhenotypeTranslator instance once
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

translator = PhenotypeTranslator(sample_translation_rules)

# Generate the initial population using the updated function
population = generate_initial_population(size=pop_size, translator=translator)

# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []

for gen in range(num_generations):
    # Evolve the population using the updated function with dynamic weights and the translator
    # The dynamic fitness calculation is now handled within evolve_population by calling dynamic_fitness
    population = evolve_one_generation(
        population,
        pop_size=pop_size,
        generation=gen,
        selection_temp=selection_temp,
        mutation_std=mutation_std,
        env_factor=env_factor,
        phenotype_translator=translator # Pass the translator
    )

    # Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    avg_phenotype = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    avg_phenotypes_history.append(avg_phenotype)

    # Track and store population statistics (fitness is already calculated in evolve_population)
    gen_stats = track_population_statistics(population, gen)
    population_statistics_history.append(gen_stats)

    # Visualize the fittest individual and sampled individuals
    fittest_individual = get_fittest_individual(population)
    if fittest_individual:
        create_radar_plot(fittest_individual.phenotype,
                          f"Fittest Phenotype - Gen {gen}",
                          f"fittest_phenotype_gen_{gen}")

    if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
        sampled_individual = random.choice(population)
        create_radar_plot(sampled_individual.phenotype,
                          f"Sampled Phenotype - Gen {gen}",
                          f"sampled_phenotype_gen_{gen}")


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label="Resource Efficiency", color="green")
ax.plot(gens, knowledge_ex, label="Knowledge Exchange", color="blue")
ax.plot(gens, structural_res, label="Structural Resilience", color="red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label="Resource Efficiency Std Dev", color="green", linestyle='--')
ax_std.plot(gens, knowledge_ex_std, label="Knowledge Exchange Std Dev", color="blue", linestyle='--')
ax_std.plot(gens, structural_res_std, label="Structural Resilience Std Dev", color="red", linestyle='--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")

## Track and analyze population statistics

### Subtask:
Implement tracking of population statistics (mean and std dev of traits, average fitness) over generations and add code to analyze or display these statistics.

**Reasoning**:
Add code to the main simulation loop to call the `track_population_statistics` helper function in each generation and store the results. Convert the collected statistics into a pandas DataFrame for easier analysis and display the first few rows of the DataFrame.

from typing import List, Dict, Any
import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Phenotype:
    """Represents the observable characteristics of a simulated being."""
    resource_efficiency: float = 0.0
    knowledge_exchange: float = 0.0
    structural_resilience: float = 0.0

class PhenotypeTranslator:
    """Translates a blueprint dictionary into a Phenotype object based on defined rules."""

    def __init__(self, blueprint_to_phenotype_rules: dict):
        """
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
        self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

    def translate(self, blueprint: dict) -> Phenotype:
        """
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
        phenotype_values = {}

        # Apply translation rules to calculate phenotype trait values
        for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
            calculated_value = 0.0
            for rule in rules:
                component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
                calculated_value += component_value * rule['weight']

            # Apply tanh normalization
            phenotype_values[phenotype_trait] = np.tanh(calculated_value)

        # Create and return a Phenotype object
        return Phenotype(
            resource_efficiency=phenotype_values.get("resource_efficiency", 0.0),
            knowledge_exchange=phenotype_values.get("knowledge_exchange", 0.0),
            structural_resilience=phenotype_values.get("structural_resilience", 0.0)
        )

class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    def __init__(self, blueprint: dict, phenotype_translator: PhenotypeTranslator):
        """
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
        self.blueprint = blueprint
        self.phenotype_translator = phenotype_translator
        self.phenotype: Phenotype = self.translate_blueprint_to_phenotype()
        self.fitness: float = 0.0 # Initialize fitness
        # Add a base mutation rate to the being's attributes
        self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)


    def translate_blueprint_to_phenotype(self) -> Phenotype:
        """
        Translates the being's blueprint into its phenotype using the provided translator.
        """
        # Ensure all expected keys are present in the blueprint, using defaults if necessary
        # This part can be more sophisticated depending on the desired blueprint structure and defaults
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        # This ensures the translator always receives a complete blueprint structure
        complete_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


        return self.phenotype_translator.translate(self.blueprint)


    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None, mutation_std: float = 0.05):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
            mutation_std: The standard deviation of the random noise added during mutation.
        """
        current_mutation_rate = self.base_mutation_rate
        # mutation_strength = 0.05 # Base strength of mutation - now passed as mutation_std

        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value, rate, strength):
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_std # Use passed mutation_std

                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages is low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint and the translator
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
        return offspring

# Helper function for dynamic fitness (Placeholder - replace with actual logic)
def dynamic_fitness(phenotype: Phenotype, generation: int, env_factor: float) -> Dict[str, float]:
    """
    Calculates dynamic fitness weights based on phenotype, generation, and environmental factor.

    Args:
        phenotype: The Phenotype object of the being.
        generation: The current generation number.
        env_factor: A factor representing environmental pressure.

    Returns:
        A dictionary of weights for each phenotype trait.
    """
    # Example dynamic fitness: favor resource efficiency early, then structural resilience
    if generation < 25:
        weights = {
            "resource_efficiency": 0.4 + 0.1 * env_factor,
            "knowledge_exchange": 0.4 - 0.05 * env_factor,
            "structural_resilience": 0.2 + 0.05 * env_factor
        }
    else:
        weights = {
            "resource_efficiency": 0.2 - 0.05 * env_factor,
            "knowledge_exchange": 0.2 + 0.05 * env_factor,
            "structural_resilience": 0.6 + 0.1 * env_factor
        }

    # Ensure weights sum to 1 (optional, depending on desired fitness scaling)
    # total_weight = sum(weights.values())
    # weights = {k: v / total_weight for k, v in weights.items()}

    return weights

# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp=0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    # Calculate fitness for each being if not already done
    fitnesses = np.array([being.fitness for being in population], dtype=float)

    # Numeric stable softmax
    scaled_fitness = np.array(fitnesses) / float(temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    return probs


def evolve_one_generation(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_std: float, env_factor: float, phenotype_translator: PhenotypeTranslator) -> List[SimulatedBeing]:
    """
    Simulates one generation of evolution.

    Args:
        population (List[SimulatedBeing]): The current population of beings.
        pop_size (int): The target size of the new population.
        generation (int): The current generation number, used for dynamic fitness.
        selection_temp (float): The temperature for softmax selection. Lower values
                                 lead to more "greedy" selection.
        mutation_std (float): The standard deviation of the random noise added during mutation.
        env_factor (float): A factor representing environmental pressure on certain traits.
        phenotype_translator: An instance of PhenotypeTranslator.


    Returns:
        List[SimulatedBeing]: The new population of beings after one generation of evolution.
    """
    if not population:
        return []

    # 1. Evaluate Fitness for all beings in the population using dynamic fitness
    # Define dynamic fitness weights based on the generation number and env_factor
    # This logic is now within the dynamic_fitness helper function

    for being in population:
        # Call dynamic_fitness to get the weights for this generation
        current_fitness_weights = dynamic_fitness(being.phenotype, generation, env_factor)
        being.fitness = being.calculate_fitness(weights=current_fitness_weights)


    # Get all fitness values to calculate selection probabilities
    fitnesses = [b.fitness for b in population]

    # Softmax for selection probability. Lower temp leads to more greedy selection.
    scaled_fitness = np.array(fitnesses) / float(selection_temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    # 2. Select Parents based on fitness probabilities
    # Select pop_size * 2 indices as we need two parents per offspring for crossover
    parent_indices = np.random.choice(
        range(len(population)),
        size=pop_size, # Select enough parents to create a full population
        p=probs,
        replace=True # Allows a highly-fit parent to be selected multiple times
    )


    # 3. Create the New Generation through Crossover & Mutation
    new_population = []
    # We'll create offspring in pairs for crossover, so we need pop_size // 2 pairs
    # Ensure pop_size is even or handle the last offspring separately if odd
    num_offspring_pairs = pop_size // 2
    remaining_offspring = pop_size % 2 # For odd pop_size

    # Select enough parents for crossover pairs
    crossover_parent_indices = np.random.choice(
        range(len(population)),
        size=num_offspring_pairs * 2,
        p=probs,
        replace=True
    )


    # Calculate population statistics *before* mutation for adaptive mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    population_average_fitness = np.mean(fitnesses)
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    current_pop_stats = {
        "average_fitness": population_average_fitness,
        "trait_averages": population_trait_averages,
        "trait_variance": population_trait_variance
    }


    for i in range(0, len(crossover_parent_indices), 2):
        parent1_index = crossover_parent_indices[i]
        parent2_index = crossover_parent_indices[i+1]

        parent1 = population[parent1_index]
        parent2 = population[parent2_index]

        # Perform crossover to create offspring
        # Pass the phenotype_translator to the crossover method
        offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

        # Calculate fitness for the newly created offspring using the current generation's weights
        # This ensures self.fitness is available for fitness-dependent mutation
        # Call dynamic_fitness to get the weights for the offspring's fitness calculation
        offspring_fitness_weights = dynamic_fitness(offspring.phenotype, generation, env_factor)
        offspring.calculate_fitness(weights=offspring_fitness_weights)

        # Apply mutation to the offspring, passing population statistics and mutation_std
        offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std # Pass mutation_std
        )

        new_population.append(offspring)

    # If pop_size is odd, select one more parent and create one more offspring (mutation only)
    if remaining_offspring > 0:
        single_parent_index = np.random.choice(range(len(population)), p=probs)
        single_parent = population[single_parent_index]
        # Create offspring by cloning and mutating (no crossover)
        # This assumes blueprint can be copied and then mutated
        single_offspring_blueprint = single_parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints
        single_offspring = SimulatedBeing(single_offspring_blueprint, phenotype_translator)

        # Calculate fitness for the single offspring
        single_offspring_fitness_weights = dynamic_fitness(single_offspring.phenotype, generation, env_factor)
        single_offspring.calculate_fitness(weights=single_offspring_fitness_weights)

        single_offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std
        )
        new_population.append(single_offspring)


    return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
        }

    phenotype_keys = list(population[0].phenotype.__dict__.keys())

    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

    return generation_stats

# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    # Convert Phenotype object to a dictionary for plotting
    phenotype_dict = phenotype_data.__dict__
    labels = list(phenotype_dict.keys())
    values = list(phenotype_dict.values())

    num_vars = len(labels)
    # Compute angle each trait goes to in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

    # Plot data and fill area
    ax.fill(angles, values, color = 'red', alpha = 0.25)
    ax.plot(angles, values, color = 'red', linewidth = 2)

    # Set the grid and labels
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set title
    ax.set_title(title, size = 16, color = 'blue', y = 1.1)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f" {filename}.png"))
    plt.close(fig) # Close the figure to free up memory


def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key = lambda being: being.fitness)

# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

# --- Run the Simulation ---
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Simulation parameters for evolve_population
selection_temp = 0.1
mutation_std = 0.05 # Standard deviation for mutation noise
env_factor = 0.5 # Example environmental factor

# Create the PhenotypeTranslator instance once
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

translator = PhenotypeTranslator(sample_translation_rules)

# Generate the initial population using the updated function
population = generate_initial_population(size=pop_size, translator=translator)

# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []

for gen in range(num_generations):
    # Evolve the population using the updated function with dynamic weights and the translator
    # The dynamic fitness calculation is now handled within evolve_population by calling dynamic_fitness
    population = evolve_one_generation(
        population,
        pop_size=pop_size,
        generation=gen,
        selection_temp=selection_temp,
        mutation_std=mutation_std,
        env_factor=env_factor,
        phenotype_translator=translator # Pass the translator
    )

    # Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    avg_phenotype = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    avg_phenotypes_history.append(avg_phenotype)

    # Track and store population statistics (fitness is already calculated in evolve_population)
    gen_stats = track_population_statistics(population, gen)
    population_statistics_history.append(gen_stats)

    # Visualize the fittest individual and sampled individuals
    fittest_individual = get_fittest_individual(population)
    if fittest_individual:
        create_radar_plot(fittest_individual.phenotype,
                          f"Fittest Phenotype - Gen {gen}",
                          f"fittest_phenotype_gen_{gen}")

    if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
        sampled_individual = random.choice(population)
        create_radar_plot(sampled_individual.phenotype,
                          f"Sampled Phenotype - Gen {gen}",
                          f"sampled_phenotype_gen_{gen}")


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label="Resource Efficiency", color="green")
ax.plot(gens, knowledge_ex, label="Knowledge Exchange", color="blue")
ax.plot(gens, structural_res, label="Structural Resilience", color="red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label="Resource Efficiency Std Dev", color="green", linestyle='--')
ax_std.plot(gens, knowledge_ex_std, label="Knowledge Exchange Std Dev", color="blue", linestyle='--')
ax_std.plot(gens, structural_res_std, label="Structural Resilience Std Dev", color="red", linestyle='--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")

## Develop individual visualization

### Subtask:
Implement visualization of individual phenotypes using radar plots, focusing on the fittest individual and a few sampled individuals from different generations.

**Reasoning**:
Integrate the `create_radar_plot` function into the main simulation loop to visualize the phenotypes of the fittest individual and a few sampled individuals at regular intervals and the final generation.

from typing import List, Dict, Any
import random
import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import os
import pandas as pd
from dataclasses import dataclass, field

@dataclass
class Phenotype:
    """Represents the observable characteristics of a simulated being."""
    resource_efficiency: float = 0.0
    knowledge_exchange: float = 0.0
    structural_resilience: float = 0.0

class PhenotypeTranslator:
    """Translates a blueprint dictionary into a Phenotype object based on defined rules."""

    def __init__(self, blueprint_to_phenotype_rules: dict):
        """
        Initializes the PhenotypeTranslator with translation rules.

        Args:
            blueprint_to_phenotype_rules: A dictionary defining how blueprint components
                                          map to phenotype traits.
        """
        self.blueprint_to_phenotype_rules = blueprint_to_phenotype_rules

    def translate(self, blueprint: dict) -> Phenotype:
        """
        Translates the given blueprint into a Phenotype object.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.

        Returns:
            A Phenotype object with calculated and normalized trait values.
        """
        phenotype_values = {}

        # Apply translation rules to calculate phenotype trait values
        for phenotype_trait, rules in self.blueprint_to_phenotype_rules.items():
            calculated_value = 0.0
            for rule in rules:
                component_value = blueprint.get(rule['layer'], {}).get(rule['component'], 0.0)
                calculated_value += component_value * rule['weight']

            # Apply tanh normalization
            phenotype_values[phenotype_trait] = np.tanh(calculated_value)

        # Create and return a Phenotype object
        return Phenotype(
            resource_efficiency=phenotype_values.get("resource_efficiency", 0.0),
            knowledge_exchange=phenotype_values.get("knowledge_exchange", 0.0),
            structural_resilience=phenotype_values.get("structural_resilience", 0.0)
        )

class SimulatedBeing:
    """Represents a single simulated being with its blueprint, phenotype, and fitness."""

    def __init__(self, blueprint: dict, phenotype_translator: PhenotypeTranslator):
        """
        Initializes a SimulatedBeing instance.

        Args:
            blueprint: A dictionary representing the being's genetic blueprint.
            phenotype_translator: An instance of PhenotypeTranslator.
        """
        self.blueprint = blueprint
        self.phenotype_translator = phenotype_translator
        self.phenotype: Phenotype = self.translate_blueprint_to_phenotype()
        self.fitness: float = 0.0 # Initialize fitness
        # Add a base mutation rate to the being's attributes
        self.base_mutation_rate: float = self.blueprint.get("mutation_parameters", {}).get("base_rate", 0.01)


    def translate_blueprint_to_phenotype(self) -> Phenotype:
        """
        Translates the being's blueprint into its phenotype using the provided translator.
        """
        # Ensure all expected keys are present in the blueprint, using defaults if necessary
        # This part can be more sophisticated depending on the desired blueprint structure and defaults
        blueprint_copy = {
            "raw_materials": self.blueprint.get("raw_materials", {}),
            "core_components": self.blueprint.get("core_components", {}),
            "strategic_foundation": self.blueprint.get("strategic_foundation", {}),
            "unifying_strategy": self.blueprint.get("unifying_strategy", {}),
            "mutation_parameters": self.blueprint.get("mutation_parameters", {})
        }

        # Fill in missing blueprint components with random values for consistency
        # This ensures the translator always receives a complete blueprint structure
        complete_blueprint = {
            "raw_materials": {
                "material_a_ratio": blueprint_copy["raw_materials"].get("material_a_ratio", random.random()),
                "material_b_ratio": blueprint_copy["raw_materials"].get("material_b_ratio", random.random()),
                "processing_efficiency_factor": blueprint_copy["raw_materials"].get("processing_efficiency_factor", random.random()),
            },
            "core_components": {
                "component_x_strength": blueprint_copy["core_components"].get("component_x_strength", random.random()),
                "component_y_flexibility": blueprint_copy["core_components"].get("component_y_flexibility", random.random()),
                "interconnection_density": blueprint_copy["core_components"].get("interconnection_density", random.random()),
            },
            "strategic_foundation": {
                "adaptability_gene": blueprint_copy["strategic_foundation"].get("adaptability_gene", random.random()),
                "cooperation_gene": blueprint_copy["strategic_foundation"].get("cooperation_gene", random.random()),
            },
            "unifying_strategy": {
                "optimization_parameter": blueprint_copy["unifying_strategy"].get("optimization_parameter", random.random()),
            },
            "mutation_parameters": {
                "base_rate": blueprint_copy["mutation_parameters"].get("base_rate", 0.01)
            }
        }
        self.blueprint = complete_blueprint # Update the being's blueprint with defaults if any were missing


        return self.phenotype_translator.translate(self.blueprint)


    def calculate_fitness(self, weights: dict[str, float]) -> float:
        """
        Calculates the fitness of the being based on its phenotype and given weights.

        Args:
            weights: A dictionary of weights for each phenotype trait.

        Returns:
            The calculated fitness score.
        """
        # Ensure the phenotype has the expected keys based on the weights
        keys = list(weights.keys())
        if not all(hasattr(self.phenotype, k) for k in keys):
            raise ValueError(f"Phenotype is missing required keys for fitness calculation. Expected: {keys}, Has: {self.phenotype.__dict__.keys()}")


        self.fitness = sum(weights[key] * getattr(self.phenotype, key) for key in keys)
        return self.fitness


    def mutate(self, population_average_fitness: float = None, population_trait_variance: dict[str, float] = None, population_trait_averages: dict[str, float] = None, mutation_std: float = 0.05):
        """
        Mutates the being's blueprint based on adaptive and targeted strategies.

        Args:
            population_average_fitness: Average fitness of the population.
            population_trait_variance: Dictionary of trait variances in the population.
            population_trait_averages: Dictionary of trait averages in the population.
            mutation_std: The standard deviation of the random noise added during mutation.
        """
        current_mutation_rate = self.base_mutation_rate
        # mutation_strength = 0.05 # Base strength of mutation - now passed as mutation_std

        # 1. Fitness-dependent mutation: Increase mutation for less fit individuals
        if population_average_fitness is not None and self.fitness < population_average_fitness:
            current_mutation_rate *= 1.5 # Increase rate by 50% for less fit


        # Define a helper function for mutating a value
        def _apply_mutation(value, rate, strength):
            if random.random() < rate:
                # Ensure value is a number before adding random noise
                if isinstance(value, (int, float)):
                    return np.clip(value + np.random.normal(0, strength), 0, 1)
                else:
                    # Handle non-numeric types if necessary, or skip mutation
                    return value
            return value

        # 2. Apply mutation to blueprint components with adaptive/targeted strategies
        for layer, components in self.blueprint.items():
            if layer == "mutation_parameters": # Skip mutation parameters themselves
                continue

            for component, value in components.items():
                effective_mutation_rate = current_mutation_rate
                effective_mutation_strength = mutation_std # Use passed mutation_std

                # 3. Variance-based mutation: Increase mutation for traits with low population variance
                # This requires mapping blueprint components to phenotype traits, which is complex.
                # For this example, we'll apply variance-based mutation to all blueprint components
                # if overall population variance is low (simplification).
                # A more sophisticated approach would involve understanding which blueprint components
                # influence which phenotype traits.
                if population_trait_variance is not None:
                    # Check if any trait has low variance (example threshold 0.01)
                    if any(v < 0.01 for v in population_trait_variance.values()):
                        effective_mutation_rate *= 1.2 # Increase rate if population variance is low

                # 4. Targeted mutation based on population trait averages
                # This also requires mapping blueprint components to phenotype traits.
                # We'll apply targeted mutation based on overall low average phenotype scores (simplification).
                if population_trait_averages is not None:
                    # Check if any trait has a low average (example threshold 0.5)
                    if any(avg < 0.5 for avg in population_trait_averages.values()):
                        effective_mutation_rate *= 1.3 # Increase rate if population averages is low


                # Apply the mutation to the component value
                self.blueprint[layer][component] = _apply_mutation(value, effective_mutation_rate, effective_mutation_strength)

        # 5. Update the phenotype after mutation
        self.phenotype = self.translate_blueprint_to_phenotype()

    @staticmethod
    def uniform_crossover(parent1: 'SimulatedBeing', parent2: 'SimulatedBeing', phenotype_translator: PhenotypeTranslator) -> 'SimulatedBeing':
        """
        Performs uniform crossover between two parent blueprints to create an offspring blueprint.

        Args:
            parent1: The first parent SimulatedBeing instance.
            parent2: The second parent SimulatedBeing instance.
            phenotype_translator: An instance of PhenotypeTranslator to be used by the offspring.

        Returns:
            A new SimulatedBeing instance representing the offspring.
        """
        offspring_blueprint = {}

        # Iterate through each layer of the blueprint
        for layer in parent1.blueprint.keys():
            offspring_blueprint[layer] = {}
            # Iterate through each component within the layer
            for component in parent1.blueprint[layer].keys():
                # Randomly select the value from either parent
                if random.random() < 0.5:
                    offspring_blueprint[layer][component] = parent1.blueprint[layer][component]
                else:
                    offspring_blueprint[layer][component] = parent2.blueprint[layer][component]

        # Create a new SimulatedBeing instance with the offspring blueprint and the translator
        offspring = SimulatedBeing(offspring_blueprint, phenotype_translator)
        return offspring

# Helper function for dynamic fitness (Placeholder - replace with actual logic)
def dynamic_fitness(phenotype: Phenotype, generation: int, env_factor: float) -> Dict[str, float]:
    """
    Calculates dynamic fitness weights based on phenotype, generation, and environmental factor.

    Args:
        phenotype: The Phenotype object of the being.
        generation: The current generation number.
        env_factor: A factor representing environmental pressure.

    Returns:
        A dictionary of weights for each phenotype trait.
    """
    # Example dynamic fitness: favor resource efficiency early, then structural resilience
    if generation < 25:
        weights = {
            "resource_efficiency": 0.4 + 0.1 * env_factor,
            "knowledge_exchange": 0.4 - 0.05 * env_factor,
            "structural_resilience": 0.2 + 0.05 * env_factor
        }
    else:
        weights = {
            "resource_efficiency": 0.2 - 0.05 * env_factor,
            "knowledge_exchange": 0.2 + 0.05 * env_factor,
            "structural_resilience": 0.6 + 0.1 * env_factor
        }

    # Ensure weights sum to 1 (optional, depending on desired fitness scaling)
    # total_weight = sum(weights.values())
    # weights = {k: v / total_weight for k, v in weights.items()}

    return weights

# Helper function to calculate selection probabilities (adapted for SimulatedBeing)
def selection_probabilities(population: list[SimulatedBeing], temp=0.1):
    """
    Compute selection probabilities with softmax( fitness / temp ).
    Lower temp -> more greedy selection; higher temp -> more exploratory.
    """
    # Calculate fitness for each being if not already done
    fitnesses = np.array([being.fitness for being in population], dtype=float)

    # Numeric stable softmax
    scaled_fitness = np.array(fitnesses) / float(temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    return probs


def evolve_one_generation(population: List[SimulatedBeing], pop_size: int, generation: int, selection_temp: float, mutation_std: float, env_factor: float, phenotype_translator: PhenotypeTranslator) -> List[SimulatedBeing]:
    """
    Simulates one generation of evolution.

    Args:
        population (List[SimulatedBeing]): The current population of beings.
        pop_size (int): The target size of the new population.
        generation (int): The current generation number, used for dynamic fitness.
        selection_temp (float): The temperature for softmax selection. Lower values
                                 lead to more "greedy" selection.
        mutation_std (float): The standard deviation of the random noise added during mutation.
        env_factor (float): A factor representing environmental pressure on certain traits.
        phenotype_translator: An instance of PhenotypeTranslator.


    Returns:
        List[SimulatedBeing]: The new population of beings after one generation of evolution.
    """
    if not population:
        return []

    # 1. Evaluate Fitness for all beings in the population using dynamic fitness
    # Define dynamic fitness weights based on the generation number and env_factor
    # This logic is now within the dynamic_fitness helper function

    for being in population:
        # Call dynamic_fitness to get the weights for this generation
        current_fitness_weights = dynamic_fitness(being.phenotype, generation, env_factor)
        being.fitness = being.calculate_fitness(weights=current_fitness_weights)


    # Get all fitness values to calculate selection probabilities
    fitnesses = [b.fitness for b in population]

    # Softmax for selection probability. Lower temp leads to more greedy selection.
    scaled_fitness = np.array(fitnesses) / float(selection_temp)
    e = np.exp(scaled_fitness - np.max(scaled_fitness) + 1e-9) # Added small value for stability
    probs = e / np.sum(e)

    # Handle potential issues with probabilities (e.g., if all fitnesses are the same or very low)
    if np.sum(probs) == 0 or not np.isfinite(np.sum(probs)):
         probs = np.ones(len(population)) / len(population) # Assign equal probability if softmax fails


    # 2. Select Parents based on fitness probabilities
    # Select pop_size * 2 indices as we need two parents per offspring for crossover
    parent_indices = np.random.choice(
        range(len(population)),
        size=pop_size, # Select enough parents to create a full population
        p=probs,
        replace=True # Allows a highly-fit parent to be selected multiple times
    )


    # 3. Create the New Generation through Crossover & Mutation
    new_population = []
    # We'll create offspring in pairs for crossover, so we need pop_size // 2 pairs
    # Ensure pop_size is even or handle the last offspring separately if odd
    num_offspring_pairs = pop_size // 2
    remaining_offspring = pop_size % 2 # For odd pop_size

    # Select enough parents for crossover pairs
    crossover_parent_indices = np.random.choice(
        range(len(population)),
        size=num_offspring_pairs * 2,
        p=probs,
        replace=True
    )


    # Calculate population statistics *before* mutation for adaptive mutation
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    population_average_fitness = np.mean(fitnesses)
    population_trait_averages = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    population_trait_variance = {
        key: np.var([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    current_pop_stats = {
        "average_fitness": population_average_fitness,
        "trait_averages": population_trait_averages,
        "trait_variance": population_trait_variance
    }


    for i in range(0, len(crossover_parent_indices), 2):
        parent1_index = crossover_parent_indices[i]
        parent2_index = crossover_parent_indices[i+1]

        parent1 = population[parent1_index]
        parent2 = population[parent2_index]

        # Perform crossover to create offspring
        # Pass the phenotype_translator to the crossover method
        offspring = SimulatedBeing.uniform_crossover(parent1, parent2, phenotype_translator)

        # Calculate fitness for the newly created offspring using the current generation's weights
        # This ensures self.fitness is available for fitness-dependent mutation
        # Call dynamic_fitness to get the weights for the offspring's fitness calculation
        offspring_fitness_weights = dynamic_fitness(offspring.phenotype, generation, env_factor)
        offspring.calculate_fitness(weights=offspring_fitness_weights)

        # Apply mutation to the offspring, passing population statistics and mutation_std
        offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std # Pass mutation_std
        )

        new_population.append(offspring)

    # If pop_size is odd, select one more parent and create one more offspring (mutation only)
    if remaining_offspring > 0:
        single_parent_index = np.random.choice(range(len(population)), p=probs)
        single_parent = population[single_parent_index]
        # Create offspring by cloning and mutating (no crossover)
        # This assumes blueprint can be copied and then mutated
        single_offspring_blueprint = single_parent.blueprint.copy() # Simple copy, might need deepcopy for complex blueprints
        single_offspring = SimulatedBeing(single_offspring_blueprint, phenotype_translator)

        # Calculate fitness for the single offspring
        single_offspring_fitness_weights = dynamic_fitness(single_offspring.phenotype, generation, env_factor)
        single_offspring.calculate_fitness(weights=single_offspring_fitness_weights)

        single_offspring.mutate(
            population_average_fitness=current_pop_stats["average_fitness"],
            population_trait_variance=current_pop_stats["trait_variance"],
            population_trait_averages=current_pop_stats["trait_averages"],
            mutation_std=mutation_std
        )
        new_population.append(single_offspring)


    return new_population

# Helper function to track population statistics
def track_population_statistics(population: list[SimulatedBeing], generation: int) -> dict:
    """
    Calculates and returns key population statistics for a given generation.

    Args:
        population: The list of SimulatedBeing instances in the current generation.
        generation: The current generation number.

    Returns:
        A dictionary containing the generation number and calculated statistics.
    """
    if not population:
        return {
            "generation": generation, "trait_means": {}, "trait_stds": {}, "average_fitness": 0.0
        }

    phenotype_keys = list(population[0].phenotype.__dict__.keys())

    generation_stats = {
        "generation": generation,
        "trait_means": {},
        "trait_stds": {}
    }

    for trait_name in phenotype_keys:
        trait_values = [getattr(being.phenotype, trait_name) for being in population]
        generation_stats['trait_means'][trait_name] = np.mean(trait_values)
        generation_stats['trait_stds'][trait_name] = np.std(trait_values)

    generation_stats['average_fitness'] = np.mean([being.fitness for being in population])

    return generation_stats

# Helper function to create radar plots
def create_radar_plot(phenotype_data: Phenotype, title: str, filename: str, save_dir: str = "dna_evolution_output"):
    """
    Creates and saves a radar plot for a single phenotype.

    Args:
        phenotype_data: A Phenotype object.
        title: The title of the radar plot.
        filename: The base filename to save the plot.
        save_dir: The directory to save the plot files.
    """
    # Convert Phenotype object to a dictionary for plotting
    phenotype_dict = phenotype_data.__dict__
    labels = list(phenotype_dict.keys())
    values = list(phenotype_dict.values())

    num_vars = len(labels)
    # Compute angle each trait goes to in the plot
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values = values + values[:1]
    angles = angles + angles[:1]

    fig, ax = plt.subplots(figsize = (6, 6), subplot_kw = dict(polar = True))

    # Plot data and fill area
    ax.fill(angles, values, color = 'red', alpha = 0.25)
    ax.plot(angles, values, color = 'red', linewidth = 2)

    # Set the grid and labels
    ax.set_yticklabels([]) # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Set title
    ax.set_title(title, size = 16, color = 'blue', y = 1.1)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok = True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f" {filename}.png"))
    plt.close(fig) # Close the figure to free up memory


def get_fittest_individual(population: list[SimulatedBeing]) -> SimulatedBeing:
    """
    Finds and returns the individual with the highest fitness in the population.
    """
    if not population:
        return None
    return max(population, key = lambda being: being.fitness)

# --- Main Simulation Loop ---
# Initial population (generate a list of SimulatedBeing instances)
def generate_initial_population(size: int = 100, translator: PhenotypeTranslator = None) -> list[SimulatedBeing]:
    """
    Generates a list of SimulatedBeing instances with random initial blueprints.

    Args:
        size: The number of beings to generate.
        translator: The PhenotypeTranslator instance to use for each being.

    Returns:
        A list of SimulatedBeing instances.
    """
    if translator is None:
        raise ValueError("PhenotypeTranslator instance must be provided to generate_initial_population.")

    population = []
    for _ in range(size):
        # Create a sample blueprint for initial population with random values
        blueprint = {
            "raw_materials": {
                "material_a_ratio": random.random(),
                "material_b_ratio": random.random(),
                "processing_efficiency_factor": random.random(),
            },
            "core_components": {
                "component_x_strength": random.random(),
                "component_y_flexibility": random.random(),
                "interconnection_density": random.random(),
            },
            "strategic_foundation": {
                "adaptability_gene": random.random(),
                "cooperation_gene": random.random(),
            },
            "unifying_strategy": {
                "optimization_parameter": random.random(),
            },
            "mutation_parameters": {
                "base_rate": 0.01 # Initial base mutation rate
            }
        }
        # Create a SimulatedBeing instance and pass the translator
        population.append(SimulatedBeing(blueprint, translator))
    return population

# --- Run the Simulation ---
num_generations = 50
pop_size = 100 # Keep population size constant
sampling_interval = 10 # Sample individuals every 10 generations

# Simulation parameters for evolve_population
selection_temp = 0.1
mutation_std = 0.05 # Standard deviation for mutation noise
env_factor = 0.5 # Example environmental factor

# Create the PhenotypeTranslator instance once
sample_translation_rules = {
    "resource_efficiency": [
        {"layer": "raw_materials", "component": "material_a_ratio", "weight": 0.3},
        {"layer": "raw_materials", "component": "material_b_ratio", "weight": 0.4},
        {"layer": "raw_materials", "component": "processing_efficiency_factor", "weight": 0.3},
    ],
    "knowledge_exchange": [
        {"layer": "core_components", "component": "interconnection_density", "weight": 0.6},
        {"layer": "strategic_foundation", "component": "cooperation_gene", "weight": 0.4},
    ],
    "structural_resilience": [
        {"layer": "core_components", "component": "component_x_strength", "weight": 0.5},
        {"layer": "core_components", "component": "component_y_flexibility", "weight": 0.3},
        {"layer": "unifying_strategy", "component": "optimization_parameter", "weight": 0.2},
    ]
}

translator = PhenotypeTranslator(sample_translation_rules)

# Generate the initial population using the updated function
population = generate_initial_population(size=pop_size, translator=translator)

# Store the average phenotypes and all population statistics for plotting and analysis
avg_phenotypes_history = []
population_statistics_history = []

for gen in range(num_generations):
    # Evolve the population using the updated function with dynamic weights and the translator
    # The dynamic fitness calculation is now handled within evolve_population by calling dynamic_fitness
    population = evolve_one_generation(
        population,
        pop_size=pop_size,
        generation=gen,
        selection_temp=selection_temp,
        mutation_std=mutation_std,
        env_factor=env_factor,
        phenotype_translator=translator # Pass the translator
    )

    # Calculate and store the average phenotype for the current generation (for backward compatibility with plotting)
    phenotype_keys = list(population[0].phenotype.__dict__.keys())
    avg_phenotype = {
        key: np.mean([getattr(being.phenotype, key) for being in population]) for key in phenotype_keys
    }
    avg_phenotypes_history.append(avg_phenotype)

    # Track and store population statistics (fitness is already calculated in evolve_population)
    gen_stats = track_population_statistics(population, gen)
    population_statistics_history.append(gen_stats)

    # Visualize the fittest individual and sampled individuals
    fittest_individual = get_fittest_individual(population)
    if fittest_individual:
        create_radar_plot(fittest_individual.phenotype,
                          f"Fittest Phenotype - Gen {gen}",
                          f"fittest_phenotype_gen_{gen}")

    if gen % sampling_interval == 0 or gen == num_generations - 1: # Sample at intervals and the final generation
        sampled_individual = random.choice(population)
        create_radar_plot(sampled_individual.phenotype,
                          f"Sampled Phenotype - Gen {gen}",
                          f"sampled_phenotype_gen_{gen}")


# Convert population statistics history to a pandas DataFrame for easier analysis
stats_df = pd.DataFrame(population_statistics_history)

# Print the first few rows of the statistics DataFrame
print("\nPopulation Statistics History (first 5 generations):")
display(stats_df.head())

# Plot the evolution of average phenotypes (using the previously generated history)
fig, ax = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff = [p["resource_efficiency"] for p in avg_phenotypes_history]
knowledge_ex = [p["knowledge_exchange"] for p in avg_phenotypes_history]
structural_res = [p["structural_resilience"] for p in avg_phenotypes_history]

ax.plot(gens, resource_eff, label="Resource Efficiency", color="green")
ax.plot(gens, knowledge_ex, label="Knowledge Exchange", color="blue")
ax.plot(gens, structural_res, label="Structural Resilience", color="red")
ax.set_title("Evolution of Average Phenotype Traits Over Generations with Dynamic Fitness")
ax.set_xlabel("Generation")
ax.set_ylabel("Average Trait Value")
ax.legend()
ax.grid(True)
plt.show()

# Plot the evolution of trait standard deviations
fig_std, ax_std = plt.subplots(figsize=(8, 5))
gens = range(num_generations)
resource_eff_std = [s["trait_stds"]["resource_efficiency"] for s in population_statistics_history]
knowledge_ex_std = [s["trait_stds"]["knowledge_exchange"] for s in population_statistics_history]
structural_res_std = [s["trait_stds"]["structural_resilience"] for s in population_statistics_history]

ax_std.plot(gens, resource_eff_std, label="Resource Efficiency Std Dev", color="green", linestyle='--')
ax_std.plot(gens, knowledge_ex_std, label="Knowledge Exchange Std Dev", color="blue", linestyle='--')
ax_std.plot(gens, structural_res_std, label="Structural Resilience Std Dev", color="red", linestyle='--')
ax_std.set_title("Evolution of Phenotype Trait Standard Deviations Over Generations")
ax_std.set_xlabel("Generation")
ax_std.set_ylabel("Standard Deviation")
ax_std.legend()
ax_std.grid(True)
plt.show()

print("Simulation complete. Individual phenotype radar plots have been generated in 'dna_evolution_output' directory.")