import flapBird
import neuralNetwork
import os

path = "simulation_data/generation_"

if not os.path.exists("simulation_data"):
    os.makedirs("simulation_data")

# n° of entities at the end of generation, n° of generations, n° of new entities for each survivor, rate of mutation
number_survivors = 10
generations = 10
children = 100
mutation_rate = 0.1

population_size = number_survivors * children
layer_sizes = (2, 1)

population = neuralNetwork.generate_population(population_size, layer_sizes)
g = 1


while g < generations:

    survirvors, perfomance = flapBird.simulate(population, number_survivors)

    neuralNetwork.save_population(survirvors, number_survivors, perfomance, layer_sizes, path + f"{g}.json")
    population = neuralNetwork.new_generation(survirvors, perfomance, mutation_rate, children, layer_sizes)
    g += 1


