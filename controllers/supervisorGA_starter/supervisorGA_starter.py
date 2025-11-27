from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy, struct
import ga, os
import sys

import json
import gc

import pandas as pd

from ga_parameters import (
    NUM_GENERATIONS,
    POPULATION_SIZE,
    NUM_ELITE,
    INITIAL_ROT,
    INITIAL_TRANS,
    CROSSOVER_RATE,
    TOURNAMENT_K,
    MUTATION_RATE,
    EXPERIMENT_TIME
)

import time
class SupervisorGA:
    def __init__(self):
        # Simulation Parameters
        # Please, do not change these parameters
        self.time_step = 32  # ms
        self.time_experiment = EXPERIMENT_TIME  # s

        # Initiate Supervisor Module
        self.supervisor = Supervisor()
        # Check if the robot node exists in the current world file
        self.robot_node = self.supervisor.getFromDef("Controller")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
        # Get the robots translation and rotation current parameters
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")

        # Check Receiver and Emitter are enabled
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)

        # Initialize the receiver and emitter data to null
        self.receivedData = ""
        self.receivedWeights = ""
        self.receivedFitness = ""
        self.emitterData = ""

        ### Define here the GA Parameters
        self.num_generations = NUM_GENERATIONS
        self.num_population = POPULATION_SIZE
        self.num_elite = NUM_ELITE

        # size of the genotype variable
        self.num_weights = 0

        # Creating the initial population
        self.population = []

        # All Genotypes
        self.genotypes = []

        # Display: screen to plot the fitness values of the best individual and the average of the entire population
        self.display = self.supervisor.getDevice("display")
        self.width = self.display.getWidth()
        self.height = self.display.getHeight()
        self.prev_best_fitness = 0.0
        self.prev_average_fitness = 0.0
        self.display.drawText("Fitness (Best - Red)", 0, 0)
        self.display.drawText("Fitness (Average - Green)", 0, 10)

        self.data_foldername = ""

    def createRandomPopulation(self, genotype=None):
        # Wait until the supervisor receives the size of the genotypes (number of weights)
        if self.num_weights > 0:
            # Define the size of the population
            pop_size = (self.num_population, self.num_weights)
            # Create the initial population with random weights
            self.population = numpy.random.uniform(low=-1.0, high=1.0, size=pop_size)
        if genotype is not None:
            self.population[0] = genotype
            self.population[1] = genotype

    def handle_receiver(self):
        while self.receiver.getQueueLength() > 0:
            # Webots 2022:
            # self.receivedData = self.receiver.getData().decode("utf-8")
            # Webots 2023:
            self.receivedData = self.receiver.getString()
            typeMessage = self.receivedData[0:7]
            # Check Message
            if typeMessage == "weights":
                self.receivedWeights = self.receivedData[9 : len(self.receivedData)]
                self.num_weights = int(self.receivedWeights)
            elif typeMessage == "fitness":
                self.receivedFitness = float(
                    self.receivedData[9 : len(self.receivedData)]
                )
            elif typeMessage == "current":
                float_list_str = self.receivedData[9:].strip()
                # Split string by commas and convert each to float
                if float_list_str:
                    self.receivedCurrentFitness = [float(item) for item in float_list_str.split(',')]
                else:
                    self.receivedCurrentFitness = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                # print("Received Fitness:", self.receivedFitness)
            self.receiver.nextPacket()

    def handle_emitter(self):
        if self.num_weights > 0:
            # Send genotype of an individual
            string_message = str(self.emitterData)
            string_message = string_message.encode("utf-8")
            # print("Supervisor send:", string_message)
            self.emitter.send(string_message)

    def run_seconds(self, seconds):
        # print("Run Simulation")
        robot_state_list = []
        stop = int((seconds * 1000) / self.time_step)
        iterations = 0
        while self.supervisor.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            if stop == iterations:
                break
            iterations = iterations + 1
            # save robot position with its current fitness
            robot_pos = self.robot_node.getPosition()
            robot_orientation = self.robot_node.getOrientation()
            robot_state = {
                "fitness": round(self.receivedCurrentFitness[0], 2),
                "line_fitness": round(self.receivedCurrentFitness[1], 2),
                "collision_fitness": round(self.receivedCurrentFitness[2], 2),
                "forward_fitness": round(self.receivedCurrentFitness[3], 2),
                "follow_line_fitness": round(self.receivedCurrentFitness[4], 2),
                "avoid_collision_fitness": round(self.receivedCurrentFitness[5], 2),
                "spinning_fitness": round(self.receivedCurrentFitness[6], 2),
                "x": round(robot_pos[0], 2),
                "y": round(robot_pos[1], 2),
                "ox": round(robot_orientation[0], 2),
                "oy": round(robot_orientation[1], 2),
            }
            robot_state_list.append(robot_state)
        return robot_state_list

    def evaluate_genotype(self, genotype, generation, population):
        # Send genotype to robot for evaluation
        self.emitterData = str(genotype)

        # Reset robot position and physics
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()

        # Evaluation genotype
        robot_state_list = self.run_seconds(self.time_experiment)
        for state in robot_state_list:
            state["generation"] = generation
            state["population"] = population

        df = pd.DataFrame(robot_state_list)
        robot_pos_filename = self.data_foldername + "robot_position.csv"
        df.to_csv(robot_pos_filename, mode="a", index=False, header=not os.path.exists(robot_pos_filename))

        # Measure fitness
        fitness = self.receivedFitness
        print("{}.Fitness: {:.2f}".format(population, fitness))
        # current = (generation, genotype, fitness)
        # self.genotypes.append(current)

        # Store genome data
        df_genome = pd.DataFrame(
            {
                "generation": [generation],
                "population": [population],
                "fitness": [fitness],
                "genotype": [genotype.tolist()],
            }
        )

        genome_filename = self.data_foldername + "genome_data.csv"
        df_genome.to_csv(genome_filename, mode="a", index=False, header=not os.path.exists(genome_filename))
        os.makedirs(self.data_foldername + "genomes/genotype/gen{}".format(generation), exist_ok=True)
        numpy.save(
            self.data_foldername + "genomes/genotype/gen{}/g{}pop{}.npy".format(generation, generation, population),
            genotype,
        )

        del robot_state_list
        del df
        del df_genome
        gc.collect()

        return fitness

    def run_demo(self, robot_file="Best.npy"):
        print("Running Best Individual Demo ...\n")
        # Read File
        genotype = numpy.load(robot_file)
        # Send Genotype to controller
        self.emitterData = str(genotype)

        # Reset robot position and physics
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
        while self.num_weights == 0:
            self.handle_receiver()
            self.createRandomPopulation(genotype)

        # Evaluation genotype
        self.run_seconds(self.time_experiment)

        # Measure fitness
        fitness = self.receivedFitness
        print(f"X.Fitness: {fitness:.2f}")

    def run_optimization(self):
        # Wait until the number of weights is updated
        while self.num_weights == 0:
            self.handle_receiver()
            self.createRandomPopulation()

        print("starting GA optimization ...\n")
        # Time for filename
        self.data_foldername = "data/{}/".format(time.strftime("%Y%m%d-%H%M%S"))
        # store GA hyperparameters
        hyperparameters = {
            "num_generations": self.num_generations,
            "population_size": self.num_population,
            "num_elite": self.num_elite,
            "crossover_rate": CROSSOVER_RATE,
            "tournament_k": TOURNAMENT_K,
            "mutation_rate": MUTATION_RATE,
        }
        os.makedirs(self.data_foldername, exist_ok=True)
        os.makedirs(self.data_foldername + "genomes/", exist_ok=True)
        ga_params_filename = self.data_foldername + "ga_parameters.json"
        with open(ga_params_filename, "w") as f:
            json.dump(hyperparameters, f, indent=4)

        # For each Generation
        for generation in range(self.num_generations):
            print("\nGENERATION: {}".format(generation))
            current_population = []
            # Select each Genotype or Individual
            for population in range(self.num_population):
                genotype = self.population[population]
                # Evaluate
                fitness = self.evaluate_genotype(genotype, generation, population)
                # print(fitness)
                # Save its fitness value
                current_population.append((genotype, float(fitness)))
                # print(current_population)

            # After checking the fitness value of all indivuals
            # Save genotype of the best individual
            best = ga.getBestGenotype(current_population)
            average = ga.getAverageGenotype(current_population)
            numpy.save("Best.npy", best[0])
            self.plot_fitness(generation, best[1], average)
            # Generate the new population using genetic operators
            if generation < self.num_generations - 1:
                self.population = ga.population_reproduce(
                    current_population, self.num_elite
                )

        print(f"GA optimization terminated, saved data to {self.data_foldername}\n")

    def draw_scaled_line(self, generation, y1, y2):
        # Define the scale of the fitness plot
        XSCALE = int(self.width / self.num_generations)
        YSCALE = 100
        self.display.drawLine(
            (generation - 1) * XSCALE,
            self.height - int(y1 * YSCALE),
            generation * XSCALE,
            self.height - int(y2 * YSCALE),
        )

    def plot_fitness(self, generation, best_fitness, average_fitness):
        if generation > 0:
            self.display.setColor(0xFF0000)  # red
            self.draw_scaled_line(generation, self.prev_best_fitness, best_fitness)
            self.display.setColor(0x00FF00)  # green
            self.draw_scaled_line(
                generation, self.prev_average_fitness, average_fitness
            )

        self.prev_best_fitness = best_fitness
        self.prev_average_fitness = average_fitness


if __name__ == "__main__":
    # Call Supervisor function to initiate the supervisor module
    gaModel = SupervisorGA()

    # Function used to run the best individual or the GA
    keyboard = Keyboard()
    keyboard.enable(50)

    # Interface
    print("(R|r)un Best Individual or (S|s)earch for New Best Individual:")
    while gaModel.supervisor.step(gaModel.time_step) != -1:
        resp = keyboard.getKey()
        if resp == 83 or resp == 65619:  # S or s key
            gaModel.run_optimization()
            print(
                "Optimization: (R|r)un Best Individual or (S|s)earch for New Best Individual:"
            )
        elif resp == 82 or resp == 65619:  # R or r key
            gaModel.run_demo()
            print(
                "Demo: (R|r)un Best Individual or (S|s)earch for New Best Individual:"
            )
        elif resp == 81 or resp == 65619:  # Q or q key
            gaModel.run_demo("g199pop0.npy")
            print(
                "Reference: (R|r)un Best Individual or (S|s)earch for New Best Individual:"
            )
        elif resp == 80 or resp == 65619:  # P or p key
            gaModel.run_demo("bad_bf.npy")
            print(
                "Reference: (R|r)un Best Individual or (S|s)earch for New Best Individual:"
            )
