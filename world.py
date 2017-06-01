#!/usr/bin/env python
from cosc343world import Creature, World
import numpy as np
import time
import random
from operator import itemgetter, attrgetter, methodcaller
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# You can change this number to specify how many generations creatures are going to evolve over...
numGenerations = 300

# You can change this number to specify how many turns in simulation of the world for given generation
numTurns = 100

# You can change this number to change the percept format.  You have three choice - format 0, 1 and 2 (described in
# the assignment 2 pdf document)
perceptFormat=1

# You can change this number to chnage the world size
gridSize=24

# You can set this mode to True to have same initial conditions for each simulation in each generation.  Good
# for development, when you want to have some determinism in how the world runs from generatin to generation.
repeatableMode=False

# Holds data for graphing
avg_life = []
avgfitness = []

# This is a class implementing your creature a.k.a MyCreature.  It extends the basic Creature, which provides the
# basic functionality of the creature for the world simulation.  Your job is to implement the AgentFunction
# that controls creature's behavoiur by producing actions in respons to percepts.
class MyCreature(Creature):

    # Initialisation function.  This is where your creature
    # should be initialised with a chromosome in random state.
    # You need to decide the format of your
    # chromosome and the model that it's going to give rise to
    #
    # Input: numPercepts - the size of percepts list that creature will receive in each turn
    #        numActions - the size of actions list that creature must create on each turn
    # 27 11
    def __init__(self, numPercepts, numActions):

        # Place your initialisation code here.  Ideally this should set up the creature's chromosome
        # and set it to some random state.

        # MMA | Monster move away
        # MMC | Monster move closer
        # CMA | Creature move away
        # CMC | Creature move closer
        # FMA | Food move away
        # FMC | Food move closer
        # EAT | Eat square
        # RND | Random move
        self.chromosome = []

        # How well the chromosome performed
        self.fitness = 0

        for weight in range(0, 8):
            self.chromosome.append(round(random.uniform(0.0, 0.9), 2))

        # Do not remove this line at the end.  It calls constructors
        # of the parent classes.
        Creature.__init__(self)

    # This is the implementation of the agent function that is called on every turn, giving your
    # creature a chance to perform an action.  You need to implement a model here, that takes its
    # parameters
    # from the chromosome and it produces a set of actions from provided percepts
    #
    # Input: percepts - a list of percepts
    #        numAction - the size of the actions list that needs to be returned
    def AgentFunction(self, percepts, numActions):

        # At the moment the actions is a list of random numbers.  You need to
        # replace this with some model that maps percepts to actions.  The model
        # should be parametrised by the chromosome
        movement = [0]*(numActions-2)
        decide = [0]*(numActions-9)
        actions = []

        monsters = percepts[:9]
        creatures = percepts[9:18]
        foods = percepts[18:]

        for index in range(0, 8):
            if sum(percepts) != 0:
                # Knowing if to eat or not
                if 8-index == 4:
                    decide[0] += self.chromosome[6]
                else:
                    # Knowing when to move away to a monster
                    if monsters[index] > 0:
                        movement[8-index] += self.chromosome[0]

                    # Knowing when to move closer from a moster
                    if monsters[index] > 0:
                        movement[index] += self.chromosome[1]

                    # Knowing when to move away to a creature
                    if creatures[index] > 0:
                        movement[8-index] += self.chromosome[2]

                    # Knowing when to move closer from a creature
                    if creatures[index] > 0:
                        movement[index] += self.chromosome[3]

                    # Knowing when to move away to a food
                    if foods[index] > 0:
                        movement[8-index] += self.chromosome[4]

                    # Knowing when to move closer from a food
                    if foods[index] > 0:
                        movement[index] += self.chromosome[5]

            else:
                # Knowing when to move random
                decide[1] += self.chromosome[7]

        actions = movement + decide

        return actions


# This function is called after every simulation, passing a list of the old population of
# creatures whose fitness
# you need to evaluate and whose chromosomes you can use to create new creatures.
#
# Input: old_population - list of objects of MyCreature type that participated in the last
# simulation.
# You can query the state of the creatures by using some built-in methods as well as any methods
# you decide to add to MyCreature class.  The length of the list is the size of
# the population.  You need to generate a new population of the same size.  Creatures from
# old population can be used in the new population - simulation will reset them to starting
#                         state.
#
# Returns: a list of MyCreature objects of the same length as the old_population.
def newPopulation(old_population):
    global numTurns

    nSurvivors = 0
    avgLifeTime = 0
    fitnessScore = 0

    # For each individual you can extract the following information left over
    # from evaluation to let you figure out how well individual did in the
    # simulation of the world: whether the creature is dead or not, how much
    # energy did the creature have at the end of simualation (0 if dead), tick number
    # of creature's death (if dead).  You should use this information to build
    # a fitness function, score for how the individual did
    for individual in old_population:

        # You can read the creature's energy at the end of the simulation.
        energy = individual.getEnergy()

        # This method tells you if the creature died during the simulation
        dead = individual.isDead()

        # If the creature is dead, you can get its time of death (in turns)
        if dead:
            timeOfDeath = individual.timeOfDeath()
            avgLifeTime += timeOfDeath
            individual.fitness += 25
        else:
            individual.fitness += 50 + (individual.getEnergy() * 2)
            nSurvivors += 1
            avgLifeTime += numTurns
        
        if individual.timeOfDeath() > 50:
                individual.fitness += individual.timeOfDeath() - 25
        
        if individual.timeOfDeath() > 50:
                individual.fitness += individual.timeOfDeath()
        fitnessScore += (int(individual.fitness))


    # Here are some statistics, which you may or may not find useful
    avgLifeTime = float(avgLifeTime)/float(len(population))
    avgFitness = float(fitnessScore)/float(len(population))
    print("Simulation stats:")
    print("  Survivors    : %d out of %d" % (nSurvivors, len(population)))
    print("  Avg life time: %.1f turns" % avgLifeTime)
    print("  Avg fitness score: %d" % avgFitness)
    avg_life.append(avgLifeTime)
    avgfitness.append(avgFitness)

    # The information gathered above should allow you to build a fitness function that evaluates
    # fitness of
    # every creature.  You should show the average fitness, but also use the fitness for selecting
    # parents and
    # creating new creatures.

    # Based on the fitness you should select individuals for reproduction and create a
    # new population.  At the moment this is not done, and the same population with the same number
    # of individuals

    new_population = []

    # Picks two fit parents to make a child.
    def tournamentSelect(subset):

        hold = []

        for critter in subset:
            hold.append(critter)

        hold = sorted(hold, key=lambda Creature: Creature.fitness)
        return hold[-1], hold[-2]

    # Uses two parents and mergers their chromosomes together.
    def crossover(parent1, parent2):
        child = []

        #Picking where to split
        crossoverpoint = random.randint(1, len(parent1.chromosome)-1)
        side = random.randint(0, 1)

        if side == 0:
            child.extend(parent1.chromosome[:crossoverpoint])
            child.extend(parent2.chromosome[crossoverpoint:])
        else:
            child.extend(parent2.chromosome[:crossoverpoint])
            child.extend(parent1.chromosome[crossoverpoint:])

        child_object = MyCreature(numCreaturePercepts, numCreatureActions)
        child_object.chromosome = child

        return child_object

    # Will change the childs chromosome with a 10% chance
    def mutation(child):
        mutate_child = MyCreature(numCreaturePercepts, numCreatureActions)
        if random.randint(0, 100) > 90:
            randomindex = random.randint(0, len(child.chromosome)-1)
            randomnumber = round(random.uniform(0.0, 0.9), 2)
            child.chromosome[randomindex] = randomnumber
            mutate_child.chromosome = child.chromosome
        else:
            mutate_child.chromosome = child.chromosome

        print("**************")
        print("MMA", mutate_child.chromosome[0])
        print("MMC", mutate_child.chromosome[1])
        print("CMA", mutate_child.chromosome[2])
        print("CMC", mutate_child.chromosome[3])
        print("FMA", mutate_child.chromosome[4])
        print("FMC", mutate_child.chromosome[5])
        print("EAT", mutate_child.chromosome[6])
        print("RND", mutate_child.chromosome[7])
        print("***************")

        return mutate_child

    # Picks two fit parents to make a child.
    def elitismSelect(subset):

        hold = []

        for critter in subset:
            hold.append(critter)

        hold = sorted(hold, key=lambda Creature: Creature.fitness, reverse=True)

        best = hold[:int(len(subset)*0.10)]
        best_creatures = []

        for creature in best:
            best_object = MyCreature(numCreaturePercepts, numCreatureActions)
            best_object.chromosome = creature.chromosome
            best_creatures.append(best_object)

        return best_creatures

    # Elitism
    best = elitismSelect(old_population)
    new_population.extend(best)

    while len(new_population) < len(old_population):
        # Child
        random_sample = random.sample(old_population, int(len(old_population)/4))
        parent1, parent2 = tournamentSelect(random_sample)
        new_population.append(mutation(crossover(parent1, parent2)))

    return new_population

# Create the world.  Representaiton type choses the type of percept representation (there are three
#  types to chose from);
# gridSize specifies the size of the world, repeatable parameter allows you to run the simulation
# in exactly same way.
w = World(representationType=perceptFormat, gridSize=gridSize, repeatable=repeatableMode)

#Get the number of creatures in the world
numCreatures = w.maxNumCreatures()

#Get the number of creature percepts
numCreaturePercepts = w.numCreaturePercepts()

#Get the number of creature actions
numCreatureActions = w.numCreatureActions()

# Create a list of initial creatures - instantiations of the MyCreature class that you implemented
population = list()
for i in range(numCreatures):
   c = MyCreature(numCreaturePercepts, numCreatureActions)
   population.append(c)

# Pass the first population to the world simulator
w.setNextGeneration(population)

# Runs the simulation to evalute the first population
w.evaluate(numTurns)

# Show visualisation of initial creature behaviour
w.show_simulation(titleStr='Initial population', speed='normal')

for i in range(numGenerations):
    print("\nGeneration %d:" % (i+1))

    # Create a new population from the old one
    population = newPopulation(population)

    # Pass the new population to the world simulator
    w.setNextGeneration(population)

    # Run the simulation again to evalute the next population
    w.evaluate(numTurns)

    # Show visualisation of final generation
    if i==numGenerations-1:
        w.show_simulation(titleStr='Final population', speed='normal')

# Plots the final graph with the number
#plt.plot(avg_life)

plt.plot(avgfitness)
plt.ylabel('Avg Fitness for each evolution')
plt.xlabel('Amount of evolutions')
plt.show()
