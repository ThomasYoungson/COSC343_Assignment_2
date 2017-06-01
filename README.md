# COSC343_Assignment_2
Evolve a species

Running the code:

First install Anacodna and then create a virtual environment named cosc343.

$ conda create --name cosc343 python=3.6

Then activate the environment and install packages that the engine requires:\
$ conda activate cosc343 \
(cosc343)$ pip install numpy\
(cosc343)$ pip intall pygame

Then you can run the engine:\
(cosc343)$ python world.py


For this assignment, you will implement a genetic algorithm to optimise the fitness of a species of creatures in a simulated two-dimensional world. The world contains edible foods, placed at random, and a population of monsters (your basic zombies). An illus- tration is given below. Your creatures are the smileys; the blue things are the monsters; the strawberries are edible food (red for ripe and green for non-ripe). The algorithm should find behaviours that keep the creatures well fed and not dead.

![alt text](https://raw.githubusercontent.com/ThomasYoungson/COSC343_Assignment_2/master/Screen%20Shot%202017-06-01%20at%207.12.38%20PM.png)
