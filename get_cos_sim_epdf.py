import sys
import math

weights = []
with open(sys.argv[1]) as f:
    for line in f:
        weight,*junk = line.split()
        weights += [float(weight)]

weights = sorted(weights)
l = len(weights)
print(l)
print(weights[math.floor(.5*l)])
print(weights[math.floor(.75*l)])
print(weights[math.floor(.9*l)])
print(weights[math.floor(.95*l)])
