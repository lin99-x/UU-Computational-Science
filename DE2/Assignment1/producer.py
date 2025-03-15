#!/usr/bin/env python3
import time
import pulsar

INPUT_STRING = "I want to be capatilized"

split_message = INPUT_STRING.split(" ")

ITERATION = 5

if ITERATION > len(INPUT_STRING.split()):
    print("Iteration cannot be greater than the number of words in a string")
    print ("Terminating the benchmark")
    exit()

print("Original String: {}".format(INPUT_STRING))

client = pulsar.Client('pulsar://localhost:6650')
producer = client.create_producer('Task4')

for i in range(0, ITERATION):
    word = split_message[i]
    producer.send((word).encode('utf-8'))

#Destroy pulsar client
client.close()