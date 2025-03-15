#!/usr/bin/env python3
import time
import pulsar

ITERATION = 5

def conversion(substring, operation):
    """A conversion function which takes a string as an input and outputs a converted string

    Args:
        substring (String)
        operation (function): This is an operation on the given input

    Returns:
        [String]: Converted String
    """


    # returns the conversion applied to input
    return function(substring)

def function(string):
    """ A function that performs some operation on a string. You can change the operation accordingly

    Args:
        string (String): input string on which some operation is applied

    Returns:
        [String]: string in upper case
    """
    return string.upper()

resultant_string = ""

client = pulsar.Client('pulsar://localhost:6650')
consumer = client.subscribe('Task4', subscription_name='Task-4')

for i in range(0, ITERATION):
    msg = consumer.receive()
    msg = msg.data().decode('utf-8')
    upper_case_string = conversion(msg, function)
    resultant_string += upper_case_string + ' '

try:
    print("Resultant String: {}".format(resultant_string))
    # consumer.acknowledge(msg)

except:
    consumer.negative_acknowledge(msg)

# Destroy pulsar client
client.close()