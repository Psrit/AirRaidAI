import matplotlib.pyplot as plt
import numpy as np
import os, sys

record_dir = 'records/'
costs_filename = record_dir + 'costs'
rewards_filename = record_dir + 'test_rewards'

# check record_dir
if not os.path.isdir(record_dir):
    print("No records directory found.")
    sys.exit()

# plot costs curve
if not os.path.isfile(costs_filename):
    print("No cost record found.")
else:
    cost_values = []
    fc = open(costs_filename, 'r')
    try:
        lines = fc.readlines()
        for l in lines:
            l = l.strip()
            try:
                l = float(l)
                cost_values.append(l)
            except:
                # Note here!
                break

        cost_values = np.array(cost_values)

        fig = plt.figure()
        plt.plot(cost_values)
        plt.xlabel("Training round")
        plt.ylabel("Cost")
        fig.savefig(record_dir + "cost_curve.pdf")
    finally:
        fc.close()

# plot rewards curve
if not os.path.isfile(rewards_filename):
    print("No reward record found.")
else:
    reward_values = []
    fr = open(rewards_filename, 'r')
    try:
        lines = fr.readlines()
        for l in lines:
            l = l.strip()
            try:
                l = float(l)
                reward_values.append(l)
            except:
                # Note here!
                continue

        reward_values = np.array(reward_values)

        fig = plt.figure()
        plt.plot(reward_values)
        plt.xlabel("Test round")
        plt.ylabel("Average reward")
        fig.savefig(record_dir + "reward_curve.pdf")
    finally:
        fr.close()
