import pandas as pd 
import GSA as gsa
import benchmarks
import csv
import numpy as np 
import time
import matplotlib.pyplot as plt  

def selector(algo, func_details, popSize, Iter):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]

    if algo == 0:
        x = gsa.GSA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)    
    return x
    
# Select optimizers
GSA = True  # Code by Himanshu Mittal

# Select benchmark function
F1 = True
F2 = True
F3 = True
F4 = True  
F5 = True 
F6 = True  
F7 = True 
F8 = True 
F9 = True  
F10 = True
F11 = True 
F12 = True 
F13 = True 
F14 = True 
F15 = True 
F16 = True 
F17 = True 
F18 = True
F19 = True 
F20 = True 
F21 = True 
F22 = True 
F23 = True

Algorithm = [GSA]
objectivefunc = [F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,
               F11,F12,F13,F14,F15,F16,F17,F18,
               F19,F20,F21,F22,F23] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs 
# are executed for each algorithm.
Runs = 1

# Select general parameters for all optimizers (population size, number of iterations)
PopSize = 50
iterations = 10

# Export results?
Export = True

# Automaticly generated name by date and time
ExportToFile = "experiment" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".csv" 

# Check if it works at least once
atLeastOneIteration = False

# Create an empty DataFrame before the loop
fitness_value = pd.DataFrame(columns=['function', 'value'])

for i in range(0, len(Algorithm)):
    for j in range(0, len(objectivefunc)):
        if (Algorithm[i] and objectivefunc[j]):
            for k in range(0, Runs):
                func_details = benchmarks.getFunctionDetails(j)
                x = selector(i, func_details, PopSize, iterations)

                if Export:
                    with open(ExportToFile, 'a') as out:
                        writer = csv.writer(out, delimiter=',')
                        if not atLeastOneIteration:
                            header = np.concatenate([["Optimizer", "objfname", "startTime", "EndTime", "ExecutionTime"]])
                            writer.writerow(header)
                        a = np.concatenate([[x.Algorithm, x.objectivefunc, x.startTime, x.endTime, x.executionTime]])
                        writer.writerow(a)
                    out.close()
                atLeastOneIteration = True  # at least one experiment
                convergence_value = []

                for value_ in x.convergence:
                    convergence_value.append(value_)

                # Append new data to the DataFrame
                new_data = pd.DataFrame({'function': x.objectivefunc, 'value': convergence_value})
                fitness_value = pd.concat([fitness_value, new_data], ignore_index=True)

                plt.plot([i for i in range(len(x.convergence))], x.convergence)
                plt.title(f'Gravitational Search Algorithm for {x.objectivefunc}  : Convergence Curve')
                plt.xlabel('Iterations')
                plt.ylabel('Fitness value')
                plt.show()

# Move to_csv outside of the innermost loop
fitness_value.to_csv('gsa_experiment.csv', index=False)

if not atLeastOneIteration:
    print("No Optimizer or Cost function is selected. Check lists of available optimizers and cost functions")
