 
import multiprocessing 
# import time 
# import random

run_count = 10
models = []

def build_and_evaluate_model(x):
    for i in range(run_count):
        print(i)

def main():
    pool = multiprocessing.Pool()
    pool = multiprocessing.Pool(processes=4)  
    outputs = pool.map(build_and_evaluate_model, models)

if __name__ == "__main__":
    main()