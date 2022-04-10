import pandas as pd
import numpy as np
import pickle
import os

class Logger:
    def __init__(self, filename):
        self.filename = filename
        f = open(f"{self.filename}.csv", "w")
        f.close()
        
    def log(self, msg):
        f = open(f"{self.filename}.csv", "a+")
        f.write(f"{msg}\n")
        f.close()