#Visialize and analyze training
import sys
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import dataframe_image as dfi


IN_FILE = sys.argv[1]
MODEL = sys.argv[2]

results = pd.read_csv(IN_FILE)


scoring =["accuracy", "average_precision"]
means_test_acc = results['mean_test_accuracy']
stds_test_acc = results['std_test_accuracy']
means_test_ap = results['mean_test_average_precision']
stds_test_ap = results['std_test_average_precision']

param_cols = [col for col in results if col.startswith('param_')]
print(param_cols)
for p in param_cols:
    sns.pointplot(data=)
