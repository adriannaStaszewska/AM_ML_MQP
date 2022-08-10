#Visualize final sets
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import ast


sets = [
'G7',
'G8',
'G9',
"H3",
'H4',
'H4R',
 'H5',
'H6R',
 'H7',
 'H8',
'H9',
'J0',
'J1',
'J3',
"J3R",
'J4',
'J4R',
'J5',
 'J7',
 'J8',
 'J9',
 'K0',
 'K0R',
 'K1',
'K4',
 'K5',
 'Q3',
 'Q4',
 'Q9',
 'R0',
 "R6",
 'R6R',
 'R7',
 "G0",
 "H0",
 "Q0",
 'Q6'
]



df = pd.read_csv('/home/azstaszewska/Data/MS Data/extracted_features_clean_v2.csv', )
plt.figure()
plt.axis('equal')
for s in sets:
    df_s = df[(df["sample"] == s) & (df["include"] == True)]
    dict_s = df_s.to_dict("recorrds")
    for p in dict_s:
        x, y = zip(*(ast.literal_eval(p["polygon"])))
        plt.plot(x,y)
    plt.savefig("/home/azstaszewska/Data/MS Data/Viz/"+s+"_clean_black.png")
    plt.clf()
