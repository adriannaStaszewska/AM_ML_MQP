#get average measurements for batch detectron run
import pandas as pd
import json

models_folder = "/home/azstaszewska/Batch models"
metrics_1, metrics_2 = [], []

for i in range(100, 156):
    metrics_file = models_folder+"/model_"+str(i)+"/metrics.json"
    paths_file = open(metrics_file, 'r')
    data = paths_file.readlines()
    metrics = [line.rstrip('\n') for line in data]
    new_data_1 = json.loads(metrics[-1])
    new_data_1["id"] = i
    metrics_1.append(new_data_1)

    new_data_2 = json.loads(metrics[-2])
    new_data_2["id"] = i
    metrics_2.append(new_data_2)



'''s3a4km
IN_FILE = "/home/azstaszewska/batch_run_results_new.txt"
paths_file = open(IN_FILE, 'r')
data = paths_file.readlines()
data = [line.rstrip('\n') for line in data]

metrics_1 = data[1::2]
print(metrics_1)
df_1 = pd.json_normalize(json.loads(str(metrics_1)), sep="_")
print(df_1)

metrics_2 = data[::2]
df_2 = pd.json_normalize(json.loads("{"+str(metrics_2)+"}"), sep="_")
'''
df_1 = pd.DataFrame(metrics_1)

df_2 = pd.DataFrame(metrics_2)

data_df = pd.merge(df_1, df_2, how='inner', on="id")

print(data_df.describe(include='all'))

data_df.describe(include='all').to_csv("/home/azstaszewska/batch_run_summary.csv")
