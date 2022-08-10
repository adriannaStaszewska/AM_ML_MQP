#Generate a confusion matrix
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
import json
import ast
from matplotlib.patches import Ellipse, Polygon
import math
from scipy.spatial import ConvexHull

RANDOM_STATE = 42
CLASSES = ['lack of fusion', 'keyhole']
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


headers = ["sample", "polygon", "class", "area", "perimeter", "major_axis", "minor_axis", "convex_hull","ch_area", "centroid_x", "centroid_y", "max_feret_diameter", "min_feret_diameter", "bbox_x", "bbox_y", "bbox_h", "bbox_w", "bbox_ratio", "circularity", "aspect_ratio", "roundness", "solidity", "roughness", "shape_factor", "convexity", "pta_ratio", "ellipse_major_axis", "ellipse_minor_axis", "ellipse_angle", "ellipse_x", "ellipse_y", "ellipse_ratio", "laser_power", "scan_speed", "layer_thickness", "hatch_spacing", "sim_melt_pool_width", "linear_energy_density", "surface_energy_density", "volumetric_energy_density"]

features = ["area", "perimeter", "major_axis", "minor_axis", "ch_area", "centroid_x", "centroid_y",  "bbox_x", "bbox_y", "bbox_h", "bbox_w", "bbox_ratio", "aspect_ratio", "roundness", "solidity", "shape_factor", "convexity", "pta_ratio", "ellipse_major_axis", "ellipse_minor_axis", "ellipse_angle", "ellipse_x", "ellipse_y", "ellipse_ratio","laser_power", "scan_speed", "hatch_spacing", "sim_melt_pool_width", "linear_energy_density", "surface_energy_density", "volumetric_energy_density"]

geometric_features = ["area", "perimeter", "major_axis", "minor_axis", "ch_area", "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_h", "bbox_w", "bbox_ratio",  "aspect_ratio", "roundness", "solidity",  "shape_factor", "convexity", "pta_ratio", "ellipse_major_axis", "ellipse_minor_axis", "ellipse_angle", "ellipse_x", "ellipse_y", "ellipse_ratio"]
processing_parmas = ["laser_power", "scan_speed", "hatch_spacing", "sim_melt_pool_width", "linear_energy_density", "surface_energy_density", "volumetric_energy_density"]
geo_non_linear = ["class", "perimeter", "area", "bbox_ratio", "aspect_ratio", "solidity", "shape_factor", "pta_ratio",  "ellipse_major_axis", "ellipse_minor_axis", "ellipse_angle", "ellipse_x", "ellipse_y", "ellipse_ratio"]
geo_selected =["perimeter", "area", "bbox_ratio", "aspect_ratio", "solidity",  "shape_factor", "pta_ratio",  "ellipse_major_axis", "ellipse_minor_axis", "ellipse_angle", "ellipse_x", "ellipse_y", "ellipse_ratio"]
process_non_linear = ["class", "laser_power", "scan_speed", "hatch_spacing", "sim_melt_pool_width"]

df = pd.read_csv('/home/azstaszewska/Data/MS Data/extracted_features_clean_v6.csv', )
df = shuffle(df, random_state=RANDOM_STATE)
#train, test = train_test_split(df, test_size=0.1, random_state=RANDOM_STATE)
#print(df)
#df = train
df = df.dropna()
df = df.loc[:, (df.columns != 'layer_thickness') & (df.columns != 'max_feret_diameter') & (df.columns != 'min_feret_diameter')& (df.columns != 'include')& (df.columns != 'boundry')& (df.columns != 'area_in')]
#print(len(df.index))
#print(df[['area', 'ch_area']])

subset = df.sample(n=20).to_dict("records")
i = 0
for r in subset:
    polygon = ast.literal_eval(r["polygon"])
    x, y = zip(*polygon)
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    poly_patch1 = Polygon(xy=polygon, color="darkgray")
    poly_patch2 = Polygon(xy=polygon, color="darkgray")
    #plot pore geometry
    ax.add_patch(poly_patch1)
    #ax2.add_patch(poly_patch2)
    ax.plot(x, y, color="blue", label="perimeter")

    #plot ellipse
    ellipse = Ellipse(xy=(float(r["ellipse_x"]), float(r["ellipse_y"])), width=2*float(r["ellipse_major_axis"]), height=2*float(r["ellipse_minor_axis"]), angle=math.degrees(float(r["ellipse_angle"])), edgecolor='r', fc='None', lw=2, label="best-fitting ellipse")
    ax.add_patch(ellipse)
    plt.plot(float(r["ellipse_x"]), float(r["ellipse_y"]), marker="o", markersize=5, markerfacecolor="red", markeredgecolor="red")

    #plot convex hull
    hull = ConvexHull(polygon)
    ch_polygon = ast.literal_eval(r["convex_hull"])
    ch_order = [True if p in ch_polygon else False for p in polygon]
    ch_ordered = np.array(polygon)[ch_order]
    x_c, y_c = zip(*ch_ordered)
    ax.plot(x_c, y_c, color="green", linestyle="--", label="convex hull")
    x_bb = [r["bbox_x"], r["bbox_x"], r["bbox_x"]+r["bbox_h"], r["bbox_x"]+r["bbox_h"], r["bbox_x"]]
    y_bb = [r["bbox_y"], r["bbox_y"]+r["bbox_w"], r["bbox_y"]+r["bbox_w"], r["bbox_y"], r["bbox_y"]]
    ax.plot(x_bb, y_bb, color="cyan", linestyle="--", label="minimum bounding-box")
    #plot feret diameters
    min_feret_x, min_feret_y = zip(*ast.literal_eval(r["min_feret"]))
    max_feret_x, max_feret_y = zip(*ast.literal_eval(r["max_feret"]))
    #ax.plot(min_feret_x, min_feret_y, label="min Feret diameter", color = "purple", linestyle='dotted')
    #ax.plot(max_feret_x, max_feret_y, label="max Feret diameter", color = "violet", linestyle='dotted')

    #plot major/minor axes

    rotated_rect = ast.literal_eval(r["min_rect"])
    x, y = zip(*rotated_rect)
    ax.plot(x, y, color="darkorange", linestyle="-.", label="minimum rotated rectangle")
    #l1_x, l1_y =  zip(*[rotated_rect[0], rotated_rect[1]])
    #l2_x, l2_y = zip(*[rotated_rect[1], rotated_rect[2]])

    #plt.plot(l1_x, l1_y , label="major axis", color = "orange", linestyle='-.')
    #plt.plot(l2_x, l2_y , label="minor axis", color = "orange", linestyle='-.')


    #plot centroid
    ax.plot(float(r["centroid_x"]), float(r["centroid_y"]), marker="o", markersize=5, markerfacecolor="blue", markeredgecolor="blue", label="centroid")

    #ax.legend()
    ax.legend(loc='right', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True, fontsize=15)
    plt.axis('off')
    plt.savefig("/home/azstaszewska/Data/MS Data/Viz/polygon_"+str(i)+".png")

    plt.clf()
    i+=1

plt.close('all')
sns.set(font_scale=2)
sns.set_style(style='white')
corr = df.corr()
cmap = sns.diverging_palette(10, 220, as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(36,36))
sns.heatmap(corr, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f', annot_kws={"size":16})
#plt.title('Correlation matrix with Pearson correlation coefficient', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_pearson_pores.png")
plt.clf()


corr_s = df.corr(method="spearman")
sns.heatmap(corr_s, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
#plt.title('Correlation matrix with Spearman rank correlation', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_spearman_pores.png")
plt.clf()

corr_k = df.corr(method="kendall")
sns.heatmap(corr_k, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f', annot_kws={"size":16})
#plt.title('Correlation matrix with Kendall Tau correlation coefficient', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_kendall_pores.png")
plt.clf()
plt.close('all')

'''
def correlation_table(data,target_column, method):
	data_num = data.select_dtypes(include=['int','float'])
	corr_df = pd.DataFrame(data_num.corrwith(data_num[target_column], method=method),columns=['Correlation']).dropna()
	corr_df['ABS Correlation'] = abs(corr_df['Correlation'])
	corr_df.sort_values(by=['ABS Correlation'], ascending=False, inplace=True)
	print(corr_df)

correlation_table(df, 'class', "pearson")
correlation_table(df, 'class', "kendall")
correlation_table(df, 'class', "spearman")
'''

df['class_name'] = df.apply(lambda row: CLASSES[row["class"]], axis=1)
sns.scatterplot(x="shape_factor", y="aspect_ratio", hue="class_name", data=df)
#plt.title("Hatch Spacing vs Scan Speed with Porosity")
plt.savefig("/home/azstaszewska/sf_vs_ar.png", bbox_inches="tight")
plt.clf()
sns.scatterplot(x="shape_factor", y="pta_ratio", hue="class_name", data=df)
#plt.title("Hatch Spacing vs Scan Speed with Porosity")
plt.savefig("/home/azstaszewska/sf_vs_pta.png", bbox_inches="tight")
plt.clf()
sns.scatterplot(x="shape_factor", y="bbox_ratio", hue="class_name", data=df)
#plt.title("Hatch Spacing vs Scan Speed with Porosity")
plt.savefig("/home/azstaszewska/sf_vs_bb.png", bbox_inches="tight")
plt.clf()
sns.scatterplot(x="bbox_ratio", y="aspect_ratio", hue="class_name", data=df)
#plt.title("Hatch Spacing vs Scan Speed with Porosity")
plt.savefig("/home/azstaszewska/bb_vs_ar.png", bbox_inches="tight")
plt.clf()
sns.scatterplot(x="pta_ratio", y="aspect_ratio", hue="class_name", data=df)
#plt.title("Hatch Spacing vs Scan Speed with Porosity")
plt.savefig("/home/azstaszewska/pta_vs_ar.png", bbox_inches="tight")
plt.clf()
sns.scatterplot(x="pta_ratio", y="bbox_ratio", hue="class_name", data=df)
#plt.title("Hatch Spacing vs Scan Speed with Porosity")
plt.savefig("/home/azstaszewska/pta_vs_bb.png", bbox_inches="tight")
plt.clf()



sns.set(font_scale=1.6)
sns.set_style(style='white')
df_geo = df[geo_non_linear]
corr = df_geo.corr()
cmap = sns.diverging_palette(10, 220, as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(30,20))
sns.heatmap(corr, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f', annot_kws={"size":15})
plt.title('Correlation matrix for selected geometric features with Pearson correlation coefficient', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_pearson_pores_selected_geo.png")
plt.clf()


corr_s = df_geo.corr(method="spearman")
sns.heatmap(corr_s, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f', annot_kws={"size":15})
plt.title('Correlation matrix for selected geometric features with Spearman rank correlation', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_spearman_pores_selected_geo.png")
plt.clf()

corr_k = df_geo.corr(method="kendall")
sns.heatmap(corr_k, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f', annot_kws={"size":15})
plt.title('Correlation matrix for selected geometric features with Kendall Tau correlation coefficient', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_kendall_pores_selected_geo.png")
plt.clf()

corr_k = df_geo[["shape_factor", "solidity", "aspect_ratio", "bbox_ratio", "pta_ratio", "ellipse_ratio"]].corr(method="kendall")
mask = np.zeros_like(corr_k, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.set(font_scale=3)
sns.set_style(style='white')
plt.figure(figsize=(30,20))

sns.heatmap(corr_k, cmap = cmap,annot=True, vmax=1, vmin=-1, center = 0, mask = mask, fmt= '.2f')
#plt.title('Correlation matrix for selected geometric features with Kendall Tau correlation coefficient', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_kendall_pores_final.png")
plt.clf()

corr= df_geo[["shape_factor", "solidity", "aspect_ratio", "bbox_ratio", "pta_ratio", "ellipse_ratio"]].corr()
sns.heatmap(corr, cmap = cmap,annot=True, vmax=1, vmin=-1, center = 0, mask = mask, fmt= '.2f')
#plt.title('Correlation matrix for selected geometric features with Pearson correlation coefficient', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_pearson_pores_final.png")
plt.clf()
plt.close('all')

sns.set(font_scale=2)
sns.set_style(style='white')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40,15))
fig.suptitle('Correlation matrices for selected features', fontsize=35)
sns.heatmap(corr, ax = ax1, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
ax1.set_title('Linear correlation measured with Pearson correlation coefficient', fontsize=30)
sns.heatmap(corr_k, ax=ax2, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
ax2.set_title('Non-linear correlation measured with Kendall Tau correlation coefficient', fontsize=30)
plt.savefig("/home/azstaszewska/Data/MS Data/Plots/correlations_final.png", bbox_inches="tight")
plt.clf()

corr= df[["shape_factor", "solidity", "aspect_ratio", "bbox_ratio", "pta_ratio", "ellipse_ratio", "hatch_spacing", "laser_power", "scan_speed"]].corr()
corr_k = df[["shape_factor", "solidity", "aspect_ratio", "bbox_ratio", "pta_ratio", "ellipse_ratio","hatch_spacing", "laser_power", "scan_speed"]].corr(method="kendall")
mask = np.zeros_like(corr_k, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.set(font_scale=2)
sns.set_style(style='white')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(40,15))
fig.suptitle('Correlation matrices for selected geometric features and processing parameters', fontsize=35)
sns.heatmap(corr, ax = ax1, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
ax1.set_title('Linear correlation measured with Pearson correlation coefficient', fontsize=30)
sns.heatmap(corr_k, ax=ax2, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
ax2.set_title('Non-linear correlation measured with Kendall Tau correlation coefficient', fontsize=30)
plt.savefig("/home/azstaszewska/Data/MS Data/Plots/correlations_final_2.png", bbox_inches="tight")
plt.clf()

df_process = df[process_non_linear]
corr = df_process.corr()
cmap = sns.diverging_palette(10, 220, as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(30,20))
sns.heatmap(corr, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
plt.title('Correlation matrix for processing parameters with Pearson correlation coefficient', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_pearson_pores_selected_process.png")
plt.clf()


corr_s = df_process.corr(method="spearman")
sns.heatmap(corr_s, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
plt.title('Correlation matrix for processing parameters with Spearman rank correlation', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_spearman_pores_selected_process.png")
plt.clf()

corr_k = df_process.corr(method="kendall")
sns.heatmap(corr_k, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
plt.title('Correlation matrix for sprocessing parameters with Kendall Tau correlation coefficient', fontsize=30)
plt.savefig("/home/azstaszewska/correlations_kendall_pores_selected_process.png")
plt.clf()
plt.close('all')

plt.figure(figsize=(60,60))
sns.set(font_scale=1.6)
sns.set_style(style='white')
df_sample = pd.read_csv('/home/azstaszewska/Data/MS Data/sample_summary_v6.csv', )

sns.scatterplot(data=df_sample, x="laser_power", y="porosity")
#plt.title("Laser Power vs Porosity")
plt.savefig("/home/azstaszewska/porosity_vs_power.png")

plt.clf()
sns.scatterplot(data=df_sample, x="scan_speed", y="porosity")
#plt.title("Scan Speed vs Porosity")
plt.savefig("/home/azstaszewska/porosity_vs_speed.png")
plt.clf()
sns.scatterplot(data=df_sample, x="hatch_spacing", y="porosity")
#plt.title("Hatch Spacing vs Porosity")
plt.savefig("/home/azstaszewska/porosity_vs_hatch.png")

plt.clf()

sns.relplot(x="scan_speed", y="laser_power", size="porosity", sizes=(20, 300), data=df_sample)
#plt.title("Laser Power vs Scan Speed with Porosity")
plt.savefig("/home/azstaszewska/rel_plot_power_speed.png", bbox_inches="tight")
plt.clf()

sns.relplot(x="laser_power", y="hatch_spacing", size="porosity", sizes=(20, 300), data=df_sample)
#plt.title("Laser Power vs Hatch Spacing with Porosity")
plt.savefig("/home/azstaszewska/rel_plot_power_spacing.png", bbox_inches="tight")
plt.clf()

sns.relplot(x="hatch_spacing", y="scan_speed", size="porosity", sizes=(20, 300), data=df_sample)
#plt.title("Hatch Spacing vs Scan Speed with Porosity")
plt.savefig("/home/azstaszewska/rel_plot_spacing_speed.png", bbox_inches="tight")
plt.clf()

TEST = ['Q6', 'R6', 'G8', 'H6R', 'J3R', 'Q3']
df_sample["test"] = np.where(df_sample['set'].isin(TEST), True, False)
plt.clf()

p1 = sns.scatterplot(data=df_sample, x="laser_power", y="hatch_spacing", hue="test")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
for line in range(0,df_sample.shape[0]):
    if df_sample["test"][line]:
         p1.text(df_sample["laser_power"][line]+0.01, df_sample["hatch_spacing"][line],
                 df_sample["set"][line], horizontalalignment='left',
                 size='small', color='black')
plt.savefig("/home/azstaszewska/Data/MS Data/Plots/test_set_1.png",  bbox_inches="tight")
plt.clf()

p2 = sns.scatterplot(data=df_sample, x="scan_speed", y="laser_power", hue="test")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
for line in range(0,df_sample.shape[0]):
    if df_sample["test"][line]:
         p2.text(df_sample["scan_speed"][line]+0.01, df_sample["laser_power"][line]+0.01,
                 df_sample["set"][line], horizontalalignment='left',
                 size='small', color='black')

plt.savefig("/home/azstaszewska/Data/MS Data/Plots/test_set_2.png",  bbox_inches="tight")
plt.clf()

p3 = sns.scatterplot(data=df_sample, x="hatch_spacing", y="scan_speed", hue="test")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
for line in range(0,df_sample.shape[0]):
    if df_sample["test"][line]:
         p3.text(df_sample["hatch_spacing"][line]+0.01, df_sample["scan_speed"][line]+0.01,
                 df_sample["set"][line], horizontalalignment='left',
                 size='small', color='black')
plt.savefig("/home/azstaszewska/Data/MS Data/Plots/test_set_3.png",  bbox_inches="tight")
plt.clf()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(df_sample['laser_power'], df_sample['scan_speed'], df_sample["hatch_spacing"])
plt.savefig("/home/azstaszewska/Data/MS Data/Plots/3d_1.png")



'''
corr = df.corr()
cmap = sns.diverging_palette(10, 220, as_cmap=True)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(30,20))
sns.heatmap(corr, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
plt.title('Correlation matrix with Pearson correlation coefficient', fontsize=20)
plt.savefig("/home/azstaszewska/correlations_pearson_1.png")
plt.clf()


corr_s = df.corr(method="spearman")
sns.heatmap(corr_s, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
plt.title('Correlation matrix with Spearman rank correlation', fontsize=20)
plt.savefig("/home/azstaszewska/correlations_spearman_1.png")
plt.clf()

corr_k = df.corr(method="kendall")
sns.heatmap(corr_k, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
plt.title('Correlation matrix with Kendall Tau correlation coefficient', fontsize=20)
plt.savefig("/home/azstaszewska/correlations_kendall_1.png")
plt.clf()
plt.close('all')
'''
'''

df_trim = df[geo_selected]
abs_corr = df_trim.corr().abs()
upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
# drop the columns
df_trim.drop(to_drop,axis=1,inplace=True)
corr_trim = df_trim.corr()
mask = np.zeros_like(corr_trim, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(corr_trim, cmap = cmap,annot=True, center = 0, mask = mask, fmt= '.2f')
plt.savefig("/home/azstaszewska/correlations_dropped.png")

corr[corr < 1].unstack().transpose()\
    .sort_values( ascending=False)\
    .drop_duplicates().to_csv("/home/azstaszewska/correlations.csv")


df['class_name'] = df.apply(lambda row: CLASSES[row["class"]], axis=1)
y_train = df.loc[:,"class"]
X_train = df.loc[:, features]

y = df.loc[:,"class_name"]
#print("Test: " + str(len(test)))
#print("Train: " + str(len(train)))
plt.clf()
sns.set(font_scale=1.6)
sns.set_style(style='white')
plt.figure(figsize=(9,6))
ax = sns.countplot(y)
ax.set(xlabel='Class', ylabel="Count")
#ax.set_title("Instance counts for each class")
B, M = y.value_counts()
plt.savefig("/home/azstaszewska/counts.png")
print(B, M)
normed_data = (X_train-X_train.min())/(X_train.max() - X_train.min())



test = SelectKBest(score_func=eval("mutual_info_classif"), k='all')
fit = test.fit(normed_data.loc[:,["shape_factor", "solidity", "aspect_ratio", "bbox_ratio", "pta_ratio", "ellipse_ratio","hatch_spacing", "laser_power", "scan_speed"]], y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(["shape_factor", "solidity", "aspect_ratio", "bbox_ratio", "pta_ratio", "ellipse_ratio","hatch_spacing", "laser_power", "scan_speed"])

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']
sorted_features = featureScores.sort_values("Score")
# Now plot
plt.figure()
plt.xticks(rotation='vertical')
plt.gcf().subplots_adjust(bottom=0.5)
plt.xlabel("feature")
plt.ylabel("score")
# blue #417CA7, red "#D93A46"
colors = ['blue' if s in processing_parmas else "red" for s in sorted_features['Feature']]
plt.bar(sorted_features['Feature'],sorted_features["Score"], align='center', width=0.4, color = colors)
#plt.title('Feature importance measured with mutual information (mutual_info_classif)')
plt.savefig("/home/azstaszewska/geo_scores_mutual_info_classif.png", bbox_inches='tight')
plt.clf()


test = SelectKBest(score_func=eval("f_classif"), k='all')
fit = test.fit(normed_data.loc[:,["shape_factor", "solidity", "aspect_ratio", "bbox_ratio", "pta_ratio", "ellipse_ratio","hatch_spacing", "laser_power", "scan_speed"]], y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(["shape_factor", "solidity", "aspect_ratio", "bbox_ratio", "pta_ratio", "ellipse_ratio","hatch_spacing", "laser_power", "scan_speed"])

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']
sorted_features = featureScores.sort_values("Score")
plt.figure()
plt.xticks(rotation='vertical')
plt.gcf().subplots_adjust(bottom=0.5)
plt.xlabel("Feature")
plt.ylabel("Score")
plt.yticks(np.arange(0, 5000, step=500))
plt.grid(axis='y', which="both")
# blue #417CA7, red "#D93A46"
colors = ['blue' if s in processing_parmas else "red" for s in sorted_features['Feature']]
plt.bar(sorted_features['Feature'],sorted_features["Score"], align='center', width=0.4, color = colors)
#plt.title('Feature importance measured with ANOVA F-value (f_classif)')
plt.savefig("/home/azstaszewska/geo_process_scores_f_classif.png", bbox_inches='tight')
plt.clf()


test = SelectKBest(score_func=eval("chi2"), k='all')
fit = test.fit(normed_data.loc[:,["shape_factor", "solidity", "aspect_ratio", "bbox_ratio", "pta_ratio", "ellipse_ratio","hatch_spacing", "laser_power", "scan_speed"]], y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(["shape_factor", "solidity", "aspect_ratio", "bbox_ratio", "pta_ratio", "ellipse_ratio","hatch_spacing", "laser_power", "scan_speed"])

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']
sorted_features = featureScores.sort_values("Score")
plt.figure()
plt.xticks(rotation='vertical')
plt.gcf().subplots_adjust(bottom=0.5)
plt.xlabel("Feature")
plt.ylabel("Score")
plt.yticks(np.arange(0, 500, step=50))
plt.grid(axis='y', which="both")
colors = ['blue' if s in processing_parmas else "red" for s in sorted_features['Feature']]
plt.bar(sorted_features['Feature'],sorted_features["Score"], align='center', width=0.4, color = colors)
#plt.title('Feature importance measured with chi-squared statistical test (chi2)')
plt.savefig("/home/azstaszewska/geo_process_scores_chi2.png", bbox_inches='tight')
plt.clf()


test = SelectKBest(score_func=eval("mutual_info_classif"), k='all')
fit = test.fit(normed_data.loc[:,geo_selected], y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(geo_selected)

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']
sorted_features = featureScores.sort_values("Score")
# Now plot
plt.figure()
plt.xticks(rotation='vertical')
plt.gcf().subplots_adjust(bottom=0.5)
plt.xlabel("Feature")
plt.ylabel("Score")
# blue #417CA7, red "#D93A46"
colors = ['blue' if s in processing_parmas else "red" for s in sorted_features['Feature']]
plt.bar(sorted_features['Feature'],sorted_features["Score"], align='center', width=0.4, color = colors)
#plt.title('Feature importance measured with mutual information (mutual_info_classif)')
plt.savefig("/home/azstaszewska/geo_scores_mutual_info_classif.png", bbox_inches='tight')
plt.clf()


test = SelectKBest(score_func=eval("f_classif"), k='all')
fit = test.fit(normed_data.loc[:,geo_selected], y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(geo_selected)

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']
sorted_features = featureScores.sort_values("Score")
plt.figure()
plt.xticks(rotation='vertical')
#plt.gcf().subplots_adjust(bottom=0.5)
plt.yticks(np.arange(0, 2500, step=500))
plt.grid(axis='y', which="both")
plt.xlabel("Feature")
plt.ylabel("Score")
# blue #417CA7, red "#D93A46"
colors = ['blue' if s in processing_parmas else "red" for s in sorted_features['Feature']]
plt.bar(sorted_features['Feature'],sorted_features["Score"], align='center', width=0.4, color = colors)
#plt.title('Feature importance measured with ANOVA F-value (f_classif)')
plt.savefig("/home/azstaszewska/geo_scores_f_classif.png", bbox_inches='tight')
plt.clf()


test = SelectKBest(score_func=eval("chi2"), k='all')
fit = test.fit(normed_data.loc[:,geo_selected], y_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(geo_selected)

#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Feature','Score']
sorted_features = featureScores.sort_values("Score")
plt.figure()
plt.xticks(rotation='vertical')

#plt.gcf().subplots_adjust(bottom=0.5)
plt.xlabel("Feature")
plt.ylabel("Score")
plt.yticks(np.arange(0, 200, step=25))
plt.grid(axis='y', which="both")
colors = ['blue' if s in processing_parmas else "red" for s in sorted_features['Feature']]
plt.bar(sorted_features['Feature'],sorted_features["Score"], align='center', width=0.4, color = colors)
#plt.title('Feature importance measured with chi-squared statistical test (chi2)')
plt.savefig("/home/azstaszewska/geo_scores_chi2.png", bbox_inches='tight')
plt.clf()



'''
'''
for i in ["mutual_info_classif", "f_classif", "chi2"]:
    test = SelectKBest(score_func=eval(i), k='all')
    fit = test.fit(normed_data.loc[:,geo_selected], y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(geo_selected)

    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']
    sorted_features = featureScores.sort_values("Score")
    # Now plot
    plt.figure()
    plt.xticks(rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.5)
    # blue #417CA7, red "#D93A46"
    colors = ['blue' if s in processing_parmas else "red" for s in sorted_features['Feature']]
    plt.bar(sorted_features['Feature'],sorted_features["Score"], align='center', width=0.4, color = colors)
    plt.savefig("/home/azstaszewska/geo_scores_"+i+".png")
    plt.clf()
'''
'''

for i in ["mutual_info_classif", "f_classif", "chi2"]:
    test = SelectKBest(score_func=eval(i), k='all')
    fit = test.fit(normed_data, y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)

    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Feature','Score']
    sorted_features = featureScores.sort_values("Score")
    # Now plot
    plt.figure()
    plt.xticks(rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.5)
    colors = ["blue" if s in processing_parmas else "red" for s in sorted_features['Feature']]
    plt.bar(sorted_features['Feature'],sorted_features["Score"], align='center', width=0.4, color = colors)
    plt.savefig("/home/azstaszewska/all_scores_"+i+".png")
    plt.clf()
'''
'''
plt.figure(figsize=(30,20))
plt.clf()
boxplot = normed_data.loc[:,geometric_features].boxplot()
plt.xticks(rotation=90)
plt.savefig("/home/azstaszewska/boxes_normed.png")

plt.clf()

boxplot = X_train.loc[:,geometric_features].boxplot()
plt.xticks(rotation=90)
plt.savefig("/home/azstaszewska/boxes.png")

plt.figure(figsize=(8,8))
plt.clf()

sns.set(font_scale=1.3)
sns.set_style(style='white')
X_train["class_name"] = y
X_train["all"] = 1

for f in features:
    ax = sns.violinplot(x='all', y=f, hue="class_name", data=X_train, split=True)
    ax.set(xlabel = None)
    plt.savefig("/home/azstaszewska/Data/MS Data/Plots/violin_"+f+".png")
    plt.clf()
plt.close('all')

plt.figure(figsize=(10,10))
plt.clf()
sns.set(font_scale=2)
sns.set_style(style='white')
ax = sns.violinplot(x='all', y='hatch_spacing', hue="class_name", data=X_train, split=True)
ax.set(xlabel = None)
ax.get_legend().remove()
plt.savefig("/home/azstaszewska/Data/MS Data/Plots/violin_"+'hatch_spacing'+".png")
plt.clf()

ax = sns.violinplot(x='all', y="laser_power", hue="class_name", data=X_train, split=True)
ax.set(xlabel = None)
ax.get_legend().remove()
plt.savefig("/home/azstaszewska/Data/MS Data/Plots/violin_"+"laser_power"+".png")
plt.clf()
ax = sns.violinplot(x='all', y="scan_speed", hue="class_name", data=X_train, split=True)
ax.set(xlabel = None)
#plt.legend(bbox_to_anchor =(1, 0.5))
plt.savefig("/home/azstaszewska/Data/MS Data/Plots/violin_"+"scan_speed"+".png")
plt.clf()
'''


plt.clf()
sns.histplot(df['area'])
plt.savefig("/home/azstaszewska/Data/MS Data/Plots/area_hist.png", bbox_inches='tight')
plt.clf()

sns.histplot(df['major_axis'])
plt.savefig("/home/azstaszewska/Data/MS Data/Plots/axis_hist.png", x="Major axis length", bbox_inches='tight')
plt.clf()

'''
pp = sns.pairplot(data=X_train.loc[:, non_linear[0:5]])
plt.savefig("/home/azstaszewska/pair_plot_non_linear_1.png")
plt.clf()

pp = sns.pairplot(data=X_train.loc[:, non_linear[5:10]])
plt.savefig("/home/azstaszewska/pair_plot_non_linear_2.png")
plt.clf()

pp = sns.pairplot(data=X_train.loc[:, non_linear[10:]])
plt.savefig("/home/azstaszewska/pair_plot_non_linear_3.png")
plt.clf()

pp = sns.pairplot(data=X_train, x_vars = non_linear[0:5], y_vars = non_linear[5:10])
plt.savefig("/home/azstaszewska/pair_plot_non_linear_4.png")
plt.clf()

pp = sns.pairplot(data=X_train, x_vars = non_linear[0:5], y_vars = non_linear[10:])
plt.savefig("/home/azstaszewska/pair_plot_non_linear_5.png")
plt.clf()

pp = sns.pairplot(data=X_train, x_vars = non_linear[5:10], y_vars = non_linear[10:])
plt.savefig("/home/azstaszewska/pair_plot_non_linear_6.png")
plt.clf()
'''





#sns.violinplot(data=normed_data)
#plt.savefig("/home/azstaszewska/violin_sns.png")
#plt.clf()
#sns.boxplot(data=normed_data)
#plt.xticks(rotation=90)
#plt.savefig("/home/azstaszewska/boxes_sns2.png")
