
from __future__ import generators
import shapely
import json
from shapely.geometry import MultiPoint, Polygon, Point
import csv
import math
import matplotlib.pyplot as plt
from shapely.ops import unary_union

from math import sqrt

sets = [
'G7',
'G8',
'G9',
"H3",
'H4',
'H4R',
'H6R',
'H7',
'H9',
'J0',
'J1',
'J3',
"J3R",
'J4',
'J4R',
'J5',
'J8',
'J9',
'K0R',
'K1',
'K4',
'K5',
'Q3',
'Q4',
'Q5',
'Q6',
'Q9',
'R0',
"R6",
'R6R',
'R7',
"G0",
"H0",
'Q6'
]

for s in sets:
    print(s)
    ann_path = "/home/azstaszewska/Data/MS Data/Sets/Labelme sets/v2 fixed/" + s + "_50_v2.json"
    f_ann = open(ann_path, )
    annotation_json = json.load(f_ann)
    shapes = []
    for label in annotation_json["shapes"]:
        label["points"] = [[2*t[0], 2*t[1]] for t in label["points"]]
        shapes.append(label)
    annotation_json["shapes"] = shapes

    new_annotations = []

    for f in annotation_json["shapes"]:
        data = {}
        polygon = f["points"]
        if f["shape_type"] == "circle":
            center = (f["points"][0][0], f["points"][0][1])
            buffer = math.sqrt((center[0]-f["points"][1][0])**2+(center[1]-f["points"][1][1])**2)
            polygon = Point(center).buffer(buffer).exterior.coords
        if f["shape_type"] == "rectangle" or f["label"] == "area":
            continue
        data["class_name"] = f["label"]
        data["polygon"] = [(p[0], p[1]) for p in polygon]
        new_annotations.append(data)


    merged = False

    while not merged:
        merged_annotaions = []
        merged = True
        print("NOT MERGED YET")
        i = 0
        for a1 in new_annotations:
            merged_i = False
            p1 = Polygon(a1["polygon"])
            if not p1.is_valid:
                p1 = p1.buffer(0)
                if p1.geom_type=='MultiPolygon':
                    print("multi")
                    max_a = -1
                    max_poly = None
                    p1 = unary_union(list(p1))
                    for p in list(p1):
                        p = Polygon(p)
                        if p.area > max_a:
                            max_a = p.area
                            max_poly = p
                    p1 = max_poly
                x, y = p1.exterior.coords.xy
                p1_xy = list(zip(x, y))
                a1["polygon"] = p1_xy
            for a2 in new_annotations[i+1:]:
                p2 = Polygon(a2["polygon"])
                if not p2.is_valid:
                    p2 = p2.buffer(0)
                    if p2.geom_type=='MultiPolygon':
                        max_a = -1
                        max_poly = None
                        p2 = unary_union(list(p2))
                        for p in list(p2):
                            p = Polygon(p)
                            if p.area > max_a:
                                max_a = p.area
                                max_poly = p
                        p2 = max_poly
                    x, y = p2.exterior.coords.xy
                    p2_xy = list(zip(x, y))
                    a2["polygon"] = p2_xy
                if p1.intersects(p2) and a1["class_name"] == a2["class_name"]:
                    print("OVERLAP")
                    p3 = p1.union(p2)
                    x, y = p3.exterior.coords.xy
                    p4 = list(zip(x, y))
                    a3 = a1
                    a3["polygon"] = p4
                    merged_annotaions.append(a3)
                    new_annotations.remove(a2)
                    merged = False
                    merged_i = True
                    break
            i+=1
            if merged_i:
                continue
            if a1 not in merged_annotaions:
                merged_annotaions.append(a1)
        new_annotations = merged_annotaions
    print(str(len(new_annotations)))
    with open("/home/azstaszewska/Data/MS Data/Sets/v2 fixed sets/" + s +".json", "w+") as f:
        json.dump(merged_annotaions, f)
