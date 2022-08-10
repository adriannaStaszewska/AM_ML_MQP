#merge polygons

from shapely.geometry.polygon import Polygon
import json

s = "G8"
OUT = "/home/azstaszewska/Data/MS Data/Sets/Labelme sets/"

detailed = "/home/azstaszewska/Data/MS Data/Sets/Labelme sets/"+s+"_50.json"
to_fix = "/home/azstaszewska/Data/MS Data/Sets/"+s+"_fix.json"

fix_f = open(to_fix, )
fix_ann = json.load(fix_f)

detailed_ann_f = open(detailed, )
master = json.load(detailed_ann_f)
master_polys = [Polygon(a['points']).simplify(1) for a in master["shapes"]]
new_polys = []

for i in fix_ann["shapes"]:
    if i["shape_type"] == "rectangle":
        i["points"] = [[i["points"][0][0],i["points"][0][1]], [i["points"][0][0],i["points"][1][1]], [i["points"][1][0],i["points"][1][1]], [i["points"][1][0],i["points"][1][1]]]
    p = Polygon(i["points"])
    for m in master_polys:
        if p.intersects(m):
            x, y = m.exterior.coords.xy
            polygon_new = list(zip(x, y))
            i["points"] = [list(p) for p in polygon_new]
            i["shape_type"] = "polygon"
            new_polys.append(i)

fix_ann["shapes"] = new_polys
with open(OUT+s+"_fixed.json", 'w') as f_ann:  # write back to the JSON
    json.dump(fix_ann, f_ann, indent=2)
