#extract features from the polygons
import shapely
import json
from shapely.geometry import MultiPoint, Polygon
import csv



headers = ["sample", "file", "img_id", "polygon", "class", "area", "perimeter", "major_axis", "minor_axis", "convex_hull","ch_area"]

#load data set
ann_path = "/home/azstaszewska/Data/Detectron full set/final/test_trim.json"
f_ann = open(ann_path, )
annotation_json = json.load(f_ann)

rows = []

for f in ann_path: #extract info for each file
    file = f["file_name"]
    img_id = f["img_id"]
    sample = ["file_name"].split("/")[-2]

    #extract data por each pore
    for p in f["annotations"]:
        polygon = p["segmentation"]
        class_id = p["class_id"]
        area = p["area"]
        perimeter = Polygon(polygon).length


        new_poly = []

        for poly in polygon:
            sub_poly = list(zip(poly, [[1:])[::2])
            new_poly.append(sub_poly)

        conv_hull = MultiPoint(new_poly).convex_hull
        conv_hull_area = conv_hull.area

        mbr_points = list(zip(*Polygon(polygon).minimum_rotated_rectangle.exterior.coords.xy))

        # calculate the length of each side of the minimum bounding rectangle
        mbr_lengths = [LineString((mbr_points[i], mbr_points[i+1])).length for i in range(len(mbr_points) - 1)]

        # get major/minor axis measurements
        minor_axis = min(mbr_lengths)
        major_axis = max(mbr_lengths)

        rows.append([sample, file_name, img_id, polygon, class_id, area, perimeter, major_axis, minor_axis, convex_hull, conv_hull_area])


#save to csv
with open('/home/azstaszewska/Data/Detectron full set/final/extracted_features.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
