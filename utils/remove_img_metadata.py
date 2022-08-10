#remove metadata
import json
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
 'Q5',
 'Q9',
 'R0',
 "R6",
 'R6R',
 'R7',
 "G0",
 "H0",
 "Q0",
 "R2",
 'Q6'
]

SOURCE = "/home/azstaszewska/Data/MS Data/Stitched Final/"
for s in sets:
    ann_path = SOURCE+s+".json"
    f_ann = open(ann_path, )
    annotation_json = json.load(f_ann)

    annotation_json["imageData"] = ""

    with open(ann_path, 'w') as f_ann:  # write back to the JSON
        json.dump(annotation_json, f_ann, indent=2)
