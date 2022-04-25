import cv2
import os

def normalize_classname(class_name):  # normalize the class name to one used by the model
    class_name = class_name.lower()  # remove capitalization
    class_name = class_name.strip()  # remove leading and trailing whitespace
    classes_dict = {  # dictionary containing all class names used in labels and their appropriate model class name
        'gas entrapment porosity': 'gas entrapment porosity',
        'keyhole porosity': 'keyhole porosity',
        'lack of fusion porosity': 'lack of fusion porosity',
        'fusion porosity': 'lack of fusion porosity',
        'gas porosity': 'gas entrapment porosity',
        'lack-of-fusion': 'lack of fusion porosity',
        'keyhole': 'keyhole porosity',
        'other': 'other',
        'lack of fusion': 'lack of fusion porosity',
        'lack-of_fusion': 'lack of fusion porosity',
        'small lack of fusion porosity': 'lack of fusion porosity',
        'medium lack of fusion porosity': 'lack of fusion porosity',
        'large lack of fusion porosity': 'lack of fusion porosity'
    }
    return classes_dict.get(class_name)

def normalize_dimensions(col_min, col_max, row_min, row_max):
    return max(col_min, 0), col_max, max(row_min, 0), row_max


def load_dataset_paths(root_img_dir, root_annotation_dir, dirs):
    image_paths = []
    annotation_paths = []

    for i in range(len(dirs)):
        i_dir = root_img_dir + dirs[i] + '/'
        a_dir = root_annotation_dir + dirs[i] + '/'
        for file in os.listdir(i_dir):
            i_id = file[:-4]
            if os.path.exists(i_dir + i_id + '.tif'):
                image_paths.append(i_dir + i_id + '.tif')
            else:
                image_paths.append(i_dir + i_id + '.png')
            if os.path.exists(a_dir + i_id + '.json'):
                annotation_paths.append(a_dir + i_id + '.json')
            else:
                annotation_paths.append(a_dir + i_id + '_20X_YZ.json')
            

    if len(image_paths) == len(annotation_paths):
        return image_paths, annotation_paths
    else:
        return None, None


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]
