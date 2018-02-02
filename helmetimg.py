import numpy as np
import os
import tensorflow as tf
import cv2
import time

from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'helmet_inference_graph'
PATH_TO_CKPT = 'C:\\Users\\Saim Shaikh\\Desktop\\Project\\helmetdetection\\helmet_inference_graph\\frozen_inference_graph.pb'
PATH_TO_LABELS = 'C:\\Users\\Saim Shaikh\\Desktop\\Project\\helmetdetection\\data\\object-detection.pbtxt'
NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_images_from_folder(folder):
    images = []
    paths = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        path = os.path.join(folder, filename)
        if img is not None:
            images.append(img)
            paths.append(path)
    return images,paths

gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
def detect_helmet():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    img,paths = load_images_from_folder('C:\\Users\\Saim Shaikh\\Desktop\\Project\\motorcycle')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            for image_np,path in zip(img,paths):
                start = time.time()
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                if (max(np.squeeze(scores)) * 100) > 50:
                    os.remove(path)
                end = time.time()
                seconds = end - start
                print("Time taken : {0} seconds".format(seconds))

detect_helmet()