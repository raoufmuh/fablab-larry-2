import numpy as np
import os
import tensorflow as tf
import time
import timeit
import threading
import queue

from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image
from pprint import pprint

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_DIR = os.path.join(ROOT_DIR, MODEL_NAME)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(MODEL_DIR, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(MODEL_DIR, 'mscoco_label_map.pbtxt')

PATH_TO_TEST_IMAGES_DIR = os.path.join(ROOT_DIR, 'test_images')
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 8)]

NUM_CLASSES = 90

# Size out output image in inches
IMAGE_SIZE = (12, 8)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


image_1 = load_image_into_numpy_array(Image.open(TEST_IMAGE_PATHS[0]))
image_2 = load_image_into_numpy_array(Image.open(TEST_IMAGE_PATHS[1]))
image_3 = load_image_into_numpy_array(Image.open(TEST_IMAGE_PATHS[2]))
image_4 = load_image_into_numpy_array(Image.open(TEST_IMAGE_PATHS[3]))


class ObjectDetector(threading.Thread):
    def __init__(self, image_q):
        threading.Thread.__init__(self)
        self.detection_graph = None
        self._label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self._categories = label_map_util.convert_label_map_to_categories(self._label_map,
                                                                          max_num_classes=NUM_CLASSES,
                                                                          use_display_name=True)
        self._category_index = label_map_util.create_category_index(self._categories)
        self._load_detection_graph()
        self.image_q = image_q

    def _load_detection_graph(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                # if download -> self.download_graph

    def _download_graph(self, graph_name):
        pass

    def run(self):
        self.detect()

    def detect(self):
        print('Running model...')
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                while True:
                    print('Waiting for image')
                    image_np = self.image_q.get()
                    self.image_q.task_done()

                    print('Starting to predict...')
                    ts = timeit.default_timer()

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    te = timeit.default_timer()
                    print('Prediction time: {}'.format(te - ts))

                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        self._category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)

                    # Add threshold accuracy (50%)
                    out_boxes = np.squeeze(boxes)[:1][0]
                    out_scores = np.squeeze(scores)[:1]
                    out_classes = np.squeeze(classes).astype(np.int32)[:1]
                    pprint(num_detections)

                    plt.figure(figsize=IMAGE_SIZE)
                    plt.imshow(image_np)
                    plt.waitforbuttonpress()

if __name__ == '__main__':
    image_q = queue.Queue(0)
    detect_thread = ObjectDetector(image_q)
    detect_thread.setDaemon(True)
    detect_thread.start()
    time.sleep(5)
    image_q.put(image_1)
    time.sleep(5)
    image_q.put(image_2)
    time.sleep(5)
    image_q.put(image_3)
    time.sleep(5)
    image_q.put(image_4)

    image_q.join()
