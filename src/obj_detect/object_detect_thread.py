import numpy as np
import os
import tensorflow as tf
import time
import timeit
import threading
import queue
import logging


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


class ObjectDetector(threading.Thread):
    def __init__(self, image_q, object_q, draw_object_q, finish_detect_ev, say_objects_ev):
        # Initialize threading parameters
        threading.Thread.__init__(self, name='Detector Thread')
        self.image_q = image_q
        self.object_q = object_q
        self.draw_object_q = draw_object_q
        self.finish_detect_ev = finish_detect_ev
        self.say_objects_ev = say_objects_ev
        # self.follow_target_ev = follow_target

        # Initialize detector parameters
        self.detection_graph = None
        self._label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self._categories = label_map_util.convert_label_map_to_categories(self._label_map,
                                                                          max_num_classes=NUM_CLASSES,
                                                                          use_display_name=True)
        self._category_index = label_map_util.create_category_index(self._categories)
        self._load_detection_graph()

    def _load_detection_graph(self):
        """Loads pre-trained detector model from file"""
        # logging.debug('Loading pre-trained model from file')
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
        # print(self._category_index)
        self.detect()

    def detect(self):
        logging.debug('Initialising TensorFlow session')
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                logging.debug('TensorFlow session initialized')
                while True:
                    logging.debug('Ready for image')
                    image_np = self.image_q.get()
                    self.image_q.task_done()

                    self.finish_detect_ev.clear()

                    logging.debug('Image received, running CNN')
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

                    # NOTE: might improve accuracy but slowed detection x3
                    # select_idx = tf.image.non_max_suppression(tf.squeeze(boxes), tf.squeeze(scores),
                    #                                           max_output_size=20, iou_threshold=0.5)
                    #
                    # select_boxes = tf.gather(tf.squeeze(boxes), select_idx)
                    # select_scores = tf.gather(tf.squeeze(scores), select_idx)
                    # select_classes = tf.gather(tf.squeeze(classes), select_idx)

                    # Actual detection.
                    (boxes, scores, classes) = sess.run(
                        [boxes, scores, classes],
                        feed_dict={image_tensor: image_np_expanded})

                    # remove single dimensional entries from output arrays
                    boxes = np.squeeze(boxes)
                    classes = np.squeeze(classes).astype(np.int32)
                    scores = np.squeeze(scores)

                    # Visualization of the results of a detection.
                    # vis_util.visualize_boxes_and_labels_on_image_array(
                    #     image_np,
                    #     boxes,
                    #     classes,
                    #     scores,
                    #     self._category_index,
                    #     use_normalized_coordinates=True,
                    #     line_thickness=8)

                    object_dict = self.get_top_objects(boxes, scores, classes, threshold_prob=0.5)

                    te = timeit.default_timer()

                    #  add to detected objects q
                    self.object_q.put(object_dict)
                    self.draw_object_q.put(object_dict)  # puts object in queue then quits?
                    self.finish_detect_ev.set()
                    logging.debug('Finished detection in %s seconds. %s objects found with threshold probability %s',
                                  format(te - ts),
                                  len(object_dict),
                                  0.5)

                    # Sends to read-out queue if tracking is off
                    if self.say_objects_ev.is_set():
                        # TEMP: consume results somehow - pass to voice command
                        consume = self.object_q.get()
                        self.object_q.task_done()
                        logging.debug('Consumed detected objects %s: ', consume)
                        self.say_objects_ev.clear()

    def get_top_objects(self, boxes, scores, classes, threshold_prob=0.5):
        # filter by output probability using passed threshold
        top_idx = np.where(scores >= threshold_prob)[0]
        # top_boxes = boxes[top_idx]
        # top_scores = scores[top_idx]
        # top_classes = [self._category_index[i]['name'] for i in classes[top_idx]]
        object_dict = [{'class': self._category_index[classes[i]]['name'],
                        'score': scores[i],
                        'box': boxes[i]} for i in top_idx]
        # return top_boxes, top_scores, top_classes
        return object_dict

# if __name__ == '__main__':
#     image_q = queue.Queue(0)
#     detect_thread = ObjectDetector(image_q)
#     detect_thread.setDaemon(True)
#     detect_thread.start()




