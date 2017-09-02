import threading
import queue
import cv2
import logging


class VideoStreamThread(threading.Thread):
    def __init__(self, image_q, draw_object_q, draw_command_q, start_detect_ev, track_ev, follow_ev):
        threading.Thread.__init__(self, name='Video Thread')
        self.image_q = image_q
        self.draw_object_q = draw_object_q
        self.draw_command_q = draw_command_q
        self.start_detect_ev = start_detect_ev
        self.track_ev = track_ev
        self.follow_ev = follow_ev

        # For drawing
        self.current_objects = None
        self.frames_since_detection = 0
        self.speed_str = None
        self.steer_str = None

    def run(self):
        self._webcam_stream()

    def _webcam_stream(self):
        cap = cv2.VideoCapture(0)

        logging.debug('Starting video')
        frame_no = 0
        while True:
            # Capture frame-by-frame
            ret, image = cap.read()
            image_h, image_w, _ = image.shape

            # Add frame to object detection queue if detect event is set
            if self.start_detect_ev.is_set():
                frame_no += 1
                self.image_q.put(image)
                logging.debug('Image %s sent to detector, shape: %s', frame_no, image.shape)
                self.start_detect_ev.clear()

            # Check for newly detected objects
            try:
                self.current_objects = self.draw_object_q.get_nowait()
                self.draw_object_q.task_done()
                self.frames_since_detection = 0
            except queue.Empty:
                # Stop drawing objects if too much time has passed
                if self.current_objects is not None:
                    if self.frames_since_detection == 30:
                        self.current_objects = None
                        self.speed_str = None
                        self.steer_str = None
                        self.frames_since_detection = 0
                    else:
                        self.frames_since_detection += 1

            try:
                command_dict = self.draw_command_q.get_nowait()
                self.draw_command_q.task_done()
                self.speed_str = 'Speed: {:4.2f}'.format(command_dict['speed'])
                self.steer_str = 'Steer: {:4.2f}'.format(command_dict['steer'])
            except queue.Empty:
                pass

            # Draw objects if any recently detected
            if self.current_objects is not None:
                # Draw rectangle and label for each object
                overlay = image.copy()
                for obj in self.current_objects:
                    name = obj['class']
                    y1, _, y2, _ = obj['box'] * image_h
                    _, x1, _, x2 = obj['box'] * image_w

                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_4)
                    cv2.putText(image, str.upper(name),
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    cv2.putText(overlay, str.upper(name),
                                (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    # Draw control commands on frame
                    if self.steer_str is not None and self.speed_str is not None:
                        cv2.putText(image, self.speed_str,
                                    (5, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(image, self.steer_str,
                                    (5, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(overlay, self.speed_str,
                                    (5, 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(overlay, self.steer_str,
                                    (5, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Make label and box increasingly transparent
                    alpha = 0.9 - (0.9 * self.frames_since_detection / 30)
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Draw current frame with objects highlighted
            cv2.imshow('frame', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        logging.debug('Closing video')
        self.track_ev.clear()
        self.follow_ev.clear()
        cap.release()
        cv2.destroyAllWindows()
