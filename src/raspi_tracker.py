import threading
import queue
import numpy as np
import cv2
import logging
import time

from obj_detect.object_detect_thread import ObjectDetector
from PID import PID
from recognizer.main import RecognizerThread
from raspi_video_thread import VideoStreamThread

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(threadName)-20s %(message)s',
)


# PID thread to send commands
# PID object as part of tracker module (these will always be synchronous, so its easier

class SerialThread(threading.Thread):
    """"""
    def __init__(self, co):
        threading.Thread.__init__(self)
        # self.steer_control = PID(p=p, i=i, d=d)
        # self.speed_control = PID(p=p, i=i, d=d)


class TrackThread(threading.Thread):
    def __init__(self, object_q, target_q, auto_command_q, draw_command_q,
                 start_detect_ev, finish_detect_ev, track_ev, follow_ev):
        threading.Thread.__init__(self, name='Control Thread')
        self.object_q = object_q
        self.target_q = target_q
        self.auto_command_q = auto_command_q
        self.draw_command_q = draw_command_q
        self.start_detect_ev = start_detect_ev
        self.finish_detect_ev = finish_detect_ev
        self.track_ev = track_ev
        self.follow_ev = follow_ev

        self.current_target = None
        self.current_objects = None
        self.target_box = None
        self.frames_without_target_count = 0  # counts how many times a detection iterations did not find target

        # Steering and speed controlled by PID controller
        # Error is difference between centre of box and centre of frame
        self.steer_control = PID(p=2.0, i=0.5, d=0.5)
        self.steer_control.set_point = 0.5

        # Error is difference between base of box and base of frame
        self.speed_control = PID(p=2.0, i=0.0, d=0.5)
        self.speed_control.set_point = 0.9

    def run(self):
        while True:
            self.track_ev.wait()
            self.finish_detect_ev.wait()
            # start detection
            self.start_detect_ev.set()

            # or just use existing detection to track

            # obtains objects once detection has finished
            self.current_objects = self.object_q.get()
            self.object_q.task_done()

            # checks target queue for new (or initial) targets
            if not self.target_q.empty():
                self.current_target = self.target_q.get()
                self.target_q.task_done()

            if self.follow_ev.is_set():
                if self.target_detected():
                    self.update_control()
                # Feed into PID
                # Stack serial queue

            # self.follow()

    def follow(self):
        if self.current_target is not None:
            logging.debug('Following target %s from objects in %s', self.current_target, self.current_objects)

    def target_detected(self):
        # Get all boxes of the correct class name
        target_box_options = [obj['box'] for obj in self.current_objects if obj['class'] == self.current_target]

        # Find box closest to camera
        if len(target_box_options) == 0:
            # Target was not found in frame in detection attempt
            if self.frames_without_target_count == 5:
                # Target has not been seen in 5 attempts
                logging.debug('Lost target \"%s\", cancel follow.', self.current_target)
                self.target_box = None
                self.frames_without_target_count = 0
                self.track_ev.clear()
                self.follow_ev.clear()
                return False
            elif self.target_box is None:
                # No previous trajectory to follow
                logging.debug('Target \"%s\" not found: cancelling follow target command', self.current_target)
                self.track_ev.clear()
                self.follow_ev.clear()
                return False
            else:
                # Continue assuming target is close to where it was last seen
                self.frames_without_target_count += 1
                logging.debug('Continuing previous trajectory - target \"%s\" not detected in previous %s frames',
                              self.current_target, self.frames_without_target_count)

        elif len(target_box_options) == 1:
            # Select the only possible option
            self.target_box = target_box_options[0]
            logging.debug('Target found - following \"%s\"', self.current_target)

        else:
            # Select the closest box
            # TODO: make better way of choosing
            logging.debug('Multiple targets found - following closest \"%s\"', self.current_target)
            y_values = [box[2] for box in target_box_options]
            _, idx = max((val, idx) for (idx, val) in enumerate(y_values))
            self.target_box = target_box_options[idx]

        return True

    def update_control(self):
        y1, x1, y2, x2 = self.target_box
        speed_command = self.speed_control.update(y2)
        steer_command = self.steer_control.update((x2+x1)/2.0)
        command_dict = {'steer': steer_command, 'speed': speed_command}
        logging.debug('Command to serial: %s', command_dict)
        self.auto_command_q.put(command_dict)
        self.draw_command_q.put(command_dict)


class VoiceControlThread(threading.Thread):
    def __init__(self, voice_in_q, voice_out_q, object_q, target_q, say_objects_q,
                 start_detect_ev, say_objects_ev, track_ev, follow_ev):
        threading.Thread.__init__(self, name='Voice Command Thread')
        self.voice_in_q = voice_in_q
        self.voice_out_q = voice_out_q
        self.say_objects_q = say_objects_q
        self.start_detect_ev = start_detect_ev
        self.say_objects_ev = say_objects_ev
        self.track_ev = track_ev
        self.follow_ev = follow_ev
        self.object_q = object_q
        self.target_q = target_q

    def run(self):
        while True:
            # loop while waiting for voice commands
            voice_in = self.voice_in_q.get()
            self.voice_in_q.task_done()
            logging.debug('Received voice command %s', voice_in)

            if voice_in.get('detect', False):
                self.what_do_you_see()
            elif voice_in.get('track', False):
                self.track_objects()
            elif voice_in.get('follow', False):
                self.follow_object(voice_in['follow'])
            elif voice_in.get('stop follow', False):
                self.stop_following()

            voice_in = {}

    def what_do_you_see(self):
        logging.debug('COMMAND: What do you see?')
        self.say_objects_ev.set()
        self.start_detect_ev.set()
        # # TODO: fetch object from queue and feed into other
        # objects = self.say_objects_q.get()
        # self.say_objects_q.task_done()
        # # convert objects to neat string
        # self.voice_out_q.put(objects)

    def track_objects(self):
        self.track_ev.set()

    def follow_object(self, object_name):
        logging.debug('COMMAND: Start following object: \"%s\"', object_name)
        self.target_q.put(object_name)
        self.track_ev.set()
        self.follow_ev.set()

    def stop_following(self):
        self.track_ev.clear()
        self.follow_ev.clear()


class ThreadingServer(object):
    # Queues for voice thread to interact with sightly-modified Google scripts
    voice_in_queue = queue.Queue(0)
    voice_out_queue = queue.Queue(0)
    say_objects_queue = queue.Queue(0)

    # Queues for communication between threads
    image_queue = queue.Queue(0)
    object_queue = queue.Queue(0)
    target_queue = queue.Queue(0)

    auto_command_queue = queue.Queue()

    # Queues for video labelling
    draw_object_queue = queue.Queue(0)
    draw_target_queue = queue.Queue(0)
    draw_command_queue = queue.Queue(0)

    start_detect_event = threading.Event()
    finish_detect_event = threading.Event()
    track_event = threading.Event()
    follow_event = threading.Event()
    say_objects_event = threading.Event()

    recognizer_thread = RecognizerThread(voice_in_q=voice_in_queue,
                                         voice_out_q=voice_out_queue)

    voice_thread = VoiceControlThread(voice_in_q=voice_in_queue,
                                      voice_out_q=voice_out_queue,
                                      object_q=object_queue,
                                      target_q=target_queue,
                                      say_objects_q=say_objects_queue,
                                      start_detect_ev=start_detect_event,
                                      say_objects_ev=say_objects_event,
                                      track_ev=track_event,
                                      follow_ev=follow_event)

    video_thread = VideoStreamThread(image_q=image_queue,
                                     draw_object_q=draw_object_queue,
                                     draw_command_q=draw_command_queue,
                                     start_detect_ev=start_detect_event,
                                     track_ev=track_event,
                                     follow_ev=follow_event)

    detect_thread = ObjectDetector(image_q=image_queue,
                                   object_q=object_queue,
                                   draw_object_q=draw_object_queue,
                                   finish_detect_ev=finish_detect_event,
                                   say_objects_ev=say_objects_event)

    track_thread = TrackThread(object_q=object_queue,
                               target_q=target_queue,
                               auto_command_q=auto_command_queue,
                               draw_command_q=draw_command_queue,
                               start_detect_ev=start_detect_event,
                               finish_detect_ev=finish_detect_event,
                               track_ev=track_event,
                               follow_ev=follow_event)

    voice_thread.setDaemon(True)
    voice_thread.start()

    video_thread.setDaemon(True)
    video_thread.start()

    detect_thread.setDaemon(True)
    detect_thread.start()

    track_thread.setDaemon(True)
    track_thread.start()

    recognizer_thread.start()

    # logging.debug('Pausing to allow all threads to initialize')
    # time.sleep(5)
    # voice_in_queue.put({'detect': True})
    #
    # time.sleep(10)
    # voice_in_queue.put({'follow': 'person'})
    # # track_objects(track_event)
    #
    # time.sleep(30)
    #
    # voice_in_queue.put({'stop follow': True})
    # time.sleep(5)
    # follow_object('lost', target_queue, track_event, follow_event)

    image_queue.join()
    object_queue.join()
    draw_object_queue.join()
    target_queue.join()


if __name__ == '__main__':
    ThreadingServer()
