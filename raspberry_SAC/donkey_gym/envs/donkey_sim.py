# Original author: Tawn Kramer

import asyncore
import base64
import math
import time
from io import BytesIO
from threading import Thread

import numpy as np
from PIL import Image

from config import INPUT_DIM, ROI, THROTTLE_REWARD_WEIGHT, MAX_THROTTLE, MIN_THROTTLE, \
    REWARD_CRASH, CRASH_SPEED_WEIGHT, BASE_REWARD
from donkey_gym.core.fps import FPSTimer
from donkey_gym.core.tcp_server import IMesgHandler, SimServer

#from utils.arduino_utils import ArduinoTransmitter
from utils.tcp_utils import TcpServer
import cv2

import struct

import matplotlib.pyplot as plt
#import picamera
#from picamera.array import PiYUVArray, PiRGBArray





class DonkeyUnitySimContoller:
    """
    Wrapper for communicating with unity simulation.

    :param level: (int) Level index
    :param port: (int) Port to use for communicating with the simulator
    :param max_cte_error: (float) Max cross track error before reset
    """

    def __init__(self, level, port=9090, max_cte_error=3.0):
        self.level = level
        self.verbose = False

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM

        self.address = ('0.0.0.0', port)

        # Socket message handler
        self.handler = DonkeyUnitySimHandler(level, max_cte_error=max_cte_error)
        # Create the server to which the unity sim will connect
        #self.server = SimServer(self.address, self.handler)
        # Start the Asynchronous socket handler thread
        #self.thread = Thread(target=asyncore.loop)
        #self.thread.daemon = True
        #self.thread.start()


    def close_connection(self):
        return True

    def wait_until_loaded(self):
        """
        Wait for a client (Unity simulator).
        """
        while not self.handler.loaded:
            print("Waiting for sim to start..."
                  "if the simulation is running, press EXIT to go back to the menu")
            time.sleep(3.0)

    def reset(self):
        self.handler.reset()
        #self.handler.send_regen_road(self.handler.road_style, self.handler.seed, self.handler.turn_increment)

    def get_sensor_size(self):
        """
        :return: (int, int, int)
        """
        return self.handler.get_sensor_size()

    def take_action(self, action):
        self.handler.take_action(action)

    def observe(self):
        """
        :return: (np.ndarray)
        """
        return self.handler.observe()

    def quit(self):
        pass

    def render(self, mode):
        pass

    def is_game_over(self):
        return self.handler.is_game_over()

    def calc_reward(self, done):
        return self.handler.calc_reward(done)


class DonkeyUnitySimHandler(IMesgHandler):
    """
    Socket message handler.

    :param level: (int) Level ID
    :param max_cte_error: (float) Max cross track error before reset
    """

    def __init__(self, level, max_cte_error=3.0, cte_reduction=0):
        self.level_idx = level
        self.sock = None
        self.loaded = False
        self.verbose = False
        self.timer = FPSTimer(verbose=0)
        self.max_cte_error = max_cte_error

        # sensor size - height, width, depth
        self.camera_img_size = INPUT_DIM
        self.image_array = np.zeros(self.camera_img_size)
        self.original_image = None
        self.last_obs = None
        self.last_throttle = 0.0
        # Disabled: hit was used to end episode when bumping into an object
        self.hit = "none"
        # Cross track error
        self.cte = 0.0
        self.out_of_bounds = bytes([255,255,255,255,255,255,255,255,255,255,0])
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.steering_angle = 0.0
        self.current_step = 0
        self.speed = 0
        self.steering = None
        self.seed = 0
        self.road_style = 0
        self.turn_increment = 1

        self.cte_reduction = cte_reduction

        self.TcpServer = TcpServer("192.168.1.128", 33333)
        self.TcpServer.open_connection()

        # Define which method should be called
        # for each type of message
        '''
        self.fns = {'telemetry': self.on_telemetry,
                    "scene_selection_ready": self.on_scene_selection_ready,
                    "scene_names": self.on_recv_scene_names,
                    "car_loaded": self.on_car_loaded}
        '''
    '''
    def on_connect(self, socket_handler):
        """
        :param socket_handler: (socket object)
        """
        self.sock = socket_handler

    def on_disconnect(self):
        """
        Close socket.
        """
        self.sock.close()
        self.sock = None

    def on_recv_message(self, message):
        """
        Distribute the received message to the appropriate function.

        :param message: (dict)
        """
        if 'msg_type' not in message:
            print('Expected msg_type field')
            return

        msg_type = message['msg_type']
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            print('Unknown message type', msg_type)
    '''

    def reset(self):
        """
        Global reset, notably it
        resets car to initial position.
        """
        #print("Reset\n")
        if self.verbose:
            print("resetting")
        self.image_array = np.zeros(self.camera_img_size)
        self.last_obs = None
        self.hit = "none"
        self.cte = 0.0
        self.out_of_bounds = bytes([255,255,255,255,255,255,255,255,255,255,0])
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.current_step = 0
        #self.send_reset_car()
        #self.send_control(0, 0)
        time.sleep(1.0)
        self.timer.reset()

    def get_sensor_size(self):
        """
        :return: (tuple)
        """
        return self.camera_img_size

    def take_action(self, action):
        """
        :param action: ([float]) Steering and throttle
        """
        #print("Take action\n")
        if self.verbose:
            print("take_action")

        throttle = action[1]
        self.steering = action[0]
        self.last_throttle = throttle
        '''
        if self.speed < 0.3:
            throttle = max(throttle, 0.1)
        '''
        self.current_step += 1

        steering = int((max(min(self.steering, 0.999), -1) + 1) * 128)

        throttle = int((max(min(self.last_throttle, 0.999), -1) + 1) * 128)

        tcp_message = bytes([1]) + bytes([steering]) + bytes([throttle])

        #print("Steering: {}".format(self.steering))
        #print("Throttle: {}\n".format(self.last_throttle))

        self.TcpServer.send_message(tcp_message)
        #arduino_data = self.arduino.receive_data(1)
        #self.speed = int(arduino_data[0])
        #self.send_control(self.steering, throttle)

    def observe(self):
        #while self.last_obs is self.image_array:
        #    time.sleep(1.0 / 120.0)

        #self.last_obs = self.image_array

        #print("Observe\n")

        t = time.time()

        tcp_message = bytes([2])

        self.TcpServer.send_message(tcp_message)

        time_send = time.time() - t
        t = time.time()

        #size = (int).from_bytes(self.TcpServer.receive_message(4), byteorder='big')
        #print(size)

        #image_stream = self.TcpServer.receive_message(size)
        #image_stream = self.TcpServer.receive_message(38400)
        image_stream = self.TcpServer.receive_message(139)

        self.out_of_bounds = image_stream[128:138]

        self.speed = image_stream[138] / 255.0
        
        image_stream = image_stream[:128]

        #print("Image received in {} sec".format(time.time()-t))

        time_rec = time.time() - t

        t = time.time()

        #self.image_array = cv2.imdecode(np.fromstring(image_stream, dtype=np.uint8), cv2.IMREAD_COLOR )

        #self.image_array = np.reshape(bytearray(image_stream), self.camera_img_size)
        self.original_image = self.image_array



        #print(len(image_stream))
        #print(self.image_array.shape)
        '''
        for idx_h, height in enumerate(self.image_array):
            for idx_w, width in enumerate(height):
                for idx_c, channel in enumerate(width):
                    #print("{}".format(idx_c + idx_w*self.camera_img_size[2] + idx_h*self.camera_img_size[1]))
                    self.image_array[idx_h, idx_w, idx_c] = int(image_stream[idx_c + idx_w*self.camera_img_size[2] + idx_h*self.camera_img_size[1]])
        '''
        #print("Image processed in {} sec".format(time.time()-t))

        #observation = self.image_array
        observation = np.frombuffer(image_stream, dtype=np.float32).reshape((1,32))

        done = self.is_game_over()
        reward = self.calc_reward(done)
        info = {"speed":self.speed,
                "cte":self.cte,
                "outofbounds":self.out_of_bounds,
                "throttle": self.last_throttle,
                "steering": self.steering}
        self.timer.on_frame()

        time_proc = time.time() - t
        
        if (time_rec + time_proc + time_send) >= 0.1:
            print("Image received in {} sec".format(time_rec))
        
        return observation, reward, done, info

    def is_game_over(self):
        """
        :return: (bool)
        """
        analog_result = np.zeros((5))
        for idx, _ in enumerate(analog_result):
            high_byte = self.out_of_bounds[idx * 2] << 8
            low_byte = self.out_of_bounds[idx * 2 + 1]
            analog_result[idx] = high_byte + low_byte
        for res in analog_result:
            if res < 150:
                return True
        #print(analog_result)

        #if self.out_of_bounds == 1:
        #    print("Error game over")
        #return self.out_of_bounds == 1
        #return self.hit != "none" or math.fabs(self.cte) > self.max_cte_error

        return False

    def calc_reward(self, done):
        """
        Compute reward:
        - +1 life bonus for each step + throttle bonus
        - -10 crash penalty - penalty for large throttle during a crash

        :param done: (bool)
        :return: (float)
        """
        if done:
            # penalize the agent for getting off the road fast
            norm_throttle = (self.last_throttle - MIN_THROTTLE) / (MAX_THROTTLE - MIN_THROTTLE)
            return REWARD_CRASH - CRASH_SPEED_WEIGHT * norm_throttle
        # 1 per timesteps + throttle
        throttle_reward = THROTTLE_REWARD_WEIGHT * (self.last_throttle / MAX_THROTTLE)
        return BASE_REWARD + throttle_reward

    # ------ Socket interface ----------- #

    def on_telemetry(self, data):
        """
        Update car info when receiving telemetry message.

        :param data: (dict)
        """
        img_string = data["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))
        # Resize and crop image
        image = np.array(image)
        # Save original image for render
        self.original_image = np.copy(image)
        # Resize if using higher resolution images
        # image = cv2.resize(image, CAMERA_RESOLUTION)
        # Region of interest
        r = ROI
        image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        # Convert RGB to BGR
        image = image[:, :, ::-1]
        self.image_array = image
        # Here resize is not useful for now (the image have already the right dimension)
        # self.image_array = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # name of object we just hit. "none" if nothing.
        # NOTE: obstacle detection disabled
        # if self.hit == "none":
        #     self.hit = data["hit"]

        self.x = data["pos_x"]
        self.y = data["pos_y"]
        self.z = data["pos_z"]
        self.steering_angle = data['steering_angle']
        self.speed = data["speed"]

        if self.max_cte_error > 1 and self.cte_reduction > 0:
            self.max_cte_error -= self.cte_reduction

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 3 scenes available now.
        try:
            self.cte = data["cte"]
            # print(self.cte)
        except KeyError:
            print("No Cross Track Error in telemetry")
            pass

    def on_scene_selection_ready(self, _data):
        """
        Get the level names when the scene selection screen is ready
        """
        print("Scene Selection Ready")
        self.send_get_scene_names()

    def on_car_loaded(self, _data):
        #if self.seed != 0:
        #self.send_regen_road(self.road_style, self.seed, self.turn_increment)

        if self.verbose:
            print("Car Loaded")
        self.loaded = True

    def on_recv_scene_names(self, data):
        """
        Select the level.

        :param data: (dict)
        """
        if data is not None:
            names = data['scene_names']
            if self.verbose:
                print("SceneNames:", names)
            self.send_load_scene(names[self.level_idx])

    def send_control(self, steer, throttle):
        """
        Send message to the server for controlling the car.

        :param steer: (float)
        :param throttle: (float)
        """
        if not self.loaded:
            return
        msg = {'msg_type': 'control', 'steering': steer.__str__(), 'throttle': throttle.__str__(), 'brake': '0.0'}
        self.queue_message(msg)

    def send_reset_car(self):
        """
        Reset car to initial position.
        """
        msg = {'msg_type': 'reset_car'}
        self.queue_message(msg)

    def send_regen_road(self, road_style=0, rand_seed=0, turn_increment=0.0):
        '''
        Regenerate the road, where available. For now only in level 0.
        In level 0 there are currently 5 road styles. This changes the texture on the road
        and also the road width.
        The rand_seed can be used to get some determinism in road generation.
        The turn_increment defaults to 1.0 internally. Provide a non zero positive float
        to affect the curviness of the road. Smaller numbers will provide more shallow curves.
        '''

        print("Seed: "+ str(rand_seed))
        print("Max_CTE: "+str(self.max_cte_error))
        msg = { 'msg_type' : 'regen_road',
            'road_style': road_style.__str__(),
            'rand_seed': rand_seed.__str__(),
            'turn_increment': turn_increment.__str__() }

        self.sock.queue_message(msg)
        self.seed += 1

    def send_get_scene_names(self):
        """
        Get the different levels availables
        """
        msg = {'msg_type': 'get_scene_names'}
        self.queue_message(msg)

    def send_load_scene(self, scene_name):
        """
        Load a level.

        :param scene_name: (str)
        """
        msg = {'msg_type': 'load_scene', 'scene_name': scene_name}
        self.queue_message(msg)

    def send_exit_scene(self):
        """
        Go back to scene selection.
        """
        msg = {'msg_type': 'exit_scene'}
        self.queue_message(msg)

    def queue_message(self, msg):
        """
        Add message to socket queue.

        :param msg: (dict)
        """
        if self.sock is None:
            if self.verbose:
                print('skipping:', msg)
            return

        if self.verbose:
            print('sending', msg)
        self.sock.queue_message(msg)
