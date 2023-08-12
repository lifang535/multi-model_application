import cv2
import os
import sys
import time
import random
import numpy as np
from math import ceil

import logging
from logger import logger_loader, logger_model_1, logger_model_2, logger_model_3, logger_model_4, logger_model_5
from logger import logger_model_1_rate, logger_model_2_rate, logger_model_3_rate, logger_model_4_rate, logger_model_5_rate
# from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import threading
import multiprocessing
from concurrent import futures

from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import YolosImageProcessor, YolosForObjectDetection
import torch
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.cuda.empty_cache()

car_num = 0
person_num = 0

# Load a font with a larger size
font_size = 16
font = ImageFont.truetype("../fonts/times new roman.ttf", font_size)

car_timer = time.time()
person_timer = time.time()

model_1_lock = multiprocessing.Lock()
model_2_lock = multiprocessing.Lock()
model_3_lock = multiprocessing.Lock()
model_4_lock = multiprocessing.Lock()
model_5_lock = multiprocessing.Lock()

class Loader(multiprocessing.Process):
    def __init__(self, input_video_paths_list):
        super().__init__()
        self.input_video_paths_list = input_video_paths_list

    def run(self):
        print("[Loader] start")
        logger_loader.info("[Loader] start")
        input_video_dir = "../input_videos"
        input_video_paths = [os.path.join(input_video_dir, filename) for filename in os.listdir(input_video_dir)]
        input_video_paths.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

        for _ in range(0):
            input_video_paths.pop(-1)

        for input_video_path in input_video_paths:
            self.input_video_paths_list.append(input_video_path)
            logger_loader.info(f"[Loader] input_video_path: {input_video_path}")
            # time.sleep(random.randint(2, 4))
            time.sleep(10)
        
        while True:
            time.sleep(0.01)
            if len(self.input_video_paths_list) == 0:
                print("[Loader] end")
                logger_loader.info("[Loader] end")
                self.input_video_paths_list.append(-1)
                break


class Model_1(multiprocessing.Process):
    def __init__(self, id, input_video_paths_list, car_frames_list, person_frames_list, draw_message_list, end_signal, to_monitor_rate):
        super().__init__()
        self.id = id
        self.input_video_paths_list = input_video_paths_list
        self.car_frames_list = car_frames_list
        self.person_frames_list = person_frames_list
        self.draw_message_list = draw_message_list
        self.end_signal = end_signal
        
        self.device = None
        self.model = None
        self.processor = None

        self.timer_logger_model_1 = time.time()
        self.to_monitor_rate = to_monitor_rate

    def run(self):
        self.device = torch.device("cuda:0")
        self.model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny').to(self.device)
        self.processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        
        if self.id == 1:
            thread_monitor_rate = threading.Thread(target=self.monitor_rate)
            thread_monitor_rate.start()

        self.end_signal.value += 1

        print(f"[Model_1_{self.id}] start")
        logger_model_1.info(f"[Model_1_{self.id}] start")
        while True:
            time.sleep(0.01)
            input_video_path = None
            with model_1_lock:
                if len(self.input_video_paths_list) > 0:
                    if self.input_video_paths_list[0] == -1:
                        if len(self.car_frames_list) == 0 and len(self.person_frames_list) == 0:
                            print(f"[Model_1_{self.id}] end")
                            logger_model_1.info(f"[Model_1_{self.id}] end")
                            self.end_signal.value -= 1
                            # print(f"[Model_1_{self.id}] self.end_signal.value: {self.end_signal.value}")
                            logger_model_1.info(f"[Model_1_{self.id}] self.end_signal.value: {self.end_signal.value}")
                            if self.end_signal.value == 0:
                                self.car_frames_list.append(-1)
                                self.person_frames_list.append(-1)
                            break
                        else:
                            continue
                    input_video_path = self.input_video_paths_list.pop(0)
            if input_video_path is not None:
                self.process_video(input_video_path)

    def monitor_rate(self):
        rates = []
        sliding_window_size = 1
        last_file_video_path = ""
        last_input_video_paths_list_len = 0
        while True:
            time.sleep(1e-6)
            with model_1_lock:
                if len(self.input_video_paths_list) > 0 and self.input_video_paths_list[0] == -1:
                    break
                if (len(self.input_video_paths_list) > 0 and self.input_video_paths_list[-1] != last_file_video_path) or len(self.input_video_paths_list) > last_input_video_paths_list_len:
                    self.to_monitor_rate.append(time.time())
                    last_file_video_path = self.input_video_paths_list[-1]
                last_input_video_paths_list_len = len(self.input_video_paths_list)

                if len(self.to_monitor_rate) > 1:
                    rate = round((len(self.to_monitor_rate) - 1) / (self.to_monitor_rate[-1] - self.to_monitor_rate[0]), 3)
                    rates.append(rate)
                    if len(rates) > sliding_window_size:
                        rates.pop(0)
                    total_weight = sum(range(1, len(rates) + 1))
                    weighted_sum = sum((i + 1) * rate for i, rate in enumerate(rates))
                    moving_average = round(weighted_sum / total_weight, 3)
                    # print(f"[Model_1_{self.id}] rate: {moving_average}")
                    logger_model_1.info(f"[Model_1_{self.id}] rate: {moving_average}")
                    logger_model_1_rate.info(f"{moving_average}")
                    self.to_monitor_rate[:] = self.to_monitor_rate[-1:]
    

    def process_video(self, input_video_path):
        # Input video file path
        input_video_path = input_video_path
        print(f"[Model_1_{self.id}] input_video_path: ", input_video_path)
        logger_model_1.info(f"[Model_1_{self.id}] input_video_path: {input_video_path}")

        # Output directory for frames
        output_frames_dir = "output_frames_" + input_video_path.split('/')[-1].split('.')[0]
        output_video_dir = "../output_videos"
        os.makedirs(output_frames_dir, exist_ok=True)
        os.makedirs(output_video_dir, exist_ok=True)

        # Output video file path
        output_video_path = "../output_videos/processed_" + input_video_path.split('/')[-1]
        print(f"[Model_1_{self.id}] output_video_path: ", output_video_path)
        logger_model_1.info(f"[Model_1_{self.id}] output_video_path: {output_video_path}")

        # Open the video file
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save the processed frame as a JPG file
            frame_filename = os.path.join(output_frames_dir, f"frame_{int(cap.get(1))}.jpg")
            cv2.imwrite(frame_filename, frame)

            # Write the processed frame to the output video
            out.write(frame)

            # Process the frame (example: apply a filter)
            self.process_frame(frame, output_frames_dir + "/frame_" + str(int(cap.get(1))) + ".jpg")
        # Release the video capture and writer objects
        cap.release()
        out.release()

        self.car_frames_list.append(int(input_video_path.split('_')[-1].split('.')[0]))
        self.person_frames_list.append(int(input_video_path.split('_')[-1].split('.')[0]))

    def process_frame(self, frame, frame_filename):
        if time.time() - self.timer_logger_model_1 > 5:
            logger_model_1.info(f"[Model_1_{self.id}] frame_filename: {frame_filename}")
            self.timer_logger_model_1 = time.time()

        image = Image.fromarray(frame)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}  # move input data to GPU
        with torch.no_grad():  # execute model inference, make sure we do not compute gradientss
            outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        car = False
        person = False

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            if self.model.config.id2label[label.item()] == "car":
                car = True
            if self.model.config.id2label[label.item()] == "person":
                person = True

        if car:
            self.car_frames_list.append([frame, frame_filename])
        if person:
            self.person_frames_list.append([frame, frame_filename])

        return
    
    
class Model_2(multiprocessing.Process):
    def __init__(self, id, car_frames_list, draw_message_list, to_monitor_rate):
        super().__init__()
        self.id = id
        self.car_frames_list = car_frames_list
        self.draw_message_list = draw_message_list
        
        self.device = None
        self.model = None
        self.processor = None

        self.timer_logger_model_2 = time.time()
        self.to_monitor_rate = to_monitor_rate

    def run(self):
        self.device = torch.device("cuda:1")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

        if self.id == 1:
            thread_monitor_rate = threading.Thread(target=self.monitor_rate)
            thread_monitor_rate.start()

        print(f"[Model_2_{self.id}] start")
        logger_model_2.info(f"[Model_2_{self.id}] start")
        while True:
            time.sleep(0.01)
            frame = None
            with model_2_lock:
                if len(self.car_frames_list) > 0:
                    try:
                        if self.car_frames_list[0] == -1:
                            print(f"[Model_2_{self.id}] end")
                            logger_model_2.info(f"[Model_2_{self.id}] end")
                            self.draw_message_list.append(-1)
                            break
                        elif isinstance(self.car_frames_list[0], int):
                            # print(f"[Model_2_{self.id}] video_{self.car_frames_list[0]} is being processed")
                            video_id = self.car_frames_list.pop(0)
                            self.draw_message_list.append(video_id)
                            continue
                        frame = self.car_frames_list.pop(0)
                        if time.time() - self.timer_logger_model_2 > 5:
                            # print(f"[Model_2_{self.id}] frame_filename: ", frame[1])
                            logger_model_2.info(f"[Model_2_{self.id}] frame_filename: {frame[1]}, and car_frames_list: {len(self.car_frames_list)}")
                            self.timer_logger_model_2 = time.time()
                    except Exception as e:
                        logger_model_2.error(f"[Model_2_{self.id}] {e}")
            if frame is not None:
                self.process_car_frame(frame[0], frame[1])

    def monitor_rate(self):
        rates = []
        sliding_window_size = 5
        last_car_frame = ""
        last_car_frames_list_len = 0
        while True:
            time.sleep(1e-6)
            with model_2_lock:
                if len(self.car_frames_list) > 0 and self.car_frames_list[0] == -1:
                    break
                try: 
                    if (len(self.car_frames_list) > 0 and self.car_frames_list[-1][1] != last_car_frame) or len(self.car_frames_list) > last_car_frames_list_len:
                        self.to_monitor_rate.append(time.time())
                        last_car_frame = self.car_frames_list[-1][1]
                    last_car_frames_list_len = len(self.car_frames_list)
                except Exception as e:
                    logger_model_2.warning(f"[Model_2_{self.id}] {e}, and car_frames_list[-1]: {self.car_frames_list[-1]}, and last_car_frame: {last_car_frame}")

                if len(self.to_monitor_rate) > 1:
                    rate = round((len(self.to_monitor_rate) - 1) / (self.to_monitor_rate[-1] - self.to_monitor_rate[0]), 3)
                    rates.append(rate)
                    if len(rates) > sliding_window_size:
                        rates.pop(0)
                    total_weight = sum(range(1, len(rates) + 1))
                    weighted_sum = sum((i + 1) * rate for i, rate in enumerate(rates))
                    moving_average = round(weighted_sum / total_weight, 3)
                    # print(f"[Model_2_{self.id}] rate: {moving_average}")
                    logger_model_2.info(f"[Model_2_{self.id}] rate: {moving_average}")
                    logger_model_2_rate.info(f"{moving_average}")
                    self.to_monitor_rate[:] = self.to_monitor_rate[-1:]

    def process_car_frame(self, frame, frame_filename):
        image = Image.fromarray(frame)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}  # move input data to GPU
        with torch.no_grad():  # execute model inference, make sure we do not compute gradientss
            outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        # Filter out results for labels other than "car"
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if self.model.config.id2label[label.item()] == "car":
                box = [round(i, 2) for i in box.tolist()]

                global car_timer, car_num
                if time.time() - car_timer > 1:
                    if self.model.config.id2label[label.item()] == "car":
                        car_num += 1
                        car_timer = time.time()

                self.draw_message_list.append([frame_filename, round(score.item(), 3), self.model.config.id2label[label.item()], box])
        return
    
class Model_3(multiprocessing.Process):
    def __init__(self, id, person_frames_list, draw_message_list, to_monitor_rate):
        super().__init__()
        self.id = id
        self.person_frames_list = person_frames_list
        self.draw_message_list = draw_message_list
        
        self.device = None
        self.model = None
        self.processor = None

        self.timer_logger_model_3 = time.time()
        self.to_monitor_rate = to_monitor_rate

    def run(self):
        self.device = torch.device("cuda:0")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101").to(self.device)
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101")

        if self.id == 1:
            thread_monitor_rate = threading.Thread(target=self.monitor_rate)
            thread_monitor_rate.start()

        print(f"[Model_3_{self.id}] start")
        logger_model_3.info(f"[Model_3_{self.id}] start")
        while True:
            time.sleep(0.01)
            frame = None
            with model_3_lock:
                if len(self.person_frames_list) > 0:
                    try:
                        if self.person_frames_list[0] == -1:
                            print(f"[Model_3_{self.id}] end")
                            logger_model_3.info(f"[Model_3_{self.id}] end")
                            self.draw_message_list.append(-1)
                            break
                        elif isinstance(self.person_frames_list[0], int):
                            # print(f"[Model_3_{self.id}] video_{self.person_frames_list[0]} is being processed")
                            video_id = self.person_frames_list.pop(0)
                            self.draw_message_list.append(video_id)
                            continue
                        frame = self.person_frames_list.pop(0)
                        if time.time() - self.timer_logger_model_3 > 5:
                            # print(f"[Model_3_{self.id}] frame_filename: ", frame[1])
                            logger_model_3.info(f"[Model_3_{self.id}] frame_filename: {frame[1]}, and person_frames_list: {len(self.person_frames_list)}")
                            self.timer_logger_model_3 = time.time()
                    except Exception as e:
                        logger_model_3.error(f"[Model_3_{self.id}] {e}")
            if frame is not None:
                self.process_person_frame(frame[0], frame[1])

    def monitor_rate(self):
        rates = []
        sliding_window_size = 5
        last_person_frame = ""
        last_person_frames_list_len = 0
        while True:
            time.sleep(1e-6)
            with model_3_lock:
                if len(self.person_frames_list) > 0 and self.person_frames_list[0] == -1:
                    break
                try:
                    if (len(self.person_frames_list) > 0 and self.person_frames_list[-1][1] != last_person_frame) or len(self.person_frames_list) > last_person_frames_list_len:
                        self.to_monitor_rate.append(time.time())
                        last_person_frame = self.person_frames_list[-1][1]
                    last_person_frames_list_len = len(self.person_frames_list)
                except Exception as e:
                    logger_model_3.warning(f"[Model_3_{self.id}] {e}, and person_frames_list[-1]: {self.person_frames_list[-1]}, and last_person_frame: {last_person_frame}")

                if len(self.to_monitor_rate) > 1:
                    rate = round((len(self.to_monitor_rate) - 1) / (self.to_monitor_rate[-1] - self.to_monitor_rate[0]), 3)
                    rates.append(rate)
                    if len(rates) > sliding_window_size:
                        rates.pop(0)
                    total_weight = sum(range(1, len(rates) + 1))
                    weighted_sum = sum((i + 1) * rate for i, rate in enumerate(rates))
                    moving_average = round(weighted_sum / total_weight, 3)
                    # print(f"[Model_3_{self.id}] rate: {moving_average}")
                    logger_model_3.info(f"[Model_3_{self.id}] rate: {moving_average}")
                    logger_model_3_rate.info(f"{moving_average}")
                    self.to_monitor_rate[:] = self.to_monitor_rate[-1:]

    def process_person_frame(self, frame, frame_filename):
        image = Image.fromarray(frame)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}  # move input data to GPU
        with torch.no_grad():  # execute model inference, make sure we do not compute gradientss
            outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        
        # Filter out results for labels other than "person"
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if self.model.config.id2label[label.item()] == "person":
                box = [round(i, 2) for i in box.tolist()]

                global person_timer, person_num
                if time.time() - person_timer > 5:
                    if self.model.config.id2label[label.item()] == "person":
                        person_num += 1
                        person_timer = time.time()

                self.draw_message_list.append([frame_filename, round(score.item(), 3), self.model.config.id2label[label.item()], box])
        return


class Model_4(multiprocessing.Process):
    def __init__(self, id, draw_message_list, images_being_processed, frame_files_to_be_processed, to_monitor_rate):
        super().__init__()
        self.id = id
        self.draw_message_list = draw_message_list

        # self.lock = threading.Lock()
        self.images_being_processed = images_being_processed
        self.frame_files_to_be_processed = frame_files_to_be_processed

        self.timer_logger_model_4 = time.time()
        self.to_monitor_rate = to_monitor_rate

    def run(self):
        print(f"[Model_4_{self.id}] start")
        logger_model_4.info(f"[Model_4_{self.id}] start")

        if self.id == 1:
            thread_monitor_rate = threading.Thread(target=self.monitor_rate)
            thread_monitor_rate.start()
        
        while True:
            time.sleep(0.01)
            with model_4_lock:
                if len(self.draw_message_list) > 0:
                    try:
                        if self.draw_message_list[0] == -1:
                            print(f"[Model_4_{self.id}] end")
                            logger_model_4.info(f"[Model_4_{self.id}] end")
                            self.frame_files_to_be_processed.append(-1)
                            break
                        elif isinstance(self.draw_message_list[0], int):
                            # print(f"[Model_4_{self.id}] video_{self.draw_message_list[0]} is being processed")
                            video_id = self.draw_message_list.pop(0)
                            self.frame_files_to_be_processed.append(video_id)
                            continue
                        batch_size = 0
                        for draw_message in self.draw_message_list:
                            if isinstance(draw_message, list):
                                batch_size += 1
                            else:
                                break
                        # batch_size = min(len(self.draw_message_list), 20)
                        batch_draw_messages = [self.draw_message_list.pop(0) for _ in range(batch_size)]
                        threads = []
                        # print(len(batch_draw_messages))
                        for draw_message in batch_draw_messages:
                            self.images_being_processed.append(draw_message[0])  # TODO: consider using a lock
                            if time.time() - self.timer_logger_model_4 > 5:
                                # print(f"[Model_4_{self.id}] {draw_message[0]} is being processed, and draw_message_list: {len(self.draw_message_list)}")
                                logger_model_4.info(f"[Model_4_{self.id}] {draw_message[0]} is being processed, and draw_message_list: {len(self.draw_message_list)}")
                                self.timer_logger_model_4 = time.time()
                            if self.images_being_processed.count(draw_message[0]) > 1:
                                self.images_being_processed.remove(draw_message[0])
                                batch_draw_messages.append(draw_message)
                                continue
                            thread = threading.Thread(target=self.process_draw_message, args=(draw_message,))
                            threads.append(thread)
                            thread.start()
                            """
                            draw_message = self.draw_message_list.pop(0)
                            thread = threading.Thread(target=self.process_draw_message, args=(draw_message,))
                            thread.start()
                            """
                            #self.process_draw_message(draw_message)
                    except Exception as e:
                        logger_model_4.error(f"[Model_4_{self.id}] {e}")

    def monitor_rate(self):
        rates = []
        sliding_window_size = 10
        last_draw_message = ""
        last_draw_messages_list_len = 0
        while True:
            time.sleep(1e-6)
            with model_4_lock:
                if len(self.draw_message_list) > 0 and self.draw_message_list[0] == -1:
                    break
                try:
                    if (len(self.draw_message_list) > 0 and self.draw_message_list[-1][0] != last_draw_message) or len(self.draw_message_list) > last_draw_messages_list_len:
                        self.to_monitor_rate.append(time.time())
                        last_draw_message = self.draw_message_list[-1][0]
                    last_draw_messages_list_len = len(self.draw_message_list)
                except Exception as e:
                    logger_model_4.warning(f"[Model_4_{self.id}] {e}, and draw_message_list[-1]: {self.draw_message_list[-1]}, and last_draw_message: {last_draw_message}")

                if len(self.to_monitor_rate) > 1:
                    rate = round((len(self.to_monitor_rate) - 1) / (self.to_monitor_rate[-1] - self.to_monitor_rate[0]), 3)
                    rates.append(rate)
                    if len(rates) > sliding_window_size:
                        rates.pop(0)
                    total_weight = sum(range(1, len(rates) + 1))
                    weighted_sum = sum((i + 1) * rate for i, rate in enumerate(rates))
                    moving_average = round(weighted_sum / total_weight, 3)
                    # print(f"[Model_4_{self.id}] rate: {moving_average}")
                    logger_model_4.info(f"[Model_4_{self.id}] rate: {moving_average}")
                    logger_model_4_rate.info(f"{moving_average}")
                    self.to_monitor_rate[:] = self.to_monitor_rate[-1:]

    def process_draw_message(self, draw_message):
        frame_filename, score, label, box = draw_message

        # Draw bounding boxes on the image
        try:
            image = Image.open(frame_filename)
            draw = ImageDraw.Draw(image)

            label_text = f"{label} {score}"
            if label == "car":
                draw.rectangle(box, outline="green", width=3)
                draw.text((box[0], box[1]), label_text, fill="red", font=font)
            if label == "person":
                draw.rectangle(box, outline="blue", width=3)
                draw.text((box[0], box[1]), label_text, fill="yellow", font=font)

            # Save the annotated image
            image.save(frame_filename)
            
        except Exception as e:
            logger_model_4.error(f"[Model_4_{self.id}] {e}")
        
        # Remove the image from the set of images being processed
        self.images_being_processed.remove(frame_filename)


class Model_5(multiprocessing.Process):
    def __init__(self, id, frame_files_to_be_processed, to_monitor_rate):
        super().__init__()
        self.id = id
        self.frame_files_to_be_processed = frame_files_to_be_processed

        self.timer_logger_model_5 = time.time()
        self.to_monitor_rate = to_monitor_rate
        self.threading_lock = threading.Lock()

    def run(self):
        if self.id == 1:
            thread_monitor_rate = threading.Thread(target=self.monitor_rate)
            thread_monitor_rate.start()

        print(f"[Model_5_{self.id}] start")
        logger_model_5.info(f"[Model_5_{self.id}] start")
        while True:
            time.sleep(0.01)
            if time.time() - self.timer_logger_model_5 > 5:
                logger_model_5.info(f"[Model_5_{self.id}] frame_files_to_be_processed: {self.frame_files_to_be_processed}")
                self.timer_logger_model_5 = time.time()
            if len(self.frame_files_to_be_processed) > 0:
                try:
                    if self.frame_files_to_be_processed[0] == -1:
                        print(f"[Model_5_{self.id}] end")
                        logger_model_5.info(f"[Model_5_{self.id}] end")
                        break
                    for video_id in self.frame_files_to_be_processed:
                        if video_id != -1 and self.frame_files_to_be_processed.count(video_id) > 1:
                            with self.threading_lock:
                                self.frame_files_to_be_processed.remove(video_id)
                                self.frame_files_to_be_processed.remove(video_id)
                            frame_filename = f"output_frames_video_{video_id}"
                            self.process_video(frame_filename)
                            continue
                    
                except Exception as e:
                    logger_model_5.error(f"[Model_5_{self.id}] {e}")

    def monitor_rate(self):
        rates = []
        sliding_window_size = 1
        last_frame_file = 0
        last_frame_files_len = 0
        while True:
            time.sleep(1e-6)
            with model_5_lock and self.threading_lock:
                if len(self.frame_files_to_be_processed) > 0 and self.frame_files_to_be_processed[0] == -1:
                    break
                try:
                    if (len(self.frame_files_to_be_processed) > 0 and self.frame_files_to_be_processed[-1] != last_frame_file) or len(self.frame_files_to_be_processed) > last_frame_files_len:
                        self.to_monitor_rate.append(time.time())
                        last_frame_file = self.frame_files_to_be_processed[-1]
                    last_frame_files_len = len(self.frame_files_to_be_processed)
                except Exception as e:
                    logger_model_5.warning(f"[Model_5_{self.id}] {e}, and frame_files_to_be_processed[-1]: {self.frame_files_to_be_processed[-1]}, and last_frame_file: {last_frame_file}")

                if len(self.to_monitor_rate) > 1:
                    rate = round((len(self.to_monitor_rate) - 1) / (self.to_monitor_rate[-1] - self.to_monitor_rate[0]), 3)
                    rates.append(rate)
                    if len(rates) > sliding_window_size:
                        rates.pop(0)
                    total_weight = sum(range(1, len(rates) + 1))
                    weighted_sum = sum((i + 1) * rate for i, rate in enumerate(rates))
                    moving_average = round(weighted_sum / total_weight, 3)
                    # print(f"[Model_5_{self.id}] rate: {moving_average}")
                    logger_model_5.info(f"[Model_5_{self.id}] rate: {moving_average}")
                    logger_model_5_rate.info(f"{moving_average}")
                    self.to_monitor_rate[:] = self.to_monitor_rate[-1:]

    def process_video(self, frame_filename):
        print(f"[Model_5_{self.id}] frame_filename: ", frame_filename)
        logger_model_5.info(f"[Model_5_{self.id}] frame_filename: {frame_filename}")
        video_id = frame_filename.split('_')[-1]
        # Input video file path
        input_video_path = f"../input_videos/video_{video_id}.mp4"
        print(f"[Model_5_{self.id}] input_video_path: ", input_video_path)
        logger_model_5.info(f"[Model_5_{self.id}] input_video_path: {input_video_path}")

        # Output directory for frames
        output_frames_dir = frame_filename
        output_video_dir = "../output_videos"
        os.makedirs(output_video_dir, exist_ok=True)

        # Output video file path
        output_video_path = f"../output_videos/processed_video_{video_id}.mp4"
        print(f"[Model_5_{self.id}] output_video_path: ", output_video_path)
        logger_model_5.info(f"[Model_5_{self.id}] output_video_path: {output_video_path}")
        
        # Open the video file
        cap = cv2.VideoCapture(input_video_path)

        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Combine the processed frames back into a video
        output_frames = [os.path.join(output_frames_dir, filename) for filename in os.listdir(output_frames_dir)]
        output_frames.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        for frame_path in output_frames:
            frame = cv2.imread(frame_path)
            output_video.write(frame)
        
        # Release the output video writer
        output_video.release()
        
        # Clean up: Delete the processed frames
        for frame_path in output_frames:
            os.remove(frame_path)

        # Clean up: Delete the frames directory
        os.rmdir(output_frames_dir)
        
        print(f"[Model_5_{self.id}] {input_video_path} processed successfully")
        logger_model_5.info(f"[Model_5_{self.id}] {input_video_path} processed successfully")

def pipeline():
    manager_1 = multiprocessing.Manager()
    manager_2 = multiprocessing.Manager()
    manager_3 = multiprocessing.Manager()
    manager_4 = multiprocessing.Manager()
    manager_5 = multiprocessing.Manager()
    manager_6 = multiprocessing.Manager()

    manager_to_monitor_rate_1 = multiprocessing.Manager()
    manager_to_monitor_rate_2 = multiprocessing.Manager()
    manager_to_monitor_rate_3 = multiprocessing.Manager()
    manager_to_monitor_rate_4 = multiprocessing.Manager()
    manager_to_monitor_rate_5 = multiprocessing.Manager()

    input_video_paths_list = manager_1.list()
    car_frames_list = manager_2.list()
    person_frames_list = manager_3.list()
    draw_message_list = manager_4.list()
    images_being_processed = manager_5.list()
    frame_files_to_be_processed = manager_6.list()

    to_monitor_rate_1 = manager_to_monitor_rate_1.list()
    to_monitor_rate_2 = manager_to_monitor_rate_2.list()
    to_monitor_rate_3 = manager_to_monitor_rate_3.list()
    to_monitor_rate_4 = manager_to_monitor_rate_4.list()
    to_monitor_rate_5 = manager_to_monitor_rate_5.list()

    end_signal = manager_1.Value('i', 0)

    loader = Loader(input_video_paths_list)
    model_1s = []
    model_2s = []
    model_3s = []
    model_4s = []
    model_5s = []

    for i in range(5):
        model_1 = Model_1(i + 1, input_video_paths_list, car_frames_list, person_frames_list, draw_message_list, end_signal, to_monitor_rate_1)
        model_2 = Model_2(i + 1, car_frames_list, draw_message_list, to_monitor_rate_2)
        model_3 = Model_3(i + 1, person_frames_list, draw_message_list, to_monitor_rate_3)
        model_4 = Model_4(i + 1, draw_message_list, images_being_processed, frame_files_to_be_processed, to_monitor_rate_4)
        model_5 = Model_5(i + 1, frame_files_to_be_processed, to_monitor_rate_5)
        model_1s.append(model_1)
        model_2s.append(model_2)
        model_3s.append(model_3)
        model_4s.append(model_4)
        model_5s.append(model_5)

    loader.start()
    for i in range(2):
        model_1s[i].start()    
    for i in range(3):
        model_2s[i].start()
    for i in range(3):
        model_3s[i].start()
    for i in range(1):
        model_4s[i].start()
    for i in range(1):
        model_5s[i].start()

    loader.join()
    for i in range(2):
        model_1s[i].join()
    for i in range(3):
        model_2s[i].join()
    for i in range(3):
        model_3s[i].join()
    for i in range(1):
        model_4s[i].join()
    for i in range(1):
        model_5s[i].join()

    # print("car_num: ", car_num)
    # print("person_num: ", person_num)

if __name__ == "__main__":
    """
    # opt_run = torch.compile(run)
    run()

    print("car_num: ", car_num)
    print("person_num: ", person_num)
    """
    pipeline()
