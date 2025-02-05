import os
import cv2
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np


def save_compressed_image_msgs_as_png(bag_file, output_folder, ds_size):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    bridge = CvBridge()
    centr1 = open(os.path.join(output_folder, "centroids1.csv"), "a")
    centr5 = open(os.path.join(output_folder, "centroids5.csv"), "a")
    
    centroid = np.zeros(8)

    # Create reader instance and open for reading
    with Reader(bag_file) as reader:
        i = 0
        gray_counter = 0
        gc = 0

        # Iterate over messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/cam/camera1/image_raw':
                print(i * 100.0 / ds_size, "%")
  
                msg = deserialize_cdr(rawdata, connection.msgtype)
                # Convert sensor_msgs/Image to OpenCV image
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                
                if centroid.size == 0:
                    continue
                
                centroid = np.array(centroid)
                if np.bitwise_or(centroid < -5, centroid > 32).all():
                    gray_counter += 1
                    if gray_counter >= 0.05 * ds_size:
                        gc += 1
                        continue
                
                num = i + 3*240000
                number = (7 - len(str(num))) * "0" + str(num)
                filename = f"img_{number}.jpg"

                # Save the image as JPG
                cv2.imwrite(os.path.join(output_folder, filename), cv_image)
                
                c1 = np.insert(centroid, 4, -1)
                c1 = np.insert(c1, 2, -1)
                
                c5 = np.insert(centroid, 4, -0.5)
                c5 = np.insert(c5, 2, -0.5)
                
                centr1.write(",".join(map(str, c1)) + "\n")
                centr5.write(",".join(map(str, c5)) + "\n")
                
                i += 1
                if i >= ds_size:
                    break
                
            if connection.topic == '/object_projections':
                msg = deserialize_cdr(rawdata, connection.msgtype)
                centroid = np.array(list(msg.data))
    
        print(gray_counter - gc)
    centr1.close()
    centr5.close()

if __name__ == '__main__':
    # Specify the path to the ROS 2 bag file
    bag_file_path = 'datasets/two_-r-b/'

    # Specify the output folder where images will be saved
    output_folder_path = "datasets/new_two/"

    # Call the function to save image messages as PNGs
    save_compressed_image_msgs_as_png(bag_file_path, output_folder_path, 240000)
