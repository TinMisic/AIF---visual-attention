from aif_model.agent import Agent
import numpy as np
import torch
import aif_model.config as c
import aif_model.utils as utils

from camera_orientation.turn_cam import euler_to_quaternion, quaternion_to_euler
from geometry_msgs.msg import Quaternion
from std_msgs.msg import Float32MultiArray
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
import datetime

class Inference(Node):

    def __init__(self):
        super().__init__('active_inference')
        self.cam_orientation_publisher = self.create_publisher(Quaternion, '/cam_orientation_setter', 10)
        self.cam_orientation_subscriber = self.create_subscription(Quaternion, '/actual_cam_orientation', self.cam_orientation_callback, 1)
        self.image_subscriber = self.create_subscription(Image,'cam/camera1/image_raw', self.image_callback, 1)
        self.needs_subscriber = self.create_subscription(Float32MultiArray, '/needs', self.needs_callback, 1)
        self.projections_subscriber = self.create_subscription(Float32MultiArray, '/object_projections', self.projections_callback, 1) # only for the purpose of logging
        self.bridge = CvBridge()
        
        # init agent
        self.agent = Agent()

        # init sensory input
        self.proprioceptive = np.zeros(c.prop_len)
        self.visual = np.zeros((1,c.channels,c.height,c.width))
        self.needs = np.ones((c.needs_len))

        self.got_data = np.full((3),False)
        self.step = 1
        self.flag = False
        self.counter = 1
        self.steps = 1

        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M")
        self.log_name = f"act_inf_logs/log_{formatted_time}.csv"
        self.projections = np.zeros(4)
        self.fe_log = []
        self.err_log = []

    def wait_data(self):
        print("Waiting on data...")
        while not self.got_data.all()==True:
            rclpy.spin_once(self)
        self.agent.init_belief(self.needs,self.proprioceptive,self.visual)
        update_period = 1/c.update_frequency
        self.create_timer(update_period, self.update)

    def projections_callback(self, msg):
        self.projections = np.array(msg.data)

    def needs_callback(self, msg):
        self.needs = np.array(msg.data)
        self.got_data[0] = True

    def cam_orientation_callback(self, msg):
        r, p, y = quaternion_to_euler(msg)
        p = np.rad2deg(p)
        y = np.rad2deg(y)
        self.proprioceptive = np.array([p,y])
        self.got_data[1] = True

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.visual = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.visual = torch.tensor(np.transpose(self.visual,(2,0,1))).unsqueeze(0)

            # scale to [0,1]
            self.visual = self.visual/255
            self.got_data[2] = True
        except Exception as e:
            self.get_logger().error('Error processing image: {0}'.format(e))

    def publish_action(self, action):
        desired = self.proprioceptive + c.dt * action
        q = euler_to_quaternion(0,np.deg2rad(desired[0]),np.deg2rad(desired[1]))

        msg = q
        self.cam_orientation_publisher.publish(msg)

    def log(self):
        needs = self.needs
        targets = self.agent.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.prop_len*c.num_intentions] # grab visual positions of objects
        targets = np.reshape(targets,(c.num_intentions,c.prop_len)) # reshape
        targets = utils.denormalize(targets) # convert from range [-1,1] to [0,width]
        targets = targets.flatten()
        projections = self.projections
        concat = np.concatenate((needs,targets,projections))
        with open(self.log_name,"a") as l:
            l.write(','.join(map(str, concat))+"\n")
    
    def update(self):
        # get sensory input
        S =  self.needs, self.proprioceptive, self.visual

        action, fe, err = self.agent.inference_step(S,self.step)
        action = utils.add_gaussian_noise(action)
        print("Action:",action)

        self.fe_log.append(fe)
        self.err_log.append(err)

        self.publish_action(action)

        if self.flag==False:
            self.counter = 0
            inp = input("step "+str(self.step)+" continue>")
            if inp=="i":
                # save images
                # plt.imshow(self.agent.tmp_S)
                # plt.show()
                # plt.imshow(self.agent.tmp_P)
                # plt.show()
                fe = np.array(self.fe_log)
                err = np.array(self.err_log)
                plt.title("Visual log_likelihood (- Free-Energy)")
                plt.plot(fe/np.max(fe),label = "log likelihood") # /np.max(fe)
                plt.plot(err/np.max(err),label = "error")
                plt.legend()
                plt.show()
                # cv2.imwrite("sensory.png",self.agent.tmp_S)
                # cv2.imwrite("prediction.png",self.agent.tmp_P)
            elif inp=="c":
                self.flag = True
            elif inp=="s":
                self.steps = int(input("Number of steps(int):"))
                self.flag = True
        else:
            if self.counter%self.steps == 0:
                self.flag = False

        self.log()
        self.step+=1
        self.counter+=1

def main(args=None):
    rclpy.init(args=args)
    inf = Inference()
    inf.wait_data()
    rclpy.spin(inf)
    inf.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()