from aif_model.agent import Agent
import numpy as np
import torch
import aif_model.config as c
import aif_model.utils as utils
import time

from camera_orientation.turn_cam import euler_to_quaternion, quaternion_to_euler
from geometry_msgs.msg import Quaternion
import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import datetime

def project(position):
    f = c.width / (2 * np.tan(c.horizontal_fov/2))
    cent = (c.width/2, c.height/2) # get center point
    K = np.array([[f, 0, cent[0]],
                  [0, f, cent[1]],
                  [0, 0, 1]])
    
    R = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]])
    
    t = np.array([0,0,1])

    r_p = np.array([[0,0,1],
                    [-1,0,0],
                    [0,-1,0]])
    transformed = r_p.T.dot(R.T.dot(position - t))

    projected = K.dot(transformed)

    normalized = np.clip(projected/projected[2],-10,42)
    return normalized[0:2]

class AutoTrial(Node):

    def __init__(self):
        super().__init__('automatic_trial_execution')
        self.cam_orientation_publisher = self.create_publisher(Quaternion, '/cam_orientation_setter', 10)
        self.cam_orientation_subscriber = self.create_subscription(Quaternion, '/actual_cam_orientation', self.cam_orientation_callback, 1)
        self.image_subscriber = self.create_subscription(Image,'cam/camera1/image_raw', self.image_callback, 1)
        self.gazebo_client = self.create_client(SetEntityState, '/sim/set_entity_state')
        self.bridge = CvBridge()

        # Trial variables
        self.num_trials = 1
        self.init_t = 10 # steps
        self.cue_t = 100 # steps
        self.coa_t = 100 # steps
        self.step_max = 1000 # steps

        self.endogenous = True
        self.valid = True
        self.action_enabled = False
        
        # init agent
        self.agent = None
        self.step = 0 

        # init sensory input
        self.proprioceptive = np.zeros(c.prop_len)
        self.visual = np.zeros((1,c.channels,c.height,c.width))
        self.needs = np.zeros((c.needs_len))

        self.got_data = np.full((2),False)

        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d_%H-%M")
        self.log_name = f"act_inf_logs/experiments/log_{formatted_time}.csv"

    def reset(self):
        self.proprioceptive = np.zeros(c.prop_len)
        self.visual = np.zeros((1,c.channels,c.height,c.width))
        self.needs = np.zeros((c.needs_len))
        self.agent = None
        self.step = 0 
        self.move_ball((-1.0, 0.0, 1.0))
        self.reset_cam()

    def wait_data(self):
        print("<Auto Trials> Waiting for data...")
        while not self.got_data.all()==True:
            rclpy.spin_once(self)

        self.trials()

    def cam_orientation_callback(self, msg):
        r, p, y = quaternion_to_euler(msg)
        p = np.rad2deg(p)
        y = np.rad2deg(y)
        self.proprioceptive = np.array([p,y])
        self.got_data[0] = True

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            self.visual = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.visual = torch.tensor(np.transpose(self.visual,(2,0,1))).unsqueeze(0)
            
            # scale to [0,1]
            self.visual = self.visual/255
            self.got_data[1] = True
        except Exception as e:
            self.get_logger().error('Error processing image: {0}'.format(e))

    def publish_action(self, action):
        desired = self.proprioceptive + c.dt * action
        q = euler_to_quaternion(0,np.deg2rad(desired[0]),np.deg2rad(desired[1]))

        msg = q
        self.cam_orientation_publisher.publish(msg)

    def log(self, v):
        with open(self.log_name,"a") as l:
            l.write(','.join(map(str, v))+"\n")

    def generate_cues(self):
        endo_cue = np.ones(3)
        exo_cue = np.array([-1,0,1])
        ball_true = np.array([-1,0,1])

        exo_cue = np.array([4,np.random.random(1)[0]*4 - 2, np.random.random(1)[0]*4 - 1])
        projection = project(exo_cue)
        normalized  = utils.normalize(projection)
        endo_cue[0] = normalized[0]
        endo_cue[1] = normalized[1]

        if self.valid:
            ball_true = exo_cue
        else:
            ball_true = np.array([4,np.random.random(1)[0]*4 - 2, np.random.random(1)[0]*4 - 1])

        return endo_cue, exo_cue, ball_true

    def move_ball(self, position):
        state = EntityState()
        state.name = "red_sphere"

        # position
        state.pose.position.x = position[0]
        state.pose.position.y = position[1]
        state.pose.position.z = position[2]

        req = SetEntityState.Request()
        req._state = state
        future = self.gazebo_client.call_async(req)
        rclpy.spin_until_future_complete(self, future) 

    def reset_cam(self):
        state = EntityState()
        state.name = "camera_model"

        # position
        state.pose.position.x = 0.0
        state.pose.position.y = 0.0
        state.pose.position.z = 1.0

        # orientation
        state.pose.orientation.x = 0.0
        state.pose.orientation.y = 0.0
        state.pose.orientation.z = 0.0
        state.pose.orientation.w = 1.0

        req = SetEntityState.Request()
        req._state = state
        future = self.gazebo_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)  # Wait for the result

    def trials(self):
        print("<Auto Trials> Starting trials")
        for i in range(self.num_trials):
            
            # initialize logging variables
            reaction_time = -1
            reach_time = -1
            cue_center_dist = -1
            target_center_dist = -1
            cue_target_dist = -1
            
            # init agent
            self.agent = Agent()
            self.agent.init_belief(self.needs,self.proprioceptive,self.visual)

            # one trial
            print("<Auto Trials> Starting trial " + str(i) + "/" +str(self.num_trials))
            # INIT
            for _ in range(self.init_t):
                self.update()
                if self.ball_perceived() and reaction_time<0:
                    reaction_time = self.step - self.init_t - self.cue_t - self.coa_t
                if self.ball_reached() and reach_time<0:
                    reach_time = self.step - self.init_t - self.cue_t - self.coa_t

            # GENERATE CUES
            endo_cue, exo_cue, ball_true = self.generate_cues()

            cue_pos = project(exo_cue)
            ball_pos = project(ball_true)
            cue_center_dist = np.linalg.norm(cue_pos - np.array([16,16]))
            target_center_dist = np.linalg.norm(ball_pos - np.array([16,16]))
            cue_target_dist = np.linalg.norm(cue_pos - ball_pos)

            # SET CUES
            if self.endogenous: # set self.needs
                self.needs = endo_cue
            else: # move ball
                self.move_ball(exo_cue)
            
            # CUE
            print("<Auto Trials> Cueing...")
            for _ in range(self.cue_t):
                self.update()
                if self.ball_perceived() and reaction_time<0:
                    reaction_time = self.step - self.init_t - self.cue_t - self.coa_t
                if self.ball_reached() and reach_time<0:
                    reach_time = self.step - self.init_t - self.cue_t - self.coa_t

            # REMOVE CUE
            if self.endogenous: # set self.needs
                self.needs = np.zeros((c.needs_len))
            else: # move ball
                self.move_ball((-1, 0, 1))

            # COA: Cue Onset Asynchrony
            print("<Auto Trials> COA...")
            for _ in range(self.coa_t):
                self.update()
                if self.ball_perceived() and reaction_time<0:
                    reaction_time = self.step - self.init_t - self.cue_t - self.coa_t
                if self.ball_reached() and reach_time<0:
                    reach_time = self.step - self.init_t - self.cue_t - self.coa_t

            # SET TARGET
            print("<Auto Trials> Setting Target...")
            self.move_ball(ball_true)

            # TARGET
            print("<Auto Trials> Perception...")
            while (self.step - self.init_t - self.cue_t - self.coa_t) <= self.step_max:
                self.update()
                if self.ball_perceived() and reaction_time<0:
                    reaction_time = self.step - self.init_t - self.cue_t - self.coa_t
                    if not self.action_enabled:
                        break
                if self.ball_reached() and reach_time<0:
                    reach_time = self.step - self.init_t - self.cue_t - self.coa_t

            # log data
            v = [reaction_time,reach_time,cue_center_dist,target_center_dist,cue_target_dist]
            self.log(v)

            # reset
            print("<Auto Trials> Resetting", end="")
            self.reset()
            for _ in range(10):
                print(".",end="")
                time.sleep(0.5)
                rclpy.spin_once(self) 
            print("")

        print("<Auto Trials> Finished")

    def ball_perceived(self):
        return self.agent.mu[0,c.needs_len+c.prop_len+c.prop_len]>0.1
    
    def ball_reached(self):
        ball_coords = self.agent.mu[0,c.needs_len+c.prop_len:c.needs_len+c.prop_len+c.prop_len]
        return np.linalg.norm(ball_coords) < (1/16) and self.ball_perceived()
    
    def update(self):
        rclpy.spin_once(self)
        # get sensory input
        S =  self.needs, self.proprioceptive, self.visual

        action, _, _ = self.agent.inference_step(S)
        action = utils.add_gaussian_noise(action)

        if self.action_enabled:
            self.publish_action(action)

        self.step+=1

def main(args=None):
    rclpy.init(args=args)
    auto = AutoTrial()
    auto.wait_data()
    #rclpy.spin(auto)
    auto.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()