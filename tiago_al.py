### Tiago ROS abstraction layer
# to write stuff in python without worrying about the ROS parts too much
import numpy as np
import spatialmath as sm

import rospy
import moveit_commander
import tf2_ros
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Twist, Pose
from trajectory_msgs.msg import JointTrajectory
from control_msgs.msg import JointTrajectoryControllerState

class Tiago():
    def __init__(self):
        self.head=TiagoHead()
        self.arm=RetimingTiagoArm("time_optimal_trajectory_generation")
        self.gripper=TiagoGripper()
        
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer)
        self.tf_pub=rospy.Publisher("/tf", TFMessage, queue_size=10)
        
        self.base_pub = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=10)

class TiagoHead():
    def __init__(self):
        ## Cameras
        # RGBD Camera
        self.rgb_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image, self.rgb_callback, queue_size=10)
        self.rgb = None
        self.depth_sub = rospy.Subscriber("/xtion/depth/image_raw", Image, self.depth_callback, queue_size=10)
        self.depth = None
        self.depth_data=None
        self.bridge = CvBridge()
        # Camera intrinsics
        self.cam_info_sub = rospy.Subscriber("/xtion/depth/camera_info", CameraInfo, self.cam_info_callback, queue_size=10)
        self.cam_raw_intrinsic = None
        ## Transforms
        self.state_sub = rospy.Subscriber("/head_controller/state", JointTrajectoryControllerState, self.state_callback, queue_size=10)
        self.state = None
        self.state_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=10)
        
    def rgb_callback(self, data):
        self.rgb = self.bridge.imgmsg_to_cv2(data)
    def depth_callback(self, data):
        self.depth_data=data
        self.depth = self.bridge.imgmsg_to_cv2(data)
    def state_callback(self,data):
        self.state = data.actual.positions[0]
    def cam_info_callback(self, data):
        self.cam_raw_intrinsic=np.array(data.K).reshape((3,3)) 

class TiagoArm():
    def __init__(self):
        self.robot=moveit_commander.RobotCommander()
        self.move_group=moveit_commander.MoveGroupCommander("arm")
    def current_pose(self):
        '''Returns the current pose of the arm as SE3.'''
        return rospose_to_se3(self.move_group.get_current_pose())
    def plan_trajectory(self, ros_trajectory):
        '''Plan the given trajectory.
        ROS_TRAJECTORY is a list of Pose geometry messages.'''
        (plan, fraction) = self.move_group.compute_cartesian_path(ros_trajectory, 0.001, 0.0,
                                                                  avoid_collisions=True,
                                                                  path_constraints=None)
        plan=self.postprocess_plan(plan)
        print(f"Planning complete, successfully planned {fraction} of the path.")
        return plan, fraction
    def postprocess_plan(self, plan):
        '''Postprocess plan.'''
        # This class does no post-processing.
        return plan
    def execute_plan(self, plan):
        '''Execute plan.'''
        self.move_group.execute(plan, wait=True)

class RetimingTiagoArm(TiagoArm):
    def __init__(self, retiming_algorithm):
        assert retiming_algorithm in ["iterative_time_parametrization",
                                      "iterative_spline_parametrization",
                                      "time_optimal_trajectory_generation"]
        self.retiming_algorithm=retiming_algorithm
        super().__init__()
    def postprocess_plan(self, plan):
        '''Postprocess plan.'''
        ref_state=self.robot.get_current_state()
        retimed_plan=self.move_group.retime_trajectory(ref_state,
                                                       plan,
                                                       velocity_scaling_factor=1.0,
                                                       acceleration_scaling_factor=1.0,
                                                       algorithm=self.retiming_algorithm)
        return retimed_plan


class TiagoGripper():
    def __init__(self):
        pass
    def grasp(self):
        '''
        call a grasp.
        '''
        rospy.wait_for_service('/parallel_gripper_controller/grasp')
        try:
            grasp_service = rospy.ServiceProxy('/parallel_gripper_controller/grasp', Empty)
            response = grasp_service()
            rospy.loginfo("Grasp service call successful")
            return True
        except rospy.ServiceException as e:
            rospy.loginfo(f"Grasp service call failed: {e}")
            return False
    def release(self):
        '''
        call a release.
        '''
        rospy.wait_for_service('/parallel_gripper_controller/release')
        try:
            release_service = rospy.ServiceProxy('/parallel_gripper_controller/release', Empty)
            response = release_service()
            rospy.loginfo("Release service call successful")
            return True
        except rospy.ServiceException as e:
            rospy.loginfo(f"Release service call failed: {e}")
            return False
################################################################################
## convenience funcs
def rostf_to_se3(rostf):
    "convert a ros tf object into sm.SE3"
    trans=[rostf.transform.translation.x,
           rostf.transform.translation.y,
           rostf.transform.translation.z]
    quat=[rostf.transform.rotation.x,
          rostf.transform.rotation.y,
          rostf.transform.rotation.z,
          rostf.transform.rotation.w]
    
    tf=sm.UnitQuaternion(quat[-1],quat[:3]).SE3()
    tf.t=trans
    return tf

def se3_to_rostf(se3):   
    "convert an sm.SE3 object into a ros tf"
    tf=TransformStamped()
    tf.transform.translation.x=se3.t[0]
    tf.transform.translation.y=se3.t[1]
    tf.transform.translation.z=se3.t[2]

    quat=sm.UnitQuaternion(se3).vec_xyzs
    tf.transform.rotation.x=quat[0]
    tf.transform.rotation.y=quat[1]
    tf.transform.rotation.z=quat[2]
    tf.transform.rotation.w=quat[3]

    return tf    

def rospose_to_se3(rospose):
    "convert a ros pose into sm.SE3"
    trans=[rospose.pose.position.x,
           rospose.pose.position.y,
           rospose.pose.position.z]
    quat=[rospose.pose.orientation.x,
          rospose.pose.orientation.y,
          rospose.pose.orientation.z,
          rospose.pose.orientation.w]
    
    tf=sm.UnitQuaternion(quat[-1],quat[:3]).SE3()
    tf.t=trans
    return tf

def se3_to_pose(se3):
    "convert an sm.SE3 into a ros pose"
    pose = Pose()
    pose.position.x=se3.t[0]
    pose.position.y=se3.t[1]
    pose.position.z=se3.t[2]

    quat=sm.UnitQuaternion(se3).vec_xyzs
    pose.orientation.x=quat[0]
    pose.orientation.y=quat[1]
    pose.orientation.z=quat[2]
    pose.orientation.w=quat[3]
    
    return pose
