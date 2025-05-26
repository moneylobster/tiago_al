### Tiago ROS abstraction layer
# to write stuff in python without worrying about the ROS parts too much
import numpy as np
import spatialmath as sm

import rospy
import actionlib
import moveit_commander
import tf2_ros
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, JointState
from sensor_msgs.point_cloud2 import read_points
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Twist, Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.msg import RobotState
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal


# tiago-specific messages
from pal_statistics_msgs.msg import StatisticsValues

################################################################################
## TIAGO

class Tiago():
    def __init__(self):
        # start a ros node
        rospy.init_node("tiago_al", anonymous=True)
        # classes relating exclusively to individual components
        self.head=TiagoHead()
        self.arm=RetimingTiagoArm("time_optimal_trajectory_generation")
        self.gripper=TiagoGripper()

        # transforms
        self.tfbuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfbuffer)
        self.tf_pub=rospy.Publisher("/tf_static", TFMessage, queue_size=10)

        # motor stats
        self.stats_sub = rospy.Subscriber("/motors_statistics/values", StatisticsValues, self._stats_callback, queue_size=10)
        self.stats={}
        self.base_pub = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=10)

        #actuators
        self.torso_sub = rospy.Subscriber("/torso_controller/state", JointTrajectoryControllerState,
                                          self._torso_callback, queue_size=10)
        self.torso = None
        self.torso_pub = rospy.Publisher("/torso_controller/command", JointTrajectory, queue_size=10)
        self.base_pub = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=10)
        #play_motion
        self.play_motion_client=actionlib.SimpleActionClient('play_motion', PlayMotionAction)
        ## Moveit stuff
        self.planning_scene=moveit_commander.planning_scene_interface.PlanningSceneInterface()


    def get_transform(self, wrt, this):
        """Return an SE3 transform WRT_T_THIS from WRT to THIS."""
        return rostf_to_se3(self.tfbuffer.lookup_transform(wrt, this, rospy.Time.now(), rospy.Duration(1.0)))

    def publish_transform(self, pose, wrt, name, tries=10):
        """Publish an SE3 transform POSE which gives the relation WRT_T_NAME."""
        for _ in range(tries):
            tfmsg=TFMessage()
            goal_frame=se3_to_rostf(pose)
            goal_frame.header.stamp=rospy.Time.now()
            goal_frame.header.frame_id=wrt
            goal_frame.child_frame_id="goal_frame"
            tfmsg.transforms=[goal_frame]
            self.tf_pub.publish(tfmsg)
            rospy.sleep(0.05)

    def play_motion(self,motion_name):
        """Play the motion specified by MOTION_NAME"""
        self.play_motion_client.wait_for_server()
        msg=PlayMotionGoal()
        msg.motion_name=motion_name
        self.play_motion_client.send_goal(msg)
        self.play_motion_client.wait_for_result()
        return self.play_motion_client.get_result()

    def home(self):
        self.play_motion("home")

    def move_torso(self, goal):
        trajpt=JointTrajectoryPoint()
        trajpt.positions=[0.35]
        trajpt.time_from_start=rospy.Duration(1)
        torso_msg=JointTrajectory()
        torso_msg.joint_names=[
            "torso_lift_joint"
        ]
        torso_msg.points=[trajpt]
        self.torso_pub.publish(torso_msg)
        rospy.sleep(1)

    def _stats_callback(self, data):
        names=[
            "publish_async_attempts", "publish_async_failures", "publish_buffer_full_errors", "last_async_pub_duration",
            "arm1_mode", "arm1_current", "arm1_velocity", "arm1_position", "arm1_abs_position", "arm1_temperature",
            "arm2_mode", "arm2_current", "arm2_velocity", "arm2_position", "arm2_abs_position", "arm2_temperature",
            "arm3_mode", "arm3_current", "arm3_velocity", "arm3_position", "arm3_abs_position", "arm3_temperature",
            "arm4_mode", "arm4_current", "arm4_velocity", "arm4_position", "arm4_abs_position", "arm4_temperature",
            "arm5_mode", "arm5_current", "arm5_velocity", "arm5_position", "arm5_abs_position", "arm5_temperature",
            "arm6_mode", "arm6_current", "arm6_velocity", "arm6_position", "arm6_abs_position", "arm6_temperature",
            "arm7_mode", "arm7_current", "arm7_velocity", "arm7_position", "arm7_abs_position", "arm7_temperature",
            "gripper_left_current", "gripper_left_position", "gripper_left_abs_position", "gripper_left_temperature",
            "gripper_right_current", "gripper_right_position", "gripper_right_abs_position", "gripper_right_temperature",
            "head1_current", "head1_position", "head1_abs_position", "head1_temperature",
            "head2_current", "head2_position", "head2_abs_position", "head2_temperature",
            "torso_mode", "torso_current", "torso_velocity", "torso_position", "torso_abs_position", "torso_temperature",
            "wheel_left_mode", "wheel_left_current", "wheel_left_velocity", "wheel_left_position", "wheel_left_torque", "wheel_left_temperature",
            "wheel_right_mode", "wheel_right_current", "wheel_right_velocity", "wheel_right_position", "wheel_right_torque", "wheel_right_temperature"]
        vals=data.values
        self.stats=dict(zip(names,vals))

    def _torso_callback(self, data):
        self.torso = data.actual.positions[0]

################################################################################
## HEAD
        
class TiagoHead():
    def __init__(self):
        ## Cameras
        self.camera_frame="xtion_rgb_optical_frame" #there's also xtion_rgb_optical_frame
        # RGBD Camera
        self.rgb_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image,
                                        self._rgb_callback, queue_size=10)
        self.rgb = None
        self.depth_sub = rospy.Subscriber("/xtion/depth/image_raw", Image,
                                          self._depth_callback, queue_size=10)
        self.depth = None
        self.depth_data=None
        self.bridge = CvBridge()

        self.pointcloud_sub=rospy.Subscriber("/throttle_filtering_points/filtered_points", PointCloud2,
                                             self._pointcloud_callback, queue_size=10)
        self.pointcloud=None
        # Camera intrinsics
        self.cam_info_sub = rospy.Subscriber("/xtion/depth/camera_info", CameraInfo,
                                             self._cam_info_callback, queue_size=10)
        self.cam_raw_intrinsic = None
        ## Actuators
        self.motor_sub = rospy.Subscriber("/head_controller/state", JointTrajectoryControllerState,
                                          self._motor_callback, queue_size=10)
        self.motor = None
        self.motor_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=10)
        
    def _rgb_callback(self, data):
        self.rgb = self.bridge.imgmsg_to_cv2(data)
    def _depth_callback(self, data):
        self.depth = self.bridge.imgmsg_to_cv2(data)
    def _pointcloud_callback(self, data):
        self.pointcloud_frame=data.header.frame_id
        self.pointcloud=read_points(data)
    def _motor_callback(self,data):
        self.motor = data.actual.positions[0]
    def _cam_info_callback(self, data):
        self.cam_raw_intrinsic=np.array(data.K).reshape((3,3)) 

################################################################################
## ARM
        
class TiagoArm():
    def __init__(self):
        self.robot=moveit_commander.RobotCommander()
        self.move_group=moveit_commander.MoveGroupCommander("arm")
        # docs: https://docs.ros.org/en/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html
        self.endeff_frame="arm_tool_link"
        
    def current_pose(self):
        '''Returns the current pose of the arm as SE3.'''
        return rospose_to_se3(self.move_group.get_current_pose())
    def plan_cartesian_trajectory(self, trajectory, eef_step=0.001, jump_threshold=0.0):
        '''Plan the given trajectory.
        TRAJECTORY is a list of SE3 poses.'''
        ros_trajectory=[se3_to_rospose(se3) for se3 in trajectory]
        (plan, fraction) = self.move_group.compute_cartesian_path(ros_trajectory,
                                                                  eef_step,
                                                                  jump_threshold,
                                                                  avoid_collisions=True,
                                                                  path_constraints=None)
        plan=self.postprocess_plan(plan)
        print(f"Planning complete, successfully planned {fraction} of the path.")
        return plan, fraction
    def plan_to_pose(self, pose):
        '''Plan to the given pose.
        POSE is an SE3 pose.'''
        self.move_group.set_pose_target(se3_to_rospose(pose))
        success, plan, time, error=self.move_group.plan()
        print(f"Planning took {time} seconds.")
        if not success:
            print(f"Planning failed. Error code {error}")
        return (plan, success)
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

def merge_trajectories(traj1, traj2):
    "Merge RobotTrajectories traj1 and traj2 together. Modifies traj1."
    traj1.joint_trajectory.points = traj1.joint_trajectory.points+traj2.joint_trajectory.points
    return traj1

def robot_state_from_traj(traj):
    "Returns the RobotState that the robot will get to at the end of a RobotTrajectory traj."
    joints=traj.joint_trajectory.points[-1].positions
    joint_state=JointState()
    joint_state.header.stamp = rospy.Time.now()
    joint_state.name = ["arm_1_joint",
                        "arm_2_joint",
                        "arm_3_joint",
                        "arm_4_joint",
                        "arm_5_joint",
                        "arm_6_joint",
                        "arm_7_joint"]
    joint_state.position = joints
    moveit_robot_state = RobotState()
    moveit_robot_state.joint_state = joint_state
    return moveit_robot_state

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

def se3_to_rospose(se3):
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

def stamp_pose(pose, frame_id):
    posestamped=PoseStamped()
    posestamped.pose=pose
    posestamped.header.frame_id=frame_id
    return posestamped

def find_perp_vector(vector):
    """Find a perpendicular vector to a vector.

    Ref:
    Ken Whatmough (https://math.stackexchange.com/users/918128/ken-whatmough), How to find perpendicular vector to another vector?, URL (version: 2023-07-14): https://math.stackexchange.com/q/4112622
    """
    x,y,z=vector
    return np.array([np.copysign(z,x),
                     np.copysign(z,y),
                     -np.copysign(abs(x)+abs(y),z)])

