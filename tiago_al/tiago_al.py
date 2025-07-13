'''Tiago ROS abstraction layer
to write stuff in python without worrying about the ROS parts too much'''

import tiago_al.velocity_controllers as vc

import subprocess

import numpy as np
import spatialmath as sm

# ros stuff
import rospy
import actionlib
import moveit_commander
import tf2_ros
from cv_bridge import CvBridge
# ros messages
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, JointState, LaserScan
from sensor_msgs.point_cloud2 import read_points
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped, Twist, Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
from moveit_msgs.msg import RobotState
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from play_motion_msgs.msg import PlayMotionAction, PlayMotionGoal
# tiago-specific messages
from pal_statistics_msgs.msg import StatisticsValues
from pal_common_msgs.msg import DisableAction, DisableGoal
# services
from std_srvs.srv import Empty
from controller_manager_msgs.srv import SwitchController, ListControllers

################################################################################
## TIAGO

class Tiago():
    "A class to act as an interface to Tiago's functionality"
    def __init__(self,
                 transforms=True,
                 stats=True,
                 laser=True,
                 torso=True):
        # start a ros node
        rospy.init_node("tiago_al", anonymous=True)
        ## Submodules
        # classes relating exclusively to individual components
        self.head=TiagoHead()
        "Subclass for the head of the robot. Includes camera data."
        self.arm=TiagoArm("time_optimal_trajectory_generation")
        "Subclass for the arm of the robot. Includes motion planning stuff."
        self.gripper=TiagoGripper()
        "Subclass for things relating to gripper control."
        # transforms
        if transforms is not None:
            self.tfbuffer = tf2_ros.Buffer()
            self.tflistener = tf2_ros.TransformListener(self.tfbuffer)
            self.tf_pub=rospy.Publisher("/tf_static", TFMessage, queue_size=10)
        # motor stats
        if stats is not None:
            self.stats_sub = rospy.Subscriber("/motors_statistics/values", StatisticsValues, self._stats_callback, queue_size=10)
            self.stats={}
            "A dictionary that holds temperature, position, current, etc. information for all motors in the robot."  
        #sensors
        if laser is not None:
            self._laser_sub=rospy.Subscriber("/scan", LaserScan,
                                             self._laser_callback, queue_size=10)
            self.laser = None
            "Results of planar laser scan as [angle, distance]. Use laser_cartesian to access the points in a cartesian frame."
            self.laser_frame = "base_laser_link"
            "Reference frame for the laser scanner."
        #actuators
        if torso is not None:
            self.torso_sub = rospy.Subscriber("/torso_controller/state", JointTrajectoryControllerState,
                                              self._torso_callback, queue_size=10)
            self.torso = None
            "Torso position."
        self.torso_pub = rospy.Publisher("/torso_controller/command", JointTrajectory, queue_size=10)
        self.base_pub = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=10)
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        
        # To play prerecorded motions
        self.play_motion_client=actionlib.SimpleActionClient('play_motion', PlayMotionAction)
        # Moveit stuff
        self.planning_scene=moveit_commander.planning_scene_interface.PlanningSceneInterface()

    def get_transform(self, wrt: str, this: str) -> sm.SE3:
        """Return an SE3 transform WRT_T_THIS from WRT to THIS.

        wrt: which frame is the transform with respect to?
        this: which frame do we want the transform of?"""
        return rostf_to_se3(self.tfbuffer.lookup_transform(wrt, this, rospy.Time.now(), rospy.Duration(1.0)))

    def publish_transform(self, pose: sm.SE3, wrt: str, name: str, tries=10):
        """Publish an SE3 transform POSE which gives the relation WRT_T_NAME.

        pose: the transform as an sm.SE3 pose.
        wrt: which frame is it with respect to?
        name: name of the new frame"""
        for _ in range(tries):
            tfmsg=TFMessage()
            goal_frame=se3_to_rostf(pose)
            goal_frame.header.stamp=rospy.Time.now()
            goal_frame.header.frame_id=wrt
            goal_frame.child_frame_id=name
            tfmsg.transforms=[goal_frame]
            self.tf_pub.publish(tfmsg)
            rospy.sleep(0.05)

    def play_motion(self, motion_name: str):
        """Play the motion specified by MOTION_NAME"""
        if self.play_motion_client.wait_for_server(rospy.Duration(secs=1)):
            msg=PlayMotionGoal()
            msg.motion_name=motion_name
            self.play_motion_client.send_goal(msg)
            if self.play_motion_client.wait_for_result(rospy.Duration(secs=30)):
                return self.play_motion_client.get_result()
            else:
                rospy.logerr("[PLAY_MOTION] Sent goal, didn't get reply before timeout.")
                return False
        else:
            rospy.logerr("[PLAY_MOTION] Couldn't access play_motion server.")
            return False

    def home(self):
        "Move the robot to home configuration."
        self.play_motion("home")

    def say(self, words: str, language="en"):
        """Speak using espeak-ng.
        
        words: What to say
        language: either en or tr"""
        if language=="en":
            cmd=["espeak-ng", f'"{words}"']
        elif language=="tr":
            cmd=["espeak-ng", "-v", "tr", "-s", "150" ,f'"{words}"']
        else:
            raise NotImplementedError(f"Unsupported language {language}")
        subprocess.run(cmd)

    def move_torso(self, goal: float):
        "Move torso to specified point."
        trajpt=JointTrajectoryPoint()
        trajpt.positions=[goal]
        trajpt.time_from_start=rospy.Duration(1)
        torso_msg=JointTrajectory()
        torso_msg.joint_names=[
            "torso_lift_joint"
        ]
        torso_msg.points=[trajpt]
        self.torso_pub.publish(torso_msg)
        rospy.sleep(1)

    def move(self, x=0, y=0, z=0, rx=0, ry=0, rz=0):
        """Move Tiago's base with the specified velocity.

        +x is forward.
        +rz is to the left (CCW)
        """
        base_msg=Twist()
        base_msg.linear.x=x
        base_msg.linear.y=y
        base_msg.linear.z=z
        base_msg.angular.x=rx
        base_msg.angular.y=ry
        base_msg.angular.z=rz
        self.base_pub.publish(base_msg)

    def move_to(self, posestamped: PoseStamped):
        """Command Tiago to move to a certain pose using its nav stack.
        Check if done with is_move_done
        """
        if self.move_base_client.wait_for_server(rospy.Duration(secs=1)):
            goal=MoveBaseGoal()
            goal.target_pose.header.frame_id=posestamped.header.frame_id
            goal.target_pose.header.stamp=rospy.Time.now()
            # could probably do this better but w/e
            goal.target_pose.pose.position.x=posestamped.pose.position.x
            goal.target_pose.pose.position.y=posestamped.pose.position.y
            goal.target_pose.pose.position.z=posestamped.pose.position.z
            goal.target_pose.pose.orientation.x=posestamped.pose.orientation.x
            goal.target_pose.pose.orientation.y=posestamped.pose.orientation.y
            goal.target_pose.pose.orientation.z=posestamped.pose.orientation.z
            goal.target_pose.pose.orientation.w=posestamped.pose.orientation.w
            self.move_base_client.send_goal(goal)
            return True
        else:
            rospy.logerr("[MOVE_TO] Couldn't access move_base server.")
            return False

    def is_move_done(self):
        """Returns move_base server state, that is, whether the pose commanded by move_to has been reached."""
        if self.move_base_client.wait_for_result():
            return self.move_base_client.get_result()
        else:
            rospy.logerr("[IS_MOVE_DONE] Move_base server can't be reached!")

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
    def _laser_callback(self, data):
        angles=np.arange(data.angle_min, data.angle_max, data.angle_increment)
        self.laser=[angles, data.ranges[:len(angles)]]
    @property
    def laser_cartesian(self):
        "Results of the planar laser scan as an array of 2D points according to base frame (+x forward, +y left)."
        cossin=np.array([np.cos(self.laser[0]), np.sin(self.laser[0])])
        return (cossin*self.laser[1][len(self.laser[0,:])]).T
    def quit(self):
        "Safely stop/shutdown Tiago. (TODO) Also stops movements if velocity controller is active."
        self.board_detector.running=False
        if self.arm.controller=="arm_forward_velocity_controller":
            self.arm.stop=True
            self.arm.velocity_stop()
        
################################################################################
## HEAD
        
class TiagoHead():
    "Functions relating to Tiago's head. Cameras, motors etc."
    def __init__(self,
                 rgb=True,
                 depth=True,
                 pointcloud=True,
                 caminfo=True,
                 motor=True):
        ## Cameras
        self.camera_frame="xtion_rgb_optical_frame"
        "The frame used for the camera"
        
        # RGBD Camera
        self._bridge = CvBridge()
        if rgb is not None:
            self._rgb_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image,
                                             self._rgb_callback, queue_size=10)
        if depth is not None:
            self._depth_sub = rospy.Subscriber("/xtion/depth/image_raw", Image,
                                               self._depth_callback, queue_size=10)
        if pointcloud is not None:
            self._pointcloud_sub=rospy.Subscriber("/throttle_filtering_points/filtered_points", PointCloud2,
                                                  self._pointcloud_callback, queue_size=10)
        self.rgb = None
        "Latest fetched RGB image. Size should be 640x480."
        self.depth = None
        "Latest fetched depth image. Size should be 640x480."
        self.pointcloud=None
        "Pointcloud obtained from the depth image."
        self.pointcloud_frame=None
        "The name of the frame that the pointcloud is relative to."
        
        # Camera intrinsics
        if caminfo is not None:
            self._cam_info_sub = rospy.Subscriber("/xtion/depth/camera_info", CameraInfo,
                                                  self._cam_info_callback, queue_size=10)
        self.cam_raw_intrinsic = None
        "Camera intrinsic matrix."
        
        ## Actuators
        if motor is not None:
            self._motor_sub = rospy.Subscriber("/head_controller/state", JointTrajectoryControllerState,
                                               self._motor_callback, queue_size=10)
        self.motor = None
        "Motor positions, 2 element list. The first one is yaw, the second is pitch."
        self._motor_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=10)

        # disable head manager
        head_disable_client=actionlib.SimpleActionClient("/pal_head_manager/disable", DisableAction)
        if head_disable_client.wait_for_server(rospy.Duration(secs=1)):
            disable_goal=DisableGoal()
            head_disable_client.send_goal(disable_goal)
        else:
            rospy.logerr("[TIAGO_HEAD] Couldn't access pal_head_manager, not disabling it.")
        
        
    def _rgb_callback(self, data):
        self.rgb = self._bridge.imgmsg_to_cv2(data)
    def _depth_callback(self, data):
        self.depth = self._bridge.imgmsg_to_cv2(data)
    def _pointcloud_callback(self, data):
        self.pointcloud_frame=data.header.frame_id
        self.pointcloud=read_points(data)
    def _motor_callback(self,data):
        self.motor = data.actual.positions
    def _cam_info_callback(self, data):
        self.cam_raw_intrinsic=np.array(data.K).reshape((3,3)) 
################################################################################
## ARM
class TiagoArm():
    "Functions relating to Tiago's arm: motion planning, etc."
    def __init__(self, retiming_algorithm: str, velocity_controller=None):
        self.robot=moveit_commander.RobotCommander()
        self.move_group=moveit_commander.MoveGroupCommander("arm")
        "docs: https://docs.ros.org/en/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html"
        self.endeff_frame="arm_tool_link"
        "The end-effector frame used by moveit."
        self.planning_frame=self.move_group.get_planning_frame()
        "The planning frame used by moveit."
        self.move_group.set_num_planning_attempts(200)
        self.move_group.set_planning_time(4)

        assert retiming_algorithm in ["iterative_time_parametrization",
                                      "iterative_spline_parametrization",
                                      "time_optimal_trajectory_generation",
                                      None], f"Retiming algorithm {retiming_algorithm} is not supported."
        self.retiming_algorithm=retiming_algorithm

        # TODO: it's possible to do this a bit better by checking the
        # claimed_resources part of the listcontrollers response and
        # seeing if they contain all arm joints instead of hardcoding
        # possible controller names.
        self.all_arm_joint_controllers=[
            "arm_velocity_trajectory_controller",
            "arm_impedance_controller",
            "arm_forward_velocity_controller",
            "arm_controller"]
        "Controllers that can be used with the arm. Call switch_controller to use them."

        self.stop=False
        "Flag to determine whether arm should stop moving. Currently only works for the velocity controller."
        self._velocity_pub=rospy.Publisher("/arm_forward_velocity_controller/command", Float64MultiArray, queue_size=1)
        
        if velocity_controller is None:
            self.velocity_controller=vc.PIDController(Kp=2.0, Ki=5e-2)
        else:
            self.velocity_controller=velocity_controller
    @property
    def q(self):
        "Return the current joint configuration, a 7-element array."
        return self.move_group.get_current_joint_values()
    @property
    def current_pose(self) -> sm.SE3:
        "The current pose of the arm end effector (arm_tool_link) as SE3 wrt the planning frame (base_footprint)."
        return rospose_to_se3(self.move_group.get_current_pose())
    @property
    def jacobo(self):
        """Returns the arm Jacobian wrt. the planning frame (base_footprint).
        Matrix is 6x7. Each row shows how the joints will affect spatial velocity [vx vy vz wx wy wz]."""
        return np.array(self.move_group.get_jacobian_matrix(self.q))
    @property
    def jacobe(self):
        """Returns the arm Jacobian wrt. the end-eff frame (arm_tool_link).
        Matrix is 6x7. Each row shows how the joints will affect spatial velocity [vx vy vz wx wy wz]."""
        return sm.base.tr2jac(self.current_pose.inv().A) @ self.jacobo
    @property
    def controller(self):
        """Returns the currently running arm controller."""
        controllers=get_controllers()
        # There should only be one arm joint controller running. I
        # think ROS prevents multiple from running so we don't need to
        # check.
        for ctrl in controllers.controller:
            if ctrl.name in self.all_arm_joint_controllers and ctrl.state=="running":
                return ctrl.name
        return None
    def plan_cartesian_trajectory(self, trajectory, eef_step=0.001, jump_threshold=0.0, start_state=None):
        '''Plan the given trajectory.
        trajectory: a list of SE3 poses denoting the waypoints.
        eef_step: view move_group documentation (likely the interpolation increment)
        jump_threshold: not sure, view move_group documentation
        start_state: optional starting RobotState, default is the current state.'''
        if start_state is None:
            start_state=self.move_group.get_current_state()
        self.move_group.set_start_state(start_state)
        ros_trajectory=[se3_to_rospose(se3) for se3 in trajectory]
        (plan, fraction) = self.move_group.compute_cartesian_path(ros_trajectory,
                                                                  eef_step,
                                                                  jump_threshold,
                                                                  avoid_collisions=True,
                                                                  path_constraints=None)
        plan=self.postprocess_plan(plan)
        print(f"[PLAN_CARTESIAN_TRAJECTORY] Planning complete, successfully planned {fraction} of the path.")
        return plan, fraction
    def plan_to_pose(self, pose: sm.SE3):
        '''Plan to the given pose.
        POSE is an SE3 pose.'''
        self.move_group.set_pose_target(se3_to_rospose(pose))
        success, plan, time, error=self.move_group.plan()
        print(f"[PLAN_TO_POSE] Planning took {time} seconds.")
        if not success:
            print(f"[PLAN_TO_POSE] Planning failed. Error code {error}")
        return (plan, success)
    def plan_to_poses(self, poses, start_state=None):
        '''Plan to a sequential trajectory of POSES.
        poses: a list of SE3 poses.
        start_state: optional starting RobotState, default is the current state.'''
        if start_state is None:
            start_state=self.move_group.get_current_state()
        plans=None
        self.move_group.set_start_state(start_state)
        for i,pose in enumerate(poses):
            self.move_group.set_pose_target(se3_to_rospose(pose))
            success, plan, time, error=self.move_group.plan()
            if not success:
                rospy.logerr(f"[PLAN_TO_POSES] Planning failed on step {i}/{len(poses)}. {error}")
                raise(Exception("Failed to plan."))
            print("[PLAN_TO_POSES] planned a bit...")
            self.move_group.set_start_state(robot_state_from_traj(plan))
            plans=merge_trajectories(plans,plan)
        plans=self.postprocess_plan(plans)
        return plans
    def postprocess_plan(self, plan):
        '''Postprocess plan by running the retiming algorithm to smoothen it.'''
        if self.retiming_algorithm is None:
            return plan
        ref_state=self.robot.get_current_state()
        retimed_plan=self.move_group.retime_trajectory(ref_state,
                                                       plan,
                                                       velocity_scaling_factor=1.0,
                                                       acceleration_scaling_factor=1.0,
                                                       algorithm=self.retiming_algorithm)
        return retimed_plan
    def execute_plan(self, plan):
        '''Execute plan.'''
        self.move_group.execute(plan, wait=True)
    def switch_controller(self, to: str):
        """Switch to a controller.
        to: controller to switch to. Should be one of those listed in self.all_arm_joint_controllers."""
        assert to in self.all_arm_joint_controllers, f"Switching to the arm controller {to} is invalid."
        rospy.wait_for_service("/controller_manager/switch_controller", timeout=rospy.Duration(secs=1))
        change_service=rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
        return change_service([to], [self.controller], 2)
    def velocity_cmd(self, qd):
        """Send a velocity command. Controller needs to be set to
        arm_forward_velocity_controller for this to have an effect.
        qd: Joint velocities. array of length 7.
        """
        msg=Float64MultiArray()
        msg.layout=MultiArrayLayout()
        msg.layout.data_offset=0
        dim=MultiArrayDimension()
        dim.size=7
        dim.stride=0
        msg.layout.dim=[dim]
        msg.data=qd
        self._velocity_pub.publish(msg)
    def velocity_stop(self):
        "Command the arm joints to have zero velocity."
        self.velocity_cmd([0]*7)
    def RRMC(self, poses):
        """Use resolved-rate motion control (velocity control) to take the end effector through the given poses.
        poses: list of SE3 poses as the waypoints."""
        if self.controller != "arm_forward_velocity_controller":
            self.switch_controller("arm_forward_velocity_controller")
        self.stop=False # clear the stop flag
        for waypoint in poses:
            arrived=False
            print("next point:")
            print(waypoint)
            self.velocity_controller.reset()
            while not arrived and not self.stop:
                v, arrived=self.velocity_controller.step(self.current_pose, waypoint)
                print(f"Current error: {np.sum(np.abs(vc.calculate_error(self.current_pose, waypoint, 0.0)[0]))}", end="\r")
                qd=np.linalg.pinv(self.jacobe) @ v
                # send it
                self.velocity_cmd(qd)
        self.velocity_stop()

class TiagoGripper():
    "Functions relating to the gripper."
    def __init__(self):
        pass
    def grasp(self):
        '''
        call a grasp.
        '''
        rospy.wait_for_service('/parallel_gripper_controller/grasp', timeout=rospy.Duration(secs=1))
        try:
            grasp_service = rospy.ServiceProxy('/parallel_gripper_controller/grasp', Empty)
            grasp_service()
            rospy.loginfo("Grasp service call completed.")
            return True
        except rospy.ServiceException as e:
            rospy.loginfo(f"Grasp service call failed: {e}")
            return False
    def release(self):
        '''
        call a release.
        '''
        rospy.wait_for_service('/parallel_gripper_controller/release', timeout=rospy.Duration(secs=1))
        try:
            release_service = rospy.ServiceProxy('/parallel_gripper_controller/release', Empty)
            release_service()
            rospy.loginfo("Release service call completed")
            return True
        except rospy.ServiceException as e:
            rospy.loginfo(f"Release service call failed: {e}")
            return False
        
################################################################################
## convenience funcs

def merge_trajectories(traj1, traj2):
    "Merge RobotTrajectories traj1 and traj2 together. Modifies traj1."
    if traj1 is None:
        return traj2
    traj1.joint_trajectory.points = traj1.joint_trajectory.points+traj2.joint_trajectory.points
    return traj1

def robot_state_from_traj(traj) -> RobotState:
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

def rostf_to_se3(rostf: TransformStamped) -> sm.SE3:
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

def se3_to_rostf(se3: sm.SE3) -> TransformStamped:
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

def rospose_to_se3(rospose: Pose) -> sm.SE3:
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

def se3_to_rospose(se3: sm.SE3) -> Pose:
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

def stamp_pose(pose: Pose, frame_id: str) -> PoseStamped:
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

def rotate_se3(pose: sm.SE3, axis: str, angle: float) -> sm.SE3:
    "Rotate an SE3 POSE around an AXIS by ANGLE."
    assert axis in ["x", "y", "z"], f"{axis} is not x, y or z"
    t=np.copy(pose.t)
    pose.t=[0,0,0]
    if axis=="x":
        rot=sm.SE3.Rx(angle)
    elif axis=="y":
        rot=sm.SE3.Ry(angle)
    elif axis=="z":
        rot=sm.SE3.Rz(angle)
    pose=pose*rot
    pose.t=t
    return pose

def get_controllers():
    """Return list of all controllers on the robot, with details on whether they're running.
    See the ListControllers service type for the structure."""
    # all the controllers that I can currently see on the robot:
    # arm_velocity_trajectory_controller, arm_impedance_controller,
    # gripper_current_limit_controller, torso_controller,
    # gripper_controller, joint_state_controller,
    # arm_forward_velocity_controller, head_controller,
    # arm_controller, arm_current_limit_controller,
    # mobile_base_controller, wheels_current_limit_controller
    rospy.wait_for_service("/controller_manager/list_controllers", timeout=rospy.Duration(secs=1))
    controller_service=rospy.ServiceProxy("/controller_manager/list_controllers",ListControllers)
    return controller_service()
