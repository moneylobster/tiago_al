'''Tiago ROS abstraction layer
to write stuff in python without worrying about the ROS parts too much'''


import tiago_al.velocity_controllers as vc

import subprocess
from typing import Any, Union, List, Dict, Tuple, Literal
import numpy as np
from numpy.typing import ArrayLike
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
from moveit_msgs.msg import RobotState, RobotTrajectory
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
                 transforms:bool=True,
                 stats:bool=True,
                 laser:bool=True,
                 torso:bool=True):
        # start a ros node
        rospy.init_node("tiago_al", anonymous=True, disable_signals=True)
        ## Submodules
        # classes relating exclusively to individual components
        self.head:TiagoHead=TiagoHead()
        "Subclass for the head of the robot. Includes camera data."
        self.arm:TiagoArm=TiagoArm("time_optimal_trajectory_generation")
        "Subclass for the arm of the robot. Includes motion planning stuff."
        self.gripper:TiagoGripper=TiagoGripper()
        "Subclass for things relating to gripper control."
        # transforms
        if transforms:
            self.tfbuffer = tf2_ros.Buffer()
            self.tflistener = tf2_ros.TransformListener(self.tfbuffer)
            self.tf_pub=rospy.Publisher("/tf_static", TFMessage, queue_size=10)
        # motor stats
        if stats:
            self.stats_sub = rospy.Subscriber("/motors_statistics/values", StatisticsValues, self._stats_callback, queue_size=10)
            self.stats:Dict[str,Any]={}
            "A dictionary that holds temperature, position, current, etc. information for all motors in the robot."  
        #sensors
        if laser:
            self._laser_sub=rospy.Subscriber("/scan", LaserScan,
                                             self._laser_callback, queue_size=10)
            self.laser = None
            "Results of planar laser scan as [angle, distance]. Use laser_cartesian to access the points in a cartesian frame."
            self.laser_frame:str = "base_laser_link"
            "Reference frame for the laser scanner."
        #actuators
        if torso:
            self.torso_sub = rospy.Subscriber("/torso_controller/state", JointTrajectoryControllerState,
                                              self._torso_callback, queue_size=10)
            self.torso:Union[float,None] = None
            "Torso position."
        self.torso_pub = rospy.Publisher("/torso_controller/command", JointTrajectory, queue_size=10)
        self.base_pub = rospy.Publisher("/mobile_base_controller/cmd_vel", Twist, queue_size=10)
        self.move_base_client = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        
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

    def say(self, words: str, language:str="en"):
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

    def move_torso(self, goal: float, duration:rospy.Duration=rospy.Duration(1)):
        "Move torso to specified point."
        trajpt=JointTrajectoryPoint()
        trajpt.positions=[goal]
        trajpt.time_from_start=duration
        torso_msg=JointTrajectory()
        torso_msg.joint_names=[
            "torso_lift_joint"
        ]
        torso_msg.points=[trajpt]
        self.torso_pub.publish(torso_msg)
        rospy.sleep(duration)

    def move(self, x:float=0, y:float=0, z:float=0, rx:float=0, ry:float=0, rz:float=0):
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
        print("Trying to reach move_base_simple...")
        if self.move_base_client.wait_for_server(rospy.Duration(secs=1)):
            print("move_base_simple reached")
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
            print("sent move_base goal")
            return True
        else:
            rospy.logerr("[MOVE_TO] Couldn't access move_base server.")
            return False

    def is_move_done(self) -> bool:
        """Returns move_base server state, that is, whether the pose commanded by move_to has been reached."""
        if self.move_base_client.wait_for_result(rospy.Duration(secs=1)):
            return self.move_base_client.get_result()
        else:
            rospy.logerr("[IS_MOVE_DONE] Move_base server can't be reached!")
            return False

    def look_at(self, target:sm.SE3,
                threshold:float=0.15,
                base_kp:float=0.3):
        """Rotate to face target.
        TARGET: SE3 wrt "map" frame.
        THRESHOLD: when to stop in rads"""
        def angerr(target):
            B_T_M=self.get_transform("base_footprint", "map")
            M_T_T=target
            B_T_T=B_T_M @ M_T_T
            vec2d=B_T_T.t[:2]
            angerr=np.arctan2(vec2d[1], vec2d[0])
            return angerr
        while abs((theta:=angerr(target)))>threshold and not rospy.is_shutdown():
            self.move(rz=base_kp*theta)
        
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
        self.laser=np.array([angles, data.ranges[:len(angles)]])
    def slice_laser(self, from_angle, to_angle):
        """Get laser readouts in the specified angle range.
        Angles are both inclusive.
        from_angle should be smaller than to_angle.
        """
        assert from_angle>=self.laser[0,0], f"from_angle ({from_angle} rad) is smaller than leftmost limit ({self.laser[0,0]} rad)"
        assert to_angle<=self.laser[0,-1], f"to_angle ({to_angle} rad) larger than rightmost limit ({self.laser[0,-1]} rad)"
        assert from_angle<to_angle, f"from_angle ({from_angle} rad) should be smaller than to_angle ({to_angle} rad)."
        first_idx=np.searchsorted(self.laser[0,:], from_angle, side="left")
        second_idx=np.searchsorted(self.laser[0,:], to_angle, side="right")
        return self.laser[:,first_idx:second_idx]
    def slice_laser_cartesian(self, from_angle, to_angle):
        """Get laser readouts in the specified angle range and convert
        to cartesian.
        Angles are both inclusive.
        from_angle should be smaller than to_angle."""
        laser=self.slice_laser(from_angle, to_angle)
        cossin=np.array([np.cos(laser[0,:]), np.sin(laser[0,:])])
        return (cossin*laser[1,:][:len(laser[0,:])]).T
    @property
    def laser_cartesian(self)->Union[ArrayLike, None]:
        "Results of the planar laser scan as an array of 2D points according to laser frame (+x forward, +y left)."
        if self.laser is not None:
            return self.slice_laser_cartesian(self.laser[0,0], self.laser[0,-1])
        else:
            return None

    @property
    def base_pose_wrt_map(self)->sm.SE3:
        "Base_footprint wrt. map frame: M_T_B"
        return self.get_transform("map", "base_footprint")
    
    def quit(self):
        "Safely stop/shutdown Tiago. Also stops movements if velocity controller is active."
        if self.arm.controller=="arm_forward_velocity_controller":
            self.arm.stop=True
            self.arm.velocity_stop()
        
################################################################################
## HEAD
        
class TiagoHead():
    "Functions relating to Tiago's head. Cameras, motors etc."
    def __init__(self,
                 rgb:Literal["raw","rect",None]="raw",
                 depth:Literal["raw","rect",None]="raw",
                 pointcloud:Literal["raw","rect",None]="raw",
                 caminfo:bool=True,
                 motor:bool=True):
        ## Cameras
        self.camera_frame:str="xtion_rgb_optical_frame"
        "The frame used for the camera"
        
        # RGBD Camera
        self._bridge = CvBridge()
        if rgb=="raw":
            self._rgb_sub = rospy.Subscriber("/xtion/rgb/image_raw", Image,
                                             self._rgb_callback, queue_size=10)
        elif rgb=="rect":
            self._rgb_sub = rospy.Subscriber("/xtion/rgb/image_rect_color", Image,
                                             self._rgb_callback, queue_size=10)
        if depth=="raw":
            self._depth_sub = rospy.Subscriber("/xtion/depth/image_raw", Image,
                                               self._depth_callback, queue_size=10)
        elif depth=="rect":
            self._depth_sub = rospy.Subscriber("/xtion/depth_registered/hw_registered/image_rect_raw", Image,
                                               self._depth_callback, queue_size=10)
        if pointcloud=="raw":
            self._pointcloud_sub=rospy.Subscriber("/throttle_filtering_points/filtered_points", PointCloud2,
                                                  self._pointcloud_callback, queue_size=10)
        elif pointcloud=="rect":
            # NB. This one is a bit me-specific, launches the
            # depth-camera rectifier that I wrote and then subscribes
            # to its output.
            subprocess.Popen(["roslaunch", "depth_republisher", "depth_republisher.launch"],
                             stdout=subprocess.DEVNULL) #launch republisher
            self._pointcloud_sub=rospy.Subscriber("/points_republish", PointCloud2,
                                                  self._pointcloud_callback, queue_size=10)
        self.rgb:Union[None,ArrayLike] = None
        "Latest fetched RGB image. Size should be 640x480."
        self.depth:Union[None, ArrayLike] = None
        "Latest fetched depth image. Size should be 640x480."
        self.pointcloud:Union[None, ArrayLike]=None
        "Pointcloud obtained from the depth image."
        self.pointcloud_frame:str=None
        "The name of the frame that the pointcloud is relative to."
        
        # Camera intrinsics
        if caminfo:
            self._rgb_info_sub = rospy.Subscriber("/xtion/rgb/camera_info", CameraInfo,
                                                  self._rgb_info_callback, queue_size=10)
            
            subtopic_name="depth" if depth=="raw" else "depth_registered"
            self._depth_info_sub = rospy.Subscriber(f"/xtion/{subtopic_name}/camera_info", CameraInfo,
                                                    self._depth_info_callback, queue_size=10)
        self.rgb_raw_intrinsic:Union[ArrayLike,None] = None
        "RGB camera intrinsic matrix."
        self.rgb_raw_distortion_model:Union[str,None] = None
        "RGB camera distortion model. See CameraInfo ROSmsg for details."
        self.rgb_raw_distortion:Union[ArrayLike,None] = None
        "RGB camera distortion parameters. See CameraInfo ROSmsg for details."
        self.depth_intrinsic:Union[ArrayLike,None] = None
        "DEPTH camera intrinsic matrix."
        self.depth_distortion_model:Union[str,None] = None
        "DEPTH camera distortion model. See CameraInfo ROSmsg for details."
        self.depth_distortion:Union[ArrayLike,None] = None
        "DEPTH camera distortion parameters. See CameraInfo ROSmsg for details."
        
        ## Actuators
        if motor:
            self._motor_sub = rospy.Subscriber("/head_controller/state", JointTrajectoryControllerState,
                                               self._motor_callback, queue_size=10)
        self.motor:Union[List[float],None] = None
        "Motor positions, 2 element list. The first one is yaw, the second is pitch."
        self._motor_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=10)

        # disable head manager
        head_disable_client=actionlib.SimpleActionClient("/pal_head_manager/disable", DisableAction)
        if head_disable_client.wait_for_server(rospy.Duration(secs=1)):
            disable_goal=DisableGoal()
            head_disable_client.send_goal(disable_goal)
        else:
            rospy.logerr("[TIAGO_HEAD] Couldn't access pal_head_manager, not disabling it.")
        
    def move(self, pan:float, tilt:float, duration:rospy.Duration=rospy.Duration(1)):
        "Rotate head to specified values."
        trajpt=JointTrajectoryPoint()
        trajpt.positions=[pan, tilt]
        trajpt.time_from_start=duration
        motor_msg=JointTrajectory()
        motor_msg.joint_names=[
            "head_1_joint",
            "head_2_joint"
        ]
        motor_msg.points=[trajpt]
        self._motor_pub.publish(motor_msg)
        rospy.sleep(duration)
                
    def _rgb_callback(self, data):
        self.rgb = self._bridge.imgmsg_to_cv2(data)
    def _depth_callback(self, data):
        self.depth = self._bridge.imgmsg_to_cv2(data)
    def _pointcloud_callback(self, data):
        self.pointcloud_frame=data.header.frame_id
        self.pointcloud=np.array(list(read_points(data)))
    def _motor_callback(self,data):
        self.motor = data.actual.positions
    def _rgb_info_callback(self, data):
        self.rgb_raw_intrinsic=np.array(data.K).reshape((3,3))
        self.rgb_raw_distortion_model=data.distortion_model
        self.rgb_raw_distortion=np.array(data.D)
    def _depth_info_callback(self, data):
        self.depth_intrinsic=np.array(data.K).reshape((3,3))
        self.depth_distortion_model=data.distortion_model
        self.depth_distortion=np.array(data.D)
################################################################################
## ARM
class TiagoArm():
    "Functions relating to Tiago's arm: motion planning, etc."
    def __init__(self, retiming_algorithm: str, velocity_controller:Union[vc.VelocityController,None]=None):
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
        self.retiming_algorithm:str=retiming_algorithm

        # TODO: it's possible to do this a bit better by checking the
        # claimed_resources part of the listcontrollers response and
        # seeing if they contain all arm joints instead of hardcoding
        # possible controller names.
        self.all_arm_joint_controllers:List[str]=[
            "arm_velocity_trajectory_controller",
            "arm_impedance_controller",
            "arm_forward_velocity_controller",
            "arm_controller"]
        "Controllers that can be used with the arm. Call switch_controller to use them."

        self.stop:bool=False
        "Flag to determine whether arm should stop moving. Currently only works for the velocity controller."
        self._velocity_pub=rospy.Publisher("/arm_forward_velocity_controller/command", Float64MultiArray, queue_size=1)
        
        if velocity_controller is None:
            self.velocity_controller:vc.VelocityController=vc.PIDController(Kp=2.0, Ki=0.1, integral_max=2.0, threshold=0.05)
        else:
            self.velocity_controller=velocity_controller
    @property
    def q(self) -> List[float]:
        "Return the current joint configuration, a 7-element list."
        return self.move_group.get_current_joint_values()
    @property
    def current_pose(self) -> sm.SE3:
        "The current pose of the arm end effector (arm_tool_link) as SE3 wrt the planning frame (base_footprint)."
        return rospose_to_se3(self.move_group.get_current_pose())
    @property
    def jacobo(self) -> ArrayLike:
        """Returns the arm Jacobian wrt. the planning frame (base_footprint).
        Matrix is 6x7. Each row shows how the joints will affect spatial velocity [vx vy vz wx wy wz]."""
        return np.array(self.move_group.get_jacobian_matrix(self.q))
    @property
    def jacobe(self) -> ArrayLike:
        """Returns the arm Jacobian wrt. the end-eff frame (arm_tool_link).
        Matrix is 6x7. Each row shows how the joints will affect spatial velocity [vx vy vz wx wy wz]."""
        return sm.base.tr2jac(self.current_pose.inv().A) @ self.jacobo
    @property
    def controller(self) -> Union[str, None]:
        """Returns the currently running arm controller."""
        controllers=get_controllers()
        # There should only be one arm joint controller running. I
        # think ROS prevents multiple from running so we don't need to
        # check.
        for ctrl in controllers.controller:
            if ctrl.name in self.all_arm_joint_controllers and ctrl.state=="running":
                return ctrl.name
        return None
    def plan_cartesian_trajectory(self, trajectory:List[sm.SE3], eef_step:float=0.001, jump_threshold:float=0.0, start_state:RobotState=None) -> Tuple[RobotTrajectory, float]:
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
    def plan_to_pose(self, pose: sm.SE3) -> Tuple[List[Any], float]:
        '''Plan to the given pose.
        POSE is an SE3 pose.'''
        self.move_group.set_pose_target(se3_to_rospose(pose))
        success, plan, time, error=self.move_group.plan()
        print(f"[PLAN_TO_POSE] Planning took {time} seconds.")
        if not success:
            print(f"[PLAN_TO_POSE] Planning failed. Error code {error}")
        return (plan, success)
    def plan_to_poses(self, poses:List[sm.SE3], start_state:RobotState=None) -> RobotTrajectory:
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
    def postprocess_plan(self, plan:RobotTrajectory, retiming_algorithm=None)->RobotTrajectory:
        '''Postprocess plan by running the retiming algorithm to smoothen it.

        retiming_algorithm: Leave as None to use the one picked as the class attribute.'''
        if retiming_algorithm is None:
            retiming_algorithm=self.retiming_algorithm
        if retiming_algorithm is None:
            return plan
        ref_state=self.robot.get_current_state()
        retimed_plan=self.move_group.retime_trajectory(ref_state,
                                                       plan,
                                                       velocity_scaling_factor=1.0,
                                                       acceleration_scaling_factor=1.0,
                                                       algorithm=retiming_algorithm)
        return retimed_plan
    def execute_plan(self, plan:RobotTrajectory):
        '''Execute plan.'''
        self.move_group.execute(plan, wait=True)
        # occasionally the movement execution can raise an error, but
        # the arm moves without problems. In these cases this function
        # terminates without waiting for the movement to
        # end. Therefore we check that the movement has finished
        # manually.
        
        thresh=0.1
        # get final q in plan
        final_q=np.array(plan.joint_trajectory.points[-1].positions)
        # wait until current q matches final q of plan (to a
        # threshold)
        while np.linalg.norm(np.abs(self.q-final_q))>=thresh:
            rospy.sleep(0.1)
            
    def switch_controller(self, to: str):
        """Switch to a controller.
        to: controller to switch to. Should be one of those listed in self.all_arm_joint_controllers."""
        assert to in self.all_arm_joint_controllers, f"Switching to the arm controller {to} is invalid."
        rospy.wait_for_service("/controller_manager/switch_controller", timeout=rospy.Duration(secs=1))
        change_service=rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
        return change_service([to], [self.controller], 2)
    def velocity_cmd(self, qd:ArrayLike):
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
    def RRMC(self, poses:List[sm.SE3], keypoints=None, keypoint_threshold=None, interp_threshold=None):
        """Use resolved-rate motion control (velocity control) to take the end effector through the given poses.
        Returns endeff poses recorded during movement (wrt planning frame).

        The recording format is as follows: [[[t0, p0]], [[t1.1, p1.1], [t1.2, p1.2] ...], [[t2.1, p2.1], [t2.2, p2.2], ...], ...]
        Where px.y is the y'th pose recorded when going toward the x'th waypoint. tx.y is the corresponding time.
        
        poses: list of SE3 poses as the waypoints. 
        """
        if self.controller != "arm_forward_velocity_controller":
            self.switch_controller("arm_forward_velocity_controller")
        self.stop=False # clear the stop flag
        recording=[[[rospy.get_time(), self.current_pose]]] # to record poses
        for waypoint in poses:
            arrived=False
            segment_recording=[] # to record poses for the current segment
            print(waypoint)
            self.velocity_controller.reset()
            while not arrived and not self.stop:
                v, arrived=self.velocity_controller.step(self.current_pose, waypoint)
                print(f"Current error: {np.sum(np.abs(vc.calculate_error(self.current_pose, waypoint, 0.0)[0]))}", end="\r")
                qd=np.linalg.pinv(self.jacobe) @ v
                # send it
                self.velocity_cmd(qd)
                # save current pose
                segment_recording.append([rospy.get_time(), self.current_pose])
            recording.append(segment_recording) # save poses for the segment
        self.velocity_stop()
        return recording

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

def merge_trajectories(traj1:RobotTrajectory, traj2:RobotTrajectory):
    "Merge RobotTrajectories traj1 and traj2 together. Modifies traj1."
    if traj1 is None:
        return traj2
    traj1.joint_trajectory.points = traj1.joint_trajectory.points+traj2.joint_trajectory.points
    return traj1

def robot_state_from_traj(traj:RobotTrajectory) -> RobotState:
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

def find_perp_vector(vector:ArrayLike)->ArrayLike:
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

def get_controllers()->ListControllers:
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
