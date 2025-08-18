'''Stub with the same API as the TIAGo class without any ROS dependencies for local tests etc. Data should be correct to an order of magnitude.'''
import numpy as np
import spatialmath as sm
from typing import Union, List, Any, Literal, Tuple
from numpy.typing import ArrayLike

def stubprint(text):
    print(f"[STUB] {text}")

def generate_rgb_image():
    return np.random.default_rng().integers(low=np.iinfo(np.uint8).min,
                                            high=np.iinfo(np.uint8).max,
                                            size=(640,480,3),
                                            dtype=np.uint8)

def generate_depth_image(depthmode):
    """DEPTHMODE: if rectified, depth is given in m, not mm. Also instead
    of zeros for unreachable places we get NaNs."""
    if depthmode=="rect":
        img=np.empty((640,480))
        img.fill(np.nan)
        img[200:400,200:300]=np.random.default_rng().integers(low=0.4,
                                                              high=4,
                                                              size=(200,100))
    else:
        img=np.empty((640,480))
        img.fill(0)
        img[200:400,200:300]=np.random.default_rng().integers(low=400,
                                                              high=4000,
                                                              size=(200,100))
    return img


################################################################################
## TIAGO

class Tiago():
    "A class to act as an interface to Tiago's functionality"
    def __init__(self,
                 transforms:bool=True,
                 stats:bool=True,
                 laser:bool=True,
                 torso:bool=True):
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
            self.tfbuffer = None
            self.tflistener = None
            self.tf_pub=None
        # motor stats
        if stats:
            self.stats_sub = None
            self.stats:Dict[str,Any]=dict(zip([
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
                "wheel_right_mode", "wheel_right_current", "wheel_right_velocity", "wheel_right_position", "wheel_right_torque", "wheel_right_temperature"], [0.0]*80))
            "A dictionary that holds temperature, position, current, etc. information for all motors in the robot."  
        #sensors
        if laser:
            self._laser_sub=None
            self.laser:Union[List[Any], None]=[np.arange(-np.pi, np.pi, 0.006),
                                               np.random.default_rng().uniform(low=0.02,
                                                                               high=5.6,
                                                                               size=int(2*np.pi/0.006))]
            "Results of planar laser scan as [angle, distance]. Use laser_cartesian to access the points in a cartesian frame."
            self.laser_frame:str = "base_laser_link"
            "Reference frame for the laser scanner."
        #actuators
        if torso:
            self.torso_sub = None
            self.torso:Union[float,None] = 0.20
            "Torso position."
        self.torso_pub = None
        self.base_pub = None
        self.move_base_client = None
        
        # To play prerecorded motions
        self.play_motion_client=None
        # Moveit stuff
        self.planning_scene=None

    def get_transform(self, wrt: str, this: str) -> sm.SE3:
        """Return an SE3 transform WRT_T_THIS from WRT to THIS.

        wrt: which frame is the transform with respect to?
        this: which frame do we want the transform of?"""
        return sm.SE3()

    def publish_transform(self, pose: sm.SE3, wrt: str, name: str, tries=10):
        """Publish an SE3 transform POSE which gives the relation WRT_T_NAME.

        pose: the transform as an sm.SE3 pose.
        wrt: which frame is it with respect to?
        name: name of the new frame"""
        stubprint(f"Publishing transform {wrt}_T_{name}:\n{pose}")

    def play_motion(self, motion_name: str):
        """Play the motion specified by MOTION_NAME"""
        stubprint(f"Playing motion {motion_name}")
        
    def home(self):
        "Move the robot to home configuration."
        self.play_motion("home")

    def say(self, words: str, language:str="en"):
        """Speak using espeak-ng.
        
        words: What to say
        language: either en or tr"""
        stubprint(f"Using TTS to say {words}")
        
    def move_torso(self, goal: float, duration=None):
        "Move torso to specified point."
        stubprint(f"Moving torso to {goal}")
        self.torso=goal
    
    def move(self, x:float=0, y:float=0, z:float=0, rx:float=0, ry:float=0, rz:float=0):
        """Move Tiago's base with the specified velocity.

        +x is forward.
        +rz is to the left (CCW)
        """
        stubprint(f"Move command: {x=} {y=} {z=} {rx=} {ry=} {rz=}")

    def move_to(self, posestamped):
        """Command Tiago to move to a certain pose using its nav stack.
        Check if done with is_move_done
        """
        stubprint(f"Moving to {posestamped}")
        
    def is_move_done(self) -> bool:
        """Returns move_base server state, that is, whether the pose commanded by move_to has been reached."""
        return True
    
    def look_at(self, target:sm.SE3,
                threshold:float=0.15,
                base_kp:float=0.3):
        """Rotate to face target.
        TARGET: SE3 wrt "map" frame.
        THRESHOLD: when to stop in rads"""
        stubprint(f"Looking at target:\n{target}")
        
    @property
    def laser_cartesian(self)->Union[ArrayLike, None]:
        "Results of the planar laser scan as an array of 2D points according to laser frame (+x forward, +y left)."
        if self.laser is not None:
            cossin=np.array([np.cos(self.laser[0]), np.sin(self.laser[0])])
            print(f"{cossin.shape=}")
            print(len(self.laser[0])-1)
            return (cossin*self.laser[1][:len(self.laser[0])-1]).T
        else:
            return None
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
                 pointcloud:bool=True,
                 caminfo:bool=True,
                 motor:bool=True):
        ## Cameras
        self.camera_frame:str="xtion_rgb_optical_frame"
        "The frame used for the camera"
        
        # RGBD Camera
        self._bridge = None
        if rgb=="raw":
            self._rgb_sub = None
        elif rgb=="rect":
            self._rgb_sub = None
        if depth=="raw":
            self._depth_sub = None
        elif depth=="rect":
            self._depth_sub = None
        if pointcloud:
            self._pointcloud_sub=None
        self.rgb = generate_rgb_image()
        "Latest fetched RGB image. Size should be 640x480."
        self.depth = generate_depth_image(depth)
        "Latest fetched depth image. Size should be 640x480."
        self.pointcloud=np.random.default_rng().integers(low=0.4,
                                                         high=4,
                                                         size=(640*480,3))
        "Pointcloud obtained from the depth image."
        self.pointcloud_frame=None
        "The name of the frame that the pointcloud is relative to."
        
        # Camera intrinsics
        if caminfo:
            self._rgb_info_sub = None
            subtopic_name="depth" if depth=="raw" else "depth_registered"
            self._depth_info_sub = None
        self.rgb_raw_intrinsic:Union[ArrayLike,None] = np.array([[0.4, 0, 240],
                                                                 [0, 0.4, 320],
                                                                 [0, 0, 1]])
        "RGB camera intrinsic matrix."
        self.rgb_raw_distortion_model:Union[str,None] = None
        "RGB camera distortion model. See CameraInfo ROSmsg for details."
        self.rgb_raw_distortion:Union[ArrayLike,None] = None
        "RGB camera distortion parameters. See CameraInfo ROSmsg for details."
        self.depth_intrinsic:Union[ArrayLike,None] = np.array([[0.4, 0, 240],
                                                               [0, 0.4, 320],
                                                               [0, 0, 1]])
        "DEPTH camera intrinsic matrix."
        self.depth_distortion_model:Union[str,None] = None
        "DEPTH camera distortion model. See CameraInfo ROSmsg for details."
        self.depth_distortion:Union[ArrayLike,None] = None
        "DEPTH camera distortion parameters. See CameraInfo ROSmsg for details."
        
        ## Actuators
        if motor:
            self._motor_sub = None
        self.motor:Union[List[float],None] = [0.0, 0.0]
        "Motor positions, 2 element list. The first one is yaw, the second is pitch."
        self._motor_pub = None

    def move(self, pan:float, tilt:float, duration=None):
        "Rotate head to specified values."
        self.motor=[pan,tilt]
        stubprint(f"Moving head to {pan=} {tilt=}")

################################################################################
## ARM
class TiagoArm():
    "Functions relating to Tiago's arm: motion planning, etc."
    def __init__(self, retiming_algorithm: str, velocity_controller=None):
        self.robot=None
        self.move_group=None
        "docs: https://docs.ros.org/en/api/moveit_commander/html/classmoveit__commander_1_1move__group_1_1MoveGroupCommander.html"
        self.endeff_frame="arm_tool_link"
        "The end-effector frame used by moveit."
        self.planning_frame="base_footprint"
        "The planning frame used by moveit."
        
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
        self._velocity_pub=None
        
        # if velocity_controller is None:
            # self.velocity_controller:vc.VelocityController=vc.PIDController(Kp=2.0, Ki=0.1, integral_max=2.0, threshold=0.05)
        # else:
            # self.velocity_controller=velocity_controller
    @property
    def q(self) -> List[float]:
        "Return the current joint configuration, a 7-element list."
        return [0.0]*7
    @property
    def current_pose(self) -> sm.SE3:
        "The current pose of the arm end effector (arm_tool_link) as SE3 wrt the planning frame (base_footprint)."
        return sm.SE3.Rand()
    @property
    def jacobo(self) -> ArrayLike:
        """Returns the arm Jacobian wrt. the planning frame (base_footprint).
        Matrix is 6x7. Each row shows how the joints will affect spatial velocity [vx vy vz wx wy wz]."""
        return np.random.default_rng().random(size=(6,7))
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
    def plan_cartesian_trajectory(self, trajectory:List[sm.SE3], eef_step:float=0.001, jump_threshold:float=0.0, start_state=None):
        '''Plan the given trajectory.
        trajectory: a list of SE3 poses denoting the waypoints.
        eef_step: view move_group documentation (likely the interpolation increment)
        jump_threshold: not sure, view move_group documentation
        start_state: optional starting RobotState, default is the current state.'''
        return NotImplementedError()
        # if start_state is None:
        #     start_state=self.move_group.get_current_state()
        # self.move_group.set_start_state(start_state)
        # ros_trajectory=[se3_to_rospose(se3) for se3 in trajectory]
        # (plan, fraction) = self.move_group.compute_cartesian_path(ros_trajectory,
        #                                                           eef_step,
        #                                                           jump_threshold,
        #                                                           avoid_collisions=True,
        #                                                           path_constraints=None)
        # plan=self.postprocess_plan(plan)
        # print(f"[PLAN_CARTESIAN_TRAJECTORY] Planning complete, successfully planned {fraction} of the path.")
        # return plan, fraction
    def plan_to_pose(self, pose: sm.SE3) -> Tuple[List[Any], float]:
        '''Plan to the given pose.
        POSE is an SE3 pose.'''
        return NotImplementedError()
        # self.move_group.set_pose_target(se3_to_rospose(pose))
        # success, plan, time, error=self.move_group.plan()
        # print(f"[PLAN_TO_POSE] Planning took {time} seconds.")
        # if not success:
        #     print(f"[PLAN_TO_POSE] Planning failed. Error code {error}")
        # return (plan, success)
    def plan_to_poses(self, poses:List[sm.SE3], start_state=None):
        '''Plan to a sequential trajectory of POSES.
        poses: a list of SE3 poses.
        start_state: optional starting RobotState, default is the current state.'''
        return NotImplementedError()
        # if start_state is None:
        #     start_state=self.move_group.get_current_state()
        # plans=None
        # self.move_group.set_start_state(start_state)
        # for i,pose in enumerate(poses):
        #     self.move_group.set_pose_target(se3_to_rospose(pose))
        #     success, plan, time, error=self.move_group.plan()
        #     if not success:
        #         rospy.logerr(f"[PLAN_TO_POSES] Planning failed on step {i}/{len(poses)}. {error}")
        #         raise(Exception("Failed to plan."))
        #     print("[PLAN_TO_POSES] planned a bit...")
        #     self.move_group.set_start_state(robot_state_from_traj(plan))
        #     plans=merge_trajectories(plans,plan)
        # plans=self.postprocess_plan(plans)
        # return plans
    def postprocess_plan(self, plan, retiming_algorithm=None):
        '''Postprocess plan by running the retiming algorithm to smoothen it.

        retiming_algorithm: Leave as None to use the one picked as the class attribute.'''
        return NotImplementedError()
        # if retiming_algorithm is None:
        #     retiming_algorithm=self.retiming_algorithm
        # if retiming_algorithm is None:
        #     return plan
        # ref_state=self.robot.get_current_state()
        # retimed_plan=self.move_group.retime_trajectory(ref_state,
        #                                                plan,
        #                                                velocity_scaling_factor=1.0,
        #                                                acceleration_scaling_factor=1.0,
        #                                                algorithm=retiming_algorithm)
        # return retimed_plan
    def execute_plan(self, plan):
        '''Execute plan.'''
        stubprint(f"Executing plan")
        # self.move_group.execute(plan, wait=True)
    def switch_controller(self, to: str):
        """Switch to a controller.
        to: controller to switch to. Should be one of those listed in self.all_arm_joint_controllers."""
        assert to in self.all_arm_joint_controllers, f"Switching to the arm controller {to} is invalid."
        return NotImplementedError()
        # rospy.wait_for_service("/controller_manager/switch_controller", timeout=rospy.Duration(secs=1))
        # change_service=rospy.ServiceProxy("/controller_manager/switch_controller", SwitchController)
        # return change_service([to], [self.controller], 2)
    def velocity_cmd(self, qd:ArrayLike):
        """Send a velocity command. Controller needs to be set to
        arm_forward_velocity_controller for this to have an effect.
        qd: Joint velocities. array of length 7.
        """
        return NotImplementedError()
        # msg=Float64MultiArray()
        # msg.layout=MultiArrayLayout()
        # msg.layout.data_offset=0
        # dim=MultiArrayDimension()
        # dim.size=7
        # dim.stride=0
        # msg.layout.dim=[dim]
        # msg.data=qd
        # self._velocity_pub.publish(msg)
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
        return NotImplementedError()
        # if self.controller != "arm_forward_velocity_controller":
        #     self.switch_controller("arm_forward_velocity_controller")
        # self.stop=False # clear the stop flag
        # recording=[[[rospy.get_time(), self.current_pose]]] # to record poses
        # for waypoint in poses:
        #     arrived=False
        #     segment_recording=[] # to record poses for the current segment
        #     print(waypoint)
        #     self.velocity_controller.reset()
        #     while not arrived and not self.stop:
        #         v, arrived=self.velocity_controller.step(self.current_pose, waypoint)
        #         print(f"Current error: {np.sum(np.abs(vc.calculate_error(self.current_pose, waypoint, 0.0)[0]))}", end="\r")
        #         qd=np.linalg.pinv(self.jacobe) @ v
        #         # send it
        #         self.velocity_cmd(qd)
        #         # save current pose
        #         segment_recording.append([rospy.get_time(), self.current_pose])
        #     recording.append(segment_recording) # save poses for the segment
        # self.velocity_stop()
        # return recording

class TiagoGripper():
    "Functions relating to the gripper."
    def __init__(self):
        pass
    def grasp(self):
        '''
        call a grasp.
        '''
        return NotImplementedError()
        # rospy.wait_for_service('/parallel_gripper_controller/grasp', timeout=rospy.Duration(secs=1))
        # try:
        #     grasp_service = rospy.ServiceProxy('/parallel_gripper_controller/grasp', Empty)
        #     grasp_service()
        #     rospy.loginfo("Grasp service call completed.")
        #     return True
        # except rospy.ServiceException as e:
        #     rospy.loginfo(f"Grasp service call failed: {e}")
        #     return False
    def release(self):
        '''
        call a release.
        '''
        return NotImplementedError()
        # rospy.wait_for_service('/parallel_gripper_controller/release', timeout=rospy.Duration(secs=1))
        # try:
        #     release_service = rospy.ServiceProxy('/parallel_gripper_controller/release', Empty)
        #     release_service()
        #     rospy.loginfo("Release service call completed")
        #     return True
        # except rospy.ServiceException as e:
        #     rospy.loginfo(f"Release service call failed: {e}")
        #     return False
        
################################################################################
## convenience funcs

def merge_trajectories(traj1, traj2):
    "Merge RobotTrajectories traj1 and traj2 together. Modifies traj1."
    raise NotImplementedError()
    
def robot_state_from_traj(traj):
    "Returns the RobotState that the robot will get to at the end of a RobotTrajectory traj."
    raise NotImplementedError()

def rostf_to_se3(rostf) -> sm.SE3:
    "convert a ros tf object into sm.SE3"
    raise NotImplementedError()

def se3_to_rostf(se3: sm.SE3):
    "convert an sm.SE3 object into a ros tf"
    raise NotImplementedError()

def rospose_to_se3(rospose) -> sm.SE3:
    "convert a ros pose into sm.SE3"
    raise NotImplementedError()

def se3_to_rospose(se3: sm.SE3):
    "convert an sm.SE3 into a ros pose"
    raise NotImplementedError()

def stamp_pose(pose, frame_id: str):
    raise NotImplementedError()
    
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
    raise NotImplementedError()
