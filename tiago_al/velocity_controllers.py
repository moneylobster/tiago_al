"""Various velocity controller implementations."""
import numpy as np
import spatialmath as sm


def calculate_error(wTe, wTep, threshold):
    """
    Calculate the positional error as a 6-element vector [tx ty tz rx ry rz]. Ported from roboticstoolbox-python (the p_servo function).
    
    wTe: current endeff pose wrt world. SE3.
    wTep: desired endeff pose wrt world. SE3.
    threshold: how low the error should be to mark as arrived. float.
    """
    # pose difference
    eTep=np.linalg.inv(wTe.A) @ wTep.A
    e=np.empty(6)
    # translational error
    e[:3]=eTep[:3,3]
    # angular error.
    e[3:]=sm.base.tr2rpy(eTep)
    arrived = True if np.sum(np.abs(e)) < threshold else False
    return e, arrived

class VelocityController():
    def reset(self):
        pass
    def step(self):
        pass
    
class PController(VelocityController):
    """Apply a P term to error. Resolved-rate motion control."""
    def __init__(self, gain=1.0, threshold=0.01):
        self.gain=gain
        self.threshold=threshold
    def step(self, wTe, wTep):
        err, arrived=calculate_error(wTe, wTep, self.threshold)
        k = self.gain*np.eye(6)
        v = k@err
        return v, arrived

class PIDController(VelocityController):
    "PID controller with saturating integral term. Independent form."
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0, integral_max=1.0, threshold=0.01):
        self.Kp=Kp
        self.Ki=Ki
        self.Kd=Kd
        self.integral_max=0
        self.threshold=threshold
        self.integral_memory=0
        "Used as an accumulator for the I term."
        self.err_prev=None
        "Error value from previous timestep. Used to calculate D term."
    def reset(self):
        "Resets the I term. Calling before changing SP is recommended."
        self.integral_memory=0
    def step(self, wTe, wTep):
        err, arrived=calculate_error(wTe, wTep, self.threshold)
        self.integral_memory=np.clip(self.integral_memory+err, -self.integral_max, self.integral_max)
        k = self.Kp*np.eye(6) + self.Ki*self.integral_memory + self.Kd*(err-self.err_prev)
        v = k@err
        return v, arrived
