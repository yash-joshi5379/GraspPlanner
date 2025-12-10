from grippers.baseGripper import Gripper
import numpy as np
import pybullet as p
import time


class ThreeFingerHand(Gripper):
    """
    A three-finger gripper implementation inheriting from the base Gripper class.

    This class represents a three-finger hand gripper (SDH - Schunk Dexterous Hand)
    with independent control over each finger. Each finger has multiple joints for
    grasping, preshaping, and upper movement control.
    """
    GRASP_JOINTS = [1, 4, 7]
    PRESHAPE_JOINTS = [2, 5, 8]
    UPPER_JOINTS = [3, 6, 9]

    def __init__(self):
        """
        Initialize the ThreeFingerHand gripper.

        Loads the SDH URDF model and sets the gripper's default roll and quaternion
        offset for proper gripper orientation in the simulation.
        """
        super().__init__(urdf_path="./urdf_files/grippers/threeFingers/sdh/sdh.urdf")
        self.roll = np.pi / 2
        self.quat_offset = p.getQuaternionFromEuler([0, np.pi / 2, 0])

    # -------------------------------
    # Finger Control
    # -------------------------------

    def preshape(self):
        """
        Move the three fingers into a preshape pose.

        Positions the preshape joints (2, 5, 8) to prepare the
        gripper for grasping. Sets the open flag to False.
        """
        for i in ThreeFingerHand.PRESHAPE_JOINTS:
            self.apply_joint_command(i, target=0.9, force=999)
        self.open = False

    def open_gripper(self):
        """
        Open the gripper to be ready to grab an object.

        Iteratively adjusts all fingers by moving preshape joints, upper joints,
        and grasp joints towards their open positions. Continues until fully open
        or for a maximum of 1000 iterations. Sets the open flag to True.
        """
        closed, iteration = True, 0
        while closed and not self.open:
            joints = self.get_joint_positions()
            closed = False
            for k in range(self.num_joints):
                if k in ThreeFingerHand.PRESHAPE_JOINTS and joints[k] >= 0.9:
                    self.apply_joint_command(k, joints[k] - 0.05)
                    closed = True
                elif k in ThreeFingerHand.UPPER_JOINTS and joints[k] <= 0.9:
                    self.apply_joint_command(k, joints[k] - 0.05)
                    closed = True
                elif k in ThreeFingerHand.GRASP_JOINTS and joints[k] <= 0.9:
                    self.apply_joint_command(k, joints[k] - 0.05)
                    closed = True
            iteration += 1
            if iteration > 1000:
                break
            p.stepSimulation()
            time.sleep(0.0001)
        self.open = True

    def close_gripper(self):
        """
        Close the gripper to grab an object.

        Moves the grasp joints (specifically joint 7) to -0.5 and grasp joints 1 and 4
        to 0.5 with high force to firmly grip the object. Includes a 1-second pause
        to allow the gripper to settle. Sets the open flag to False.
        """
        self.apply_joint_command(
            joint=7,
            target=-0.5,
            force=999,
            maxVelocity=1.5)
        for j in [1, 4]:
            self.apply_joint_command(
                joint=j, target=0.5, force=999, maxVelocity=1)
        time.sleep(1)
        self.open = False
