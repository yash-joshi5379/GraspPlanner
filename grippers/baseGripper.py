import numpy as np
from abc import ABC, abstractmethod
import pybullet as p
from pybullet_utils import transformations as tf


class Gripper(ABC):
    """
    Abstract base class for a gripper to grasp objects.

    This class handles gripper initialization, position and orientation control, movement,
    and finger actuation. The gripper is spawned at a random position on a spherical
    surface and can be controlled to move towards and grasp objects.
    """

    def __init__(self, urdf_path):
        """
        Initialize the Gripper.

        Args:
            urdf_path (str): Path to the URDF file defining the gripper geometry and dynamics.
        """
        self.urdf_path = urdf_path
        # Spawn gripper in a closed position
        self.open = False

        # Initially spawn gripper on spherical surface, radius=1
        self.position = self.gen_random_points(1)
        # Initially spawn gripper at default orientation
        self.orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.gripper_id = p.loadURDF(self.urdf_path, self.position, self.orientation,
                                     globalScaling=1, useFixedBase=False,
                                     flags=p.URDF_USE_SELF_COLLISION |
                                     p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        self.num_joints = p.getNumJoints(self.gripper_id)

        # Create constraint to fix gripper at spawn position
        self.hand_base_constraint = p.createConstraint(
            parentBodyUniqueId=self.gripper_id,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=self.position
        )

        self.roll = 0.0
        self.quat_offset = p.getQuaternionFromEuler([0, 0, 0])

    # -------------------------------
    # Position Initialisation
    # -------------------------------

    @staticmethod
    def gen_random_points(radius):
        """
        Generates a random 3D position on a spherical surface at the specified radius from the origin with additional
        Gaussian noise in the x and y coordinates. Used for initializing gripper positions.

        Args:
            radius (float): The radius of the spherical surface.

        Returns:
            np.ndarray: A 3D position vector [x, y, z] on the spherical surface with noise.
        """
        points = np.array([np.random.uniform(-1, 1),
                           np.random.uniform(-1, 1),
                           np.random.uniform(0.3, 1)])

        # Get the point to lie on a sphere
        norm = radius / ((points[0]**2 + points[1]**2 + points[2]**2)**0.5)
        points *= norm

        # Add Gaussian noise only to x and y
        position_noise = np.random.normal(0, 0.5, 2)
        points[:2] += position_noise

        return points

    # -------------------------------
    # Finger Control
    # -------------------------------

    def preshape(self):
        """
        Configure the gripper to a preshape position before grasping.

        This method should be overridden by subclasses to define gripper-specific
        preshaping behavior. By default, this is a no-op function.
        """
        pass

    def apply_joint_command(self, joint, target, force=2, maxVelocity=1):
        """
        Apply a position control command to a gripper joint.

        Args:
            joint (int): The joint index to control.
            target (float): The target position for the joint in radians.
            force (float, optional): The maximum force to apply. Defaults to 2.
            maxVelocity (float, optional): The maximum velocity of the joint. Defaults to 1.
        """
        p.setJointMotorControl2(self.gripper_id, joint, p.POSITION_CONTROL,
                                targetPosition=target, maxVelocity=maxVelocity, force=force)

    def get_joint_positions(self):
        """
        Get the current positions of all gripper joints.

        Returns:
            list: A list of joint positions in radians for all joints.
        """
        return [p.getJointState(self.gripper_id, i)[0]
                for i in range(self.num_joints)]

    @abstractmethod
    def open_gripper(self):
        """
        Open the gripper fingers. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def close_gripper(self):
        """
        Close the gripper fingers. Must be implemented by subclasses.
        """
        pass

    # -------------------------------
    # Orientation Control
    # -------------------------------

    def get_direction_vector(self, obj):
        """
        Calculate the direction vector from the gripper to an object.

        Args:
            obj: An object with a body_id attribute in the simulation.

        Returns:
            tuple: A tuple containing:
                - direction (np.ndarray): Direction vector from gripper to object.
                - gripper_pos (np.ndarray): Current gripper position.
                - gripper_ori: Current gripper orientation (quaternion).
        """
        obj_pos, _ = p.getBasePositionAndOrientation(obj.body_id)
        gripper_pos, gripper_ori = p.getBasePositionAndOrientation(
            self.gripper_id)

        obj_pos = np.array(obj_pos)
        gripper_pos = np.array(gripper_pos)

        # Direction vector from gripper to object
        direction = obj_pos - gripper_pos
        return direction, gripper_pos, gripper_ori

    def orient_towards_object(self, obj, smoothness=0.05):
        """
        Smoothly rotate the gripper to face an object.

        Computes the desired yaw and pitch angles based on the object's position
        relative to the gripper, then smoothly interpolates towards that orientation
        using spherical linear interpolation (SLERP).

        Args:
            obj: An object with a body_id attribute in the simulation.
            smoothness (float, optional): The interpolation factor (0-1) for SLERP.
                Lower values result in slower rotations. Defaults to 0.05.
        """
        direction, _, gripper_ori = self.get_direction_vector(obj)
        dx, dy, dz = direction

        # Compute desired orientation
        yaw = np.atan2(dy, dx)
        pitch = np.atan2(-dz, np.sqrt(dx**2 + dy**2))

        quat_look_x = p.getQuaternionFromEuler([self.roll, pitch, yaw])
        _, target_quat = p.multiplyTransforms([0, 0, 0], quat_look_x,
                                              [0, 0, 0], self.quat_offset)

        new_quat = tf.quaternion_slerp(
            gripper_ori, target_quat, fraction=smoothness)

        p.changeConstraint(
            self.hand_base_constraint,
            jointChildPivot=self.position,
            jointChildFrameOrientation=new_quat,
            maxForce=50
        )
        # Store the smoothed orientation
        self.orientation = new_quat

    # -------------------------------
    # Movement Control
    # -------------------------------

    def update_gripper_position(self, gripper_pos):
        """
        Update the gripper's position in the simulation while maintaining its orientation.

        Args:
            gripper_pos (array-like): The new position [x, y, z] for the gripper.
        """
        p.changeConstraint(
            self.hand_base_constraint,
            jointChildPivot=gripper_pos,
            jointChildFrameOrientation=self.orientation,
            maxForce=50
        )
        self.position = gripper_pos

    def move_to_object(self, obj, approach_dist, smoothness=0.01):
        """
        Move the gripper towards an object until it reaches the desired approach distance.

        Moves the gripper incrementally along the direction vector towards the object,
        adding noise to the orientation for more realistic grasping behavior.

        Args:
            obj: An object with a body_id attribute in the simulation.
            approach_dist (float): The desired distance between the gripper and object.
            smoothness (float, optional): The movement increment per step. Defaults to 0.01.

        Returns:
            int: 0 if the gripper has reached the approach distance, 1 if still moving.
        """
        direction, gripper_pos, _ = self.get_direction_vector(obj)
        direction_norm = np.linalg.norm(direction)
        # print(direction_norm, approach_dist)

        # If gripper at correct distance to grab object, stop moving
        if direction_norm < approach_dist:
            return 0
        else:
            # Add Gaussian noise to orientation
            ori_noise = np.random.normal(0, 0.15, 3)
            direction += ori_noise

            # Move along direction vector towards object
            gripper_pos += smoothness * direction
            self.update_gripper_position(gripper_pos)
            return 1

    def move_up(self):
        """
        Move the gripper slightly upwards.

        Moves the gripper a small fixed distance in the positive z direction.
        Used after grasping to lift an object.
        """
        gripper_pos, _ = p.getBasePositionAndOrientation(self.gripper_id)
        gripper_pos = np.array(gripper_pos)

        # Direction vector points directly upwards
        direction = np.array([0, 0, 0.0008])

        # Move upwards slightly
        gripper_pos += direction
        self.update_gripper_position(gripper_pos)
