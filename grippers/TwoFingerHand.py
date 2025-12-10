from grippers.baseGripper import Gripper


class TwoFingerHand(Gripper):
    """
    A two-finger gripper implementation inheriting from the base Gripper class.

    This class represents a simple two-finger parallel gripper (PR2 gripper) with
    independent control over each finger. The gripper can open and close its fingers
    to grasp objects.
    """

    def __init__(self):
        """
        Initialize the TwoFingerHand gripper.

        Loads the PR2 gripper URDF model for simulation.
        """
        super().__init__(urdf_path="pr2_gripper.urdf")

    # -------------------------------
    # Finger Control
    # -------------------------------

    def open_gripper(self):
        """
        Open the gripper to be ready to grab an object.

        Moves both fingers (joints 0 and 2) to 0.56 radians with moderate force (4).
        This opens the gripper jaws to allow object placement between the fingers.
        """
        for j in [0, 2]:
            self.apply_joint_command(joint=j, target=0.56, force=4)

    def close_gripper(self):
        """
        Close the gripper to grab an object.

        Moves both fingers (joints 0 and 2) to 0.0 radians with moderate force (4).
        This closes the gripper jaws to firmly grip the object between the fingers.
        """
        for j in [0, 2]:
            self.apply_joint_command(joint=j, target=0.0, force=4)
