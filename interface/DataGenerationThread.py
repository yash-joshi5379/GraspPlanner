import sys
from PyQt6.QtCore import QThread, pyqtSignal
from interface.CaptureStdout import CaptureStdout
from simulation.dataset_generator import generate_dataset


class DataGenerationThread(QThread):
    """
    Thread for running data generation to prevent GUI freezing.

    This class extends QThread to run dataset generation in a separate thread,
    allowing the GUI to remain responsive while data generation is performed.
    It captures stdout output and emits signals for progress updates and completion.
    """
    finished = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, gripper, object_type, num_trials):
        """
        Initialize the DataGenerationThread.

        Args:
            gripper: The gripper type to use for generating grasps.
            object_type: The type of object to generate grasps for.
            num_trials: The number of trials/grasps to generate.
        """
        super().__init__()
        self.gripper = gripper
        self.object_type = object_type
        self.num_trials = num_trials

    def run(self):
        """
        Execute the data generation process in the thread.

        This method captures standard output to redirect print statements from the
        dataset generator, emitting them through the progress signal. The method calls
        the generate_dataset function and emits a finished signal upon completion or
        error. Standard output is restored after the operation completes.
        """
        # Capture stdout to capture print statements from dataset_generator.py
        old_stdout = sys.stdout
        capturing_stdout = CaptureStdout(self.progress)
        sys.stdout = capturing_stdout

        try:
            generate_dataset(self.gripper, self.object_type, self.num_trials)
            self.finished.emit("Data generation completed")
        except Exception:
            self.finished.emit("Data generation interrupted")
        sys.stdout = old_stdout
