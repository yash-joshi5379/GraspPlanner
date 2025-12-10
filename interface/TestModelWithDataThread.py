import sys
import os
from PyQt6.QtCore import QThread, pyqtSignal
from interface.CaptureStdout import CaptureStdout
from simulation.dataset_generator import generate_dataset
from classification.ClassifierModel import Classifier


class TestModelWithDataThread(QThread):
    """
    Thread for generating test data and testing saved models.

    This class extends QThread to run model testing in a separate thread,
    allowing the GUI to remain responsive while test data is generated and
    a trained model is evaluated on the test data. It captures stdout output
    and emits signals for progress updates and completion.
    """
    finished = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, model_path, gripper, object_type):
        """
        Initialize the TestModelWithDataThread.

        Args:
            model_path (str): The file path to the saved model to test.
            gripper (int): The number of fingers in the gripper used for generating test data.
            object_type (str): The type of object for generating test data ('B' for cube, 'S' for sphere).
        """
        super().__init__()
        self.model_path = model_path
        self.gripper = gripper
        self.object_type = object_type

    def run(self):
        """
        Execute the model testing process in the thread.

        This method generates a test dataset matching the gripper and object type,
        then loads a trained model and evaluates it on the generated test data.
        It captures standard output to emit progress messages through the progress signal
        and emits a finished signal upon completion or error. Standard output is restored
        after the operation completes.
        """
        # Capture stdout to capture print statements
        old_stdout = sys.stdout
        capturing_stdout = CaptureStdout(self.progress)
        sys.stdout = capturing_stdout

        try:
            # Generate test dataset with matching gripper and object
            object_code = "B" if self.object_type == "B" else "S"
            object_name = "cube" if self.object_type == "B" else "sphere"

            self.progress.emit(
                f"This model was trained on {
                    self.gripper} fingers and the {object_name}\n")
            self.progress.emit(
                f"Generating 10 test trials with matching configuration...\n")
            generate_dataset(self.gripper, self.object_type, num_trials=10)

            # Find the generated CSV file
            test_dataset = os.path.join(
                "datasets", f"NumFingers_{
                    self.gripper}_Object_{object_code}_NumTrials_10.csv")

            self.progress.emit(f"\nLoading model from {self.model_path}...\n")
            classifier = Classifier.load_model(self.model_path)

            self.progress.emit(f"Testing model on {test_dataset}...\n")
            classifier.test_on_new_data(test_dataset)

            self.finished.emit("Model testing completed")
        except Exception:
            self.finished.emit("Model testing failed")
        sys.stdout = old_stdout
