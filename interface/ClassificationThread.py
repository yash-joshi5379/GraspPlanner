import sys
from PyQt6.QtCore import QThread, pyqtSignal
from interface.CaptureStdout import CaptureStdout
from classification.dataset_classifier import classify_dataset


class ClassificationThread(QThread):
    """
    Thread for running classification to prevent GUI freezing.

    This class extends QThread to run dataset classification in a separate thread,
    allowing the GUI to remain responsive while classification is performed.
    It captures stdout output and emits signals for progress updates and completion.
    """
    finished = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, dataset, model_choice):
        """
        Initialize the ClassificationThread.

        Args:
            dataset: The dataset to be classified.
            model_choice: The classification model to use for classification.
        """
        super().__init__()
        self.dataset = dataset
        self.model_choice = model_choice

    def run(self):
        """
        Execute the classification process in the thread.

        This method captures standard output to redirect print statements from the
        classifier, emitting them through the progress signal. The method calls the
        classify_dataset function and emits a finished signal upon completion or error.
        Standard output is restored after the operation completes.
        """
        # Capture stdout to capture print statements from Classifier
        old_stdout = sys.stdout
        capturing_stdout = CaptureStdout(self.progress)
        sys.stdout = capturing_stdout

        try:
            classify_dataset(self.model_choice, self.dataset)
            self.finished.emit("Classification completed")
        except Exception:
            self.finished.emit("Classification interrupted")
        sys.stdout = old_stdout
