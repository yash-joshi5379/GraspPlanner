import os
import glob
from interface.DataGenerationThread import DataGenerationThread
from interface.ClassificationThread import ClassificationThread
from interface.TestModelWithDataThread import TestModelWithDataThread
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QGroupBox, QTextEdit, QButtonGroup)
from PyQt6.QtCore import Qt


class GraspPlannerGUI(QMainWindow):
    """
    Main GUI window for the Grasp Planner application. It manages three
    main modes: data generation, classification, and model testing.

    Attributes:
        initial_height (int): Initial height of the window in pixels.
        current_mode (str): Currently selected mode ("generate", "classify", or "test_model").
        options_widget (QWidget): Widget containing the current mode's options.
        options_layout (QVBoxLayout): Layout for the options container.
        simulation_viewer_group (QGroupBox): Group box containing the simulation viewer.
        simulation_label (QLabel): Label displaying simulation output.
        output_text (QTextEdit): Text area displaying application output and progress.
    """

    def __init__(self):
        """
        Initialize the Grasp Planner GUI window.

        Sets up the main window, creates UI components including mode selection
        buttons, options container, simulation viewer, and output area.
        """
        super().__init__()
        self.setWindowTitle("Grasp Planner")
        self.setGeometry(60, 60, 600, 400)
        self.initial_height = 400

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Title
        title = QLabel(
            "COMP0213 Grasp Planner \n Group 06 - Yash Joshi & Derek Chung")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold; margin: 20px;")
        main_layout.addWidget(title)

        # Mode selection buttons
        mode_group = QGroupBox("Select Mode")
        mode_layout = QHBoxLayout()

        self.generate_btn = QPushButton("Generate Data")
        self.generate_btn.setMinimumHeight(50)
        self.generate_btn.clicked.connect(self.show_generate_options)

        self.classify_btn = QPushButton("Classify Data")
        self.classify_btn.setMinimumHeight(50)
        self.classify_btn.clicked.connect(self.show_classify_options)

        self.test_model_btn = QPushButton("Test Saved Model")
        self.test_model_btn.setMinimumHeight(50)
        self.test_model_btn.clicked.connect(self.show_test_model_options)

        mode_layout.addWidget(self.generate_btn)
        mode_layout.addWidget(self.classify_btn)
        mode_layout.addWidget(self.test_model_btn)
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # Options container
        self.options_container = QWidget()
        self.options_layout = QVBoxLayout()
        self.options_container.setLayout(self.options_layout)
        main_layout.addWidget(self.options_container)

        # Simulation viewer (initially hidden)
        self.simulation_viewer_group = QGroupBox("Simulation View")
        simulation_viewer_layout = QVBoxLayout()
        self.simulation_label = QLabel()
        self.simulation_label.setMinimumSize(640, 480)
        self.simulation_label.setMaximumSize(640, 480)
        self.simulation_label.setStyleSheet(
            "background-color: black; border: 1px solid gray;")
        self.simulation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.simulation_label.setText("Simulation will appear here")
        self.simulation_label.hide()  # Initially hidden
        simulation_viewer_layout.addWidget(self.simulation_label)
        self.simulation_viewer_group.setLayout(simulation_viewer_layout)
        self.simulation_viewer_group.hide()  # Initially hidden
        main_layout.addWidget(self.simulation_viewer_group)

        # Output area
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setMinimumHeight(100)
        self.output_text.setMaximumHeight(150)
        output_layout.addWidget(self.output_text)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)

        # Initially hide options
        self.current_mode = None
        self.options_widget = None

    def update_button_style(self, selected_btn, other_btns):
        """
        Update button styles to visually indicate which button is selected.
        Selected = blue, Not selected = grey

        Args:
            selected_btn (QPushButton): The button that is currently selected/checked.
            other_btns (list[QPushButton]): List of other buttons in the same group.
        """
        # Common stylesheet for all option buttons
        button_style = """
            QPushButton {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 5px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
            QPushButton:checked {
                background-color: #007bff;
                color: white;
                font-weight: bold;
                border: 2px solid #0056b3;
            }
            QPushButton:disabled {
                background-color: #e9ecef;
                color: #6c757d;
                border: 2px solid #ced4da;
            }
        """
        # Apply to all buttons in the group
        selected_btn.setStyleSheet(button_style)
        for btn in other_btns:
            btn.setStyleSheet(button_style)

    def clear_options(self):
        """
        Clear the options container by removing the current options widget.

        This method is called when switching between modes to ensure only
        one set of options is displayed at a time.
        """
        if self.options_widget:
            self.options_widget.setParent(None)
        self.options_widget = None

    def show_generate_options(self):
        """
        Display options for data generation mode.

        It also resizes the window to accommodate the options panel.
        """
        self.clear_options()
        self.current_mode = "generate"
        # Resize window to smaller height when options are shown
        self.resize(self.width(), 250)

        options_widget = QWidget()
        options_layout = QVBoxLayout()
        options_widget.setLayout(options_layout)

        # Gripper selection
        gripper_group = QGroupBox("Gripper Selection")
        gripper_layout = QHBoxLayout()
        self.gripper_button_group = QButtonGroup()
        self.gripper_2_btn = QPushButton("2 Fingers")
        self.gripper_2_btn.setCheckable(True)
        self.gripper_2_btn.setMinimumHeight(40)
        self.gripper_3_btn = QPushButton("3 Fingers")
        self.gripper_3_btn.setCheckable(True)
        self.gripper_3_btn.setMinimumHeight(40)
        self.gripper_button_group.addButton(self.gripper_2_btn, 2)
        self.gripper_button_group.addButton(self.gripper_3_btn, 3)
        self.gripper_2_btn.setChecked(True)  # Default selection
        self.update_button_style(self.gripper_2_btn, [self.gripper_3_btn])
        gripper_layout.addWidget(self.gripper_2_btn)
        gripper_layout.addWidget(self.gripper_3_btn)
        gripper_group.setLayout(gripper_layout)
        options_layout.addWidget(gripper_group)

        # Object selection
        object_group = QGroupBox("Object Selection")
        object_layout = QHBoxLayout()
        self.object_button_group = QButtonGroup()
        self.object_cube_btn = QPushButton("Cube")
        self.object_cube_btn.setCheckable(True)
        self.object_cube_btn.setMinimumHeight(40)
        self.object_sphere_btn = QPushButton("Sphere")
        self.object_sphere_btn.setCheckable(True)
        self.object_sphere_btn.setMinimumHeight(40)
        self.object_button_group.addButton(self.object_cube_btn, 0)
        self.object_button_group.addButton(self.object_sphere_btn, 1)
        self.object_cube_btn.setChecked(True)  # Default selection
        self.update_button_style(
            self.object_cube_btn, [
                self.object_sphere_btn])
        object_layout.addWidget(self.object_cube_btn)
        object_layout.addWidget(self.object_sphere_btn)
        object_group.setLayout(object_layout)
        options_layout.addWidget(object_group)

        # Number of trials
        trials_group = QGroupBox("Number of Trials")
        trials_layout = QHBoxLayout()
        self.trials_button_group = QButtonGroup()
        self.trials_100_btn = QPushButton("100")
        self.trials_100_btn.setCheckable(True)
        self.trials_100_btn.setMinimumHeight(40)
        self.trials_150_btn = QPushButton("150")
        self.trials_150_btn.setCheckable(True)
        self.trials_150_btn.setMinimumHeight(40)
        self.trials_200_btn = QPushButton("200")
        self.trials_200_btn.setCheckable(True)
        self.trials_200_btn.setMinimumHeight(40)
        self.trials_button_group.addButton(self.trials_100_btn, 100)
        self.trials_button_group.addButton(self.trials_150_btn, 150)
        self.trials_button_group.addButton(self.trials_200_btn, 200)
        self.trials_200_btn.setChecked(True)  # Default selection
        self.update_button_style(
            self.trials_200_btn, [
                self.trials_100_btn, self.trials_150_btn])
        trials_layout.addWidget(self.trials_100_btn)
        trials_layout.addWidget(self.trials_150_btn)
        trials_layout.addWidget(self.trials_200_btn)
        trials_group.setLayout(trials_layout)
        options_layout.addWidget(trials_group)

        # Generate button
        self.run_generate_btn = QPushButton("Start Data Generation")
        self.run_generate_btn.setMinimumHeight(40)
        self.run_generate_btn.clicked.connect(self.start_data_generation)
        options_layout.addWidget(self.run_generate_btn)

        self.options_widget = options_widget
        self.options_layout.addWidget(options_widget)

    def show_classify_options(self):
        """
        Display options for data classification mode.

        It also hides the simulation viewer and resizes the window.
        """
        self.clear_options()
        self.current_mode = "classify"
        # Hide simulation viewer for classification
        self.simulation_viewer_group.hide()
        self.simulation_label.hide()
        # Resize window to smaller height when options are shown
        self.resize(self.width(), 250)

        options_widget = QWidget()
        options_layout = QVBoxLayout()
        options_widget.setLayout(options_layout)

        # Dataset selection
        dataset_group = QGroupBox("Dataset Selection")
        dataset_layout = QVBoxLayout()
        self.dataset_combo = QComboBox()

        # Find all CSV files in the datasets directory
        csv_files = glob.glob(os.path.join("datasets", "*.csv"))
        csv_files = [f for f in csv_files if not f.startswith(
            ".") and os.path.isfile(f)]
        self.dataset_combo.addItems(sorted(csv_files))

        dataset_layout.addWidget(self.dataset_combo)
        dataset_group.setLayout(dataset_layout)
        options_layout.addWidget(dataset_group)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QHBoxLayout()
        self.model_button_group = QButtonGroup()
        self.model_rf_btn = QPushButton("Random Forest")
        self.model_rf_btn.setCheckable(True)
        self.model_rf_btn.setMinimumHeight(40)
        self.model_lr_btn = QPushButton("Logistic Regression")
        self.model_lr_btn.setCheckable(True)
        self.model_lr_btn.setMinimumHeight(40)
        self.model_nn_btn = QPushButton("Neural Network")
        self.model_nn_btn.setCheckable(True)
        self.model_nn_btn.setMinimumHeight(40)
        self.model_svm_btn = QPushButton("SVM")
        self.model_svm_btn.setCheckable(True)
        self.model_svm_btn.setMinimumHeight(40)
        self.model_button_group.addButton(self.model_rf_btn, 0)  # R
        self.model_button_group.addButton(self.model_lr_btn, 1)  # L
        self.model_button_group.addButton(self.model_nn_btn, 2)  # N
        self.model_button_group.addButton(self.model_svm_btn, 3)  # S
        self.model_rf_btn.setChecked(True)  # Default selection
        self.update_button_style(
            self.model_rf_btn, [
                self.model_lr_btn, self.model_nn_btn, self.model_svm_btn])
        model_layout.addWidget(self.model_rf_btn)
        model_layout.addWidget(self.model_lr_btn)
        model_layout.addWidget(self.model_nn_btn)
        model_layout.addWidget(self.model_svm_btn)
        model_group.setLayout(model_layout)
        options_layout.addWidget(model_group)

        # Classify button
        self.run_classify_btn = QPushButton("Start Classification")
        self.run_classify_btn.setMinimumHeight(40)
        self.run_classify_btn.clicked.connect(self.start_classification)
        options_layout.addWidget(self.run_classify_btn)

        self.options_widget = options_widget
        self.options_layout.addWidget(options_widget)

    def show_test_model_options(self):
        """
        Display options for testing saved models mode.

        It also hides the simulation viewer and resizes the window.
        """
        self.clear_options()
        self.current_mode = "test_model"
        # Hide simulation viewer
        self.simulation_viewer_group.hide()
        self.simulation_label.hide()
        # Resize window
        self.resize(self.width(), 300)

        options_widget = QWidget()
        options_layout = QVBoxLayout()
        options_widget.setLayout(options_layout)

        # Model selection
        model_group = QGroupBox("Select Saved Model")
        model_layout = QVBoxLayout()
        self.test_model_combo = QComboBox()

        # Find all model files in the models directory
        model_files = glob.glob("models/*.pkl")
        if not model_files:
            model_files = ["No models found - train a model first"]
        self.test_model_combo.addItems(sorted(model_files))

        model_layout.addWidget(self.test_model_combo)
        model_group.setLayout(model_layout)
        options_layout.addWidget(model_group)

        # Test button
        self.run_test_btn = QPushButton("Generate 10 trials and test model")
        self.run_test_btn.setMinimumHeight(40)
        self.run_test_btn.clicked.connect(self.start_test_model)
        options_layout.addWidget(self.run_test_btn)

        self.options_widget = options_widget
        self.options_layout.addWidget(options_widget)

    def start_data_generation(self):
        """
        Start the data generation process in a separate thread.

        It disables UI controls during execution to prevent multiple
        simultaneous operations.
        """
        # Get selected values from buttons
        selected_gripper_id = self.gripper_button_group.checkedId()
        gripper = selected_gripper_id

        selected_object_id = self.object_button_group.checkedId()
        object_type = "B" if selected_object_id == 0 else "S"  # 0 = Cube, 1 = Sphere

        num_trials = self.trials_button_group.checkedId()

        # Clear output and disable start button and options during execution
        self.output_text.clear()
        self.run_generate_btn.setEnabled(False)
        self.gripper_2_btn.setEnabled(False)
        self.gripper_3_btn.setEnabled(False)
        self.object_cube_btn.setEnabled(False)
        self.object_sphere_btn.setEnabled(False)
        self.trials_100_btn.setEnabled(False)
        self.trials_150_btn.setEnabled(False)
        self.trials_200_btn.setEnabled(False)

        # Create and start thread
        self.gen_thread = DataGenerationThread(
            gripper, object_type, num_trials)
        self.gen_thread.progress.connect(self.update_output)
        self.gen_thread.finished.connect(self.on_generation_finished)
        self.gen_thread.start()

    def start_classification(self):
        """
        Start the classification process in a separate thread.

        It disables UI controls during execution to prevent multiple
        simultaneous operations.
        """
        # Get selected values
        dataset = self.dataset_combo.currentText()

        # Get selected model from buttons
        selected_model_id = self.model_button_group.checkedId()
        model_map = {0: "R", 1: "L", 2: "N", 3: "S"}
        model_choice = model_map[selected_model_id]

        # Clear output and disable button and options during execution
        self.output_text.clear()
        self.run_classify_btn.setEnabled(False)
        self.dataset_combo.setEnabled(False)
        self.model_rf_btn.setEnabled(False)
        self.model_lr_btn.setEnabled(False)
        self.model_nn_btn.setEnabled(False)
        self.model_svm_btn.setEnabled(False)

        # Create and start thread
        self.classify_thread = ClassificationThread(dataset, model_choice)
        self.classify_thread.progress.connect(self.update_output)
        self.classify_thread.finished.connect(self.on_classification_finished)
        self.classify_thread.start()

    def start_test_model(self):
        """
        Start testing a saved model on 10 new randomly generated trials.

        It disables UI controls during execution to prevent multiple
        simultaneous operations.
        """
        # Get selected model
        model_path = self.test_model_combo.currentText()

        # Clear output and disable button during execution
        self.output_text.clear()
        self.run_test_btn.setEnabled(False)
        self.test_model_combo.setEnabled(False)

        # Parse model filename to extract gripper and object
        # E.g. NumFingers_3_Object_S_NumTrials_200_model_N.pkl
        parts = model_path.split('_')
        gripper = int(parts[1])
        object_code = parts[3]

        # Create a thread to generate test data and then test the model
        self.test_thread = TestModelWithDataThread(
            model_path, gripper, object_code)
        self.test_thread.progress.connect(self.update_output)
        self.test_thread.finished.connect(self.on_test_finished)
        self.test_thread.start()

    def update_output(self, message):
        """
        Update the output text area with a new message.

        Appends the message to the output text area and automatically scrolls
        to the bottom to show the most recent output.

        Args:
            message (str): The message text to append to the output area.
        """
        self.output_text.append(message)
        # Auto-scroll to bottom
        self.output_text.verticalScrollBar().setValue(
            self.output_text.verticalScrollBar().maximum()
        )

    def on_generation_finished(self, message):
        """
        Handle completion of the data generation process.

        Re-enables all UI controls that were disabled during generation and displays the completion message.

        Args:
            message (str): Completion message from the generation thread.
        """
        self.update_output(message)
        self.run_generate_btn.setEnabled(True)
        self.gripper_2_btn.setEnabled(True)
        self.gripper_3_btn.setEnabled(True)
        self.object_cube_btn.setEnabled(True)
        self.object_sphere_btn.setEnabled(True)
        self.trials_100_btn.setEnabled(True)
        self.trials_150_btn.setEnabled(True)
        self.trials_200_btn.setEnabled(True)

    def on_classification_finished(self, message):
        """
        Handle completion of the classification process.

        Re-enables all UI controls that were disabled during classification and displays the completion message.

        Args:
            message (str): Completion message from the classification thread.
        """
        self.update_output(message)
        self.run_classify_btn.setEnabled(True)
        self.dataset_combo.setEnabled(True)
        self.model_rf_btn.setEnabled(True)
        self.model_lr_btn.setEnabled(True)
        self.model_nn_btn.setEnabled(True)
        self.model_svm_btn.setEnabled(True)

    def on_test_finished(self, message):
        """
        Handle completion of the model testing process.

        Re-enables all UI controls that were disabled during testing and displays the completion message.

        Args:
            message (str): Completion message from the test thread.
        """
        self.update_output(message)
        self.run_test_btn.setEnabled(True)
        self.test_model_combo.setEnabled(True)
