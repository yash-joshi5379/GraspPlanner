# GraspPlanner

This repository contains all the Python code and supporting files used for our grasp planning project, including dataset generation, data classification, and model testing, all integrated into a GUI for user convenience. The project is split into separate folders to provide simple navigation through all of our files and classes.

> ✅ **Plug-and-play ready** — just download the full repository, install the requirements and you're ready!
---

## 🧭 Project Structure
| Folder                 | Purpose                                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **classification**     | Contains all model-training and classification logic, including classifiers, model selection, and evaluation code. |
| **confusion_matrices** | Stores generated confusion matrix images for evaluating model performance.                                         |
| **datasets**           | Holds generated or collected grasp-trial datasets used for training and testing models.                            |
| **figures**            | Contains plots, charts, or other figures produced during analysis or evaluation.                                   |
| **grippers**           | Includes gripper configuration files or classes representing different robotic gripper types.                      |
| **interface**          | Houses the PyQt6 GUI code, including windows, dialogs, and threading utilities for the app.                        |
| **models**             | Stores saved machine-learning models (e.g., `.pkl` files, neural network weights) generated during classification. |
| **objects**            | Contains object definition files, meshes, or metadata used in simulation grasp trials.                             |
| **simulation**         | Implements the simulation environment, physics setup, and grasp trial execution logic.                             |
| **urdf_files**         | URDF robot/gripper description files used by the simulator for accurate physical modelling.                        |

---

## Complete Project Video
https://www.youtube.com/watch?v=ChXXNRX1s6s

## 🛠️ Setup Instructions

### 1. Clone or Download the Repository 
Clone this repository or download it to your local computer.

```bash
git clone https://github.com/yash-joshi5379/GraspPlanner
```

### 2. Create a virtual environment (venv) with Python 3.13.2

```bash
python -m venv oop_grasp_planner
```

### 3. Use the virtual environment for the selected interpreter

```bash
oop_grasp_planner\Scripts\activate
```

### 4. You should see the venv in ther terminal

```bash
(oop_grasp_planner) C:\Users\...\GraspPlanner-main>
```

### 5. Install all Required Libraries
Once in the venv, run the following instruction in the command line:

```bash
pip install -r requirements.txt
```
### 6. Run the file named 'main.py'
You should see a GUI appear, feel free to explore from there!

To run dataset generation, select the 'Generate Data' button, customise the gripper/object/number of trials and press the 'Start Data Generation' button.

To run classifier training and testing, select the 'Classify Data' button, choose a dataset from the saved `.csv` files and a machine learning model, and press the 'Start Classification' button.

To run model testing on new, unseen data, select the 'Test Saved Model' button, choose a model from the saved `.pkl` files, and press the 'Generate 10 trials and test model' button. 

To view plots of sampled poses, go to the 'figures' folder and choose the desired gripper/object/number of trials combination

To view confusion matrices, go to the 'confusion_matrices' folder and choose the desired gripper/object/model combination

![alt text](https://github.com/yash-joshi5379/GraspPlanner/blob/main/GUI_image%20(1).png "GUI_image")
