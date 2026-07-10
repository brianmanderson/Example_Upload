# Example_Upload

Despite the name, this is a research training pipeline for 3D deep-learning segmentation
models on liver disease ablation imaging (liver and ablation region masks). It sweeps
hyperparameters and learning rates, trains DenseNet121-hybrid / U-Net style models in
TensorFlow (parallel TF1 and TF2 script variants), and evaluates the results.

Dormant research code, last touched 2021; not maintained. It is **not runnable standalone**:
it imports local packages (`Base_Deeplearning_Code`, `Deep_Learning`) that are not included
in this repo, and data paths are hard-coded to internal drives.

## Layout

- `Main.py`, `Main_TF2.py`, `Main_TF2_HNet.py` — driver scripts; boolean flags step through
  the workflow: find best learning rate, plot LR curves, train ~200 epochs, tabulate results.
- `Optimization/` — LR range-finder (`Find_Best_LR*.py`) and plotting of LR/optimization
  results, logged via TensorBoard HParams.
- `Return_Train_Validation_Generators*.py` — TFRecord-based train/validation data generators
  and hyperparameter grids (layers, filters, max filters).
- `Utils/` — model builders (pretrained DenseNet121 with frozen encoder / trainable
  upsampling path), generators, path helpers, Excel metric export.
- `Main_Evaluation*.py`, `Model_Test.py` — model evaluation.
- `Direct_Testing_From_Raystation/` — scripts to export an examination and Liver/Ablation
  ROIs from RayStation and convert the MHD exports to NIfTI with SimpleITK, for testing
  models directly against treatment-planning-system data.

## Tech stack

Python, TensorFlow 1.x/2.x, TensorBoard HParams, SimpleITK, pandas/openpyxl.
