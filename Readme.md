

# MobileFCMViTv3: Breast Cancer Detection Pipeline

A modular, research-grade deep learning pipeline for breast cancer detection from ultrasound images. Combines Fuzzy C-Means clustering, MobileViTv3 neural networks, and mobile deployment (TensorFlow Lite, Flutter).

---

## 1. Project Overview

- Modular, extensible codebase for academic and clinical use
- Dataset ingestion, preprocessing, clustering, model training, evaluation, export, benchmarking, mobile deployment
- Config-driven workflows for reproducibility

---

## 2. Setup Instructions

### Python Environment
1. Install Python 3.11 (recommended)
2. Install PyTorch (with CUDA if available):
   - [PyTorch installation guide](https://pytorch.org/get-started/locally/)
3. Clone the repository:
   ```bash
   git clone https://github.com/Gabriel-mdv/early-breast-cancer-detection.git
   cd early-breast-cancer-detection
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Optional: CUDA Setup
- For GPU acceleration, install CUDA toolkit and verify PyTorch CUDA support.

### Configuration
- Edit YAML files in `config/`:
  - `dataset_config.yaml`: dataset paths, splits
  - `training_config.yaml`: epochs, batch size, optimizer, scheduler
  - `model_config.yaml`: model architecture, parameters

---

## 3. Dataset Preparation

- Place ultrasound images in the directory specified in `config/dataset_config.yaml`
- Supported formats: DICOM, PNG, JPEG
- Preprocessing pipeline handles denoising, normalization, resizing, enhancement

---

## 4. Training

### Start Training
```bash
python scripts/train.py --config_dir config/
```
- Supports resume, checkpointing, experiment tracking (Weights & Biases, MLflow, Neptune)
- Logs metrics, losses, and model checkpoints to `results/`

### Custom Training Workflows
- Edit or add scripts in `training/custom_workflows.py` for advanced experiments

---

## 5. Testing & Evaluation

### Run Evaluation
```bash
python scripts/evaluate.py --config_dir config/
```
- Outputs metrics: accuracy, precision, recall, F1, sensitivity, specificity, ROC-AUC, confusion matrix, Grad-CAM
- Results saved in `results/metrics.json`, `results/predictions.json`

### Unit Tests
```bash
python -m unittest discover tests
```
- Tests for dataset, preprocessing, clustering, model modules

---

## 6. Model Export & Quantization

### Export Model
```bash
python scripts/export_model.py \
  --input_shape 1 1 224 224 \
  --onnx_path export/model.onnx \
  --tf_path export/model_tf \
  --tflite_path export/model.tflite \
  --quantized_path export/model_quant.tflite
```
- Converts PyTorch → ONNX → TensorFlow → TensorFlow Lite
- Supports post-training quantization for mobile deployment

---

## 7. Benchmarking

### Run Benchmark
```bash
python scripts/benchmark_model.py --model_path export/model.onnx --input_shape 1 1 224 224
```
- Measures model size, latency, memory usage
- Results saved in `results/`

---

## 8. Mobile Deployment

### Flutter App
- Located in `deployment/flutter_app/`
- Features:
  - Capture ultrasound images
  - Preprocess images (denoise, normalize, resize)
  - Load TensorFlow Lite model
  - Run inference
  - Display prediction probabilities, Grad-CAM heatmap
  - Upload reports, patient info, notifications
- Supports Android and iOS

### Setup Flutter App
1. Install Flutter SDK: [Flutter installation guide](https://docs.flutter.dev/get-started/install)
2. Open `deployment/flutter_app/` in VS Code or Android Studio
3. Run:
   ```bash
   flutter pub get
   flutter run
   ```
4. Place exported `.tflite` model in app assets

---

## 9. Integration & Experiment Tracking

- External API adapters: `datasets/external_api_adapter.py`
- Hospital/cloud integration: `integration/hospital_cloud_adapter.py`
- Experiment tracking: `experiments/experiment_tracking.py`
- Configure tracking in `training_config.yaml`

---

## 10. Troubleshooting & Tips

- Check Python and CUDA versions if errors occur
- Verify dataset paths in config files
- For mobile deployment, ensure `.tflite` model matches input shape
- For experiment tracking, set API keys in environment variables
- See module documentation in `documentations/`

---

## 11. Directory Structure

- `config/`: YAML configs for dataset, training, model
- `datasets/`: dataset loading, transforms, external API adapters
- `preprocessing/`: denoising, normalization, resizing, enhancement
- `clustering/`: Fuzzy C-Means, cluster/membership map generators
- `models/`: model architectures, layers, blocks
- `training/`: trainer, loops, optimizer, scheduler, losses, callbacks
- `evaluation/`: metrics, confusion matrix, ROC, Grad-CAM
- `experiments/`: baseline models, ablation studies, experiment tracking
- `export/`: ONNX, TensorFlow, TFLite, quantization
- `benchmark/`: latency, memory, size
- `integration/`: hospital/cloud adapters
- `deployment/flutter_app/`: mobile app screens, services, widgets
- `scripts/`: CLI scripts for train, evaluate, export, benchmark
- `tests/`: unit tests
- `results/`: metrics, predictions, benchmarks
- `documentations/`: architecture, experiment setup, training docs

---

## 12. References & Documentation

- See `documentations/` for architecture, experiment setup, training details
- For code details, see docstrings and module-level documentation
- For Flutter app, see README in `deployment/flutter_app/`

---

## 13. Contact & Support

- For issues, open a GitHub issue or contact the repository owner

---

## 14. Citation

If you use MobileFCMViTv3 in academic work, please cite the repository.
- Supports PyTorch → ONNX → TensorFlow → TensorFlow Lite, with quantization.

## Benchmarking

- Measure model size, latency, memory usage:
  ```bash
  python scripts/benchmark_model.py --model_path export/model.onnx --input_shape 1 1 224 224
  ```

## Mobile Deployment

- Flutter app in `deployment/flutter_app/`:
  - Capture ultrasound images
  - Preprocess images
  - Load TensorFlow Lite model
  - Run inference
  - Display prediction probabilities and Grad-CAM heatmap
  - Supports Android and iOS

## Testing

- Unit tests for dataset, preprocessing, clustering, and model:
  ```bash
  python -m unittest discover tests
  ```

## Directory Structure

- Modular codebase: datasets, preprocessing, clustering, models, training, evaluation, experiments, export, benchmark, deployment, scripts, tests
- Configuration files: `config/dataset_config.yaml`, `config/training_config.yaml`, `config/model_config.yaml`
- Flutter app: `deployment/flutter_app/`

---

For full details, see code and documentation in each module.
