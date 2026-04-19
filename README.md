# Plant Disease Severity Grading

ResNet18 + CORN ordinal loss. Classifies plant disease severity into 4 levels: Healthy, Mild, Moderate, Severe. Trained on PlantVillage dataset (20,638 images).

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 99.23% |
| MAE | 0.0126 |
| QWK | 0.99 |
| Healthy recall | 100% |

16 misclassifications out of 2065 test images. All errors between adjacent severity levels.

## Severity Mapping

| Level | Disease Classes |
|-------|-----------------|
| 0 Healthy | Healthy leaves |
| 1 Mild | Early Blight, Bacterial Spot, Septoria Leaf Spot, Pepper Bacterial Spot |
| 2 Moderate | Leaf Mold, Spider Mites, Target Spot |
| 3 Severe | Late Blight, YellowLeaf Curl Virus, Mosaic Virus, Potato Late Blight |

## Model

- Backbone: ResNet18 (ImageNet pretrained)
- Loss: CORN ordinal
- Input: 224x224
- Output: 4 severity levels with ordering

## Training

- Optimizer: Adam, lr=1e-4
- Batch size: 16
- Epochs: 10
- Hardware: CPU only

## Files

- plant_severity_model.py - Training script
- evaluate_results.py - Metrics evaluation
- plots_generation.py - Generate figures
- final_survey_paper.tex - LaTeX paper

## Run

python plant_severity_model.py
python evaluate_results.py
python plots_generation.py

## Dataset

https://www.kaggle.com/datasets/arjuntejaswi/plant-village
