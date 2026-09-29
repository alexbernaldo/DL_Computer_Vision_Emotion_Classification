# DL_Computer_Vision_Emotion_Classification

Facial emotion recognition across **5 classes (Angry, Fear, Happy, Sad, Surprise)** on the
Human Face Emotions Kaggle dataset — comparing a **CNN from scratch** with **MobileNetV2
transfer learning** under a shared test protocol.

## Dataset

**Human Face Emotions** (Kaggle) — 59,099 images (59,077 usable), grayscale 96×96,
stratified 75/12.5/12.5 split (shared 7,385-sample test set for both exercises).
Imbalanced: Happy 18,437 vs Surprise 8,225.

## Approach

- **Exercise 1 — CNN from scratch:** Hyperband search (Keras Tuner), 4 dataset strategies
  (class weights, undersampling, oversampling+augmentation, RGB variant), EarlyStopping +
  ReduceLROnPlateau.
- **Exercise 2 — Transfer learning:** MobileNetV2 (ImageNet) two-phase fine-tuning
  (head frozen → last 30 layers), same 4 dataset strategies, same test set.

## Results

| Exercise | Strategy | Test accuracy | Macro F1 |
|---|---|---|---|
| E1 CNN (521k params) | Oversampling | **0.8422** | 0.832 |
| E2 MobileNetV2 (2.6M params) | Class weights | **0.8942** | 0.888 |

Class balancing was the most influential design decision; undersampling consistently worst.
Fear is the hardest class (often confused with Sad).

## Reproduce

Download the Human Face Emotions Kaggle dataset, open `PW2_E1_GROUP2.ipynb` and
`PW2_E2_GROUP2.ipynb` in Colab (T4 GPU), run top to bottom.
