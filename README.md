# Automated Dog Breed Identification from Shelter Photos

An image classifier that predicts a dog's breed from a photo, built to give animal shelters
a consistent, evidence-informed check on visual breed labeling at intake -- a process known
to be inconsistent enough that it affects real adoption, housing, and bite-report outcomes.

## Problem

Shelter staff assign breed labels visually, and research (Voith et al., 2009) shows only
25% of visual breed calls match DNA-verified breed composition. Inaccurate labels carry
real consequences: adoption outcomes, breed-restricted housing/insurance eligibility, and
skewed bite-report demographics. This project tests whether a trained image classifier can
offer a more consistent second opinion than visual guesswork alone.

## Approach

- **Model:** MobileNetV2 (ImageNet-pretrained, via Hugging Face), fine-tuned for 115-class
  dog breed classification.
- **Training data:** [Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
  -- 20,486 images, split 70/15/15 (train/val/test), stratified by breed.
- **Real-world evaluation set:** [PetFinder.my Adoption Prediction dataset](https://www.kaggle.com/c/petfinder-adoption-prediction)
  -- 1,604 real, non-staged shelter photos with breed labels harmonized onto Stanford Dogs'
  vocabulary, kept completely separate from training.
- **Model comparison:** three fine-tuning depths were compared (frozen backbone / last 3
  blocks unfrozen / fully unfrozen) to isolate how much backbone adaptation actually helps.

## Results

| Model | Stanford top-1 | Stanford top-5 | PetFinder top-1 |
|---|---|---|---|
| Baseline (stratified) | 1.1% | -- | 1.1% |
| A: Frozen | 23.3% | 56.0% | 14.0% |
| B: Last 3 blocks unfrozen | 32.8% | 69.6% | 16.9% |
| **C: Fully unfrozen (final)** | **36.2%** | **73.8%** | **19.0%** |

Full metrics, hyperparameters, and dataset details: [`model_metrics.csv`](./model_metrics.csv).
Full write-up, including error analysis and real-world generalization findings:
[`Capstone_Final_Report.pdf`](./Capstone_Final_Report.pdf).

**Key finding:** the model's errors show a strong bias toward predicting "Poodle" for many
different curly/wavy-coated breeds -- a sign it's currently keying on coat texture as a
shortcut rather than finer breed-distinguishing features. Real-world performance (PetFinder)
also drops far more in macro-F1 than in raw accuracy, indicating the model generalizes well
to common breeds but poorly to rarer ones -- see the report for the full analysis.

## Repository Structure

```
├── 02_data_wrangling.ipynb           # Data cleaning, breed harmonization, EDA prep
├── 03_exploratory_data_analysis.ipynb
├── 04_preprocessing_and_training.ipynb  # Label encoding, image normalization, train/val/test split
├── 05_modeling.ipynb                 # Baseline + 3 model comparison, final model selection
├── model_metrics.csv                 # Final model parameters, hyperparameters, and metrics
├── Capstone_Final_Report.pdf         # Full project report
└── figures/                          # Report figures
```

## Limitations & Next Steps

- Trained for only 3 epochs on CPU; every increase in fine-tuning freedom improved results
  with no plateau yet, suggesting more training is the most promising next step.
- No DNA-verified ground truth exists for any breed label used (Stanford Dogs or PetFinder)
  -- results describe agreement with human-assigned labels, not biological accuracy.
- See the full report for a complete discussion and three concrete recommendations.
