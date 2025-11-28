# 🌪️ DisasterScope AI v2
### Multimodal Disaster Detection • Image Authenticity Verification • Geolocated Severity Mapping

---

## 📘 Project Overview

**DisasterScope AI v2** is an end-to-end multimodal disaster intelligence system that analyzes both **images** and **tweets** to detect real disasters, counter misinformation, and generate severity-based geolocated alerts.

The system performs:
- **Real vs Fake Image Detection** (EfficientNet-B0 + AI Artifact Analysis)
- **Disaster Tweet Classification** (DistilBERT fine-tuned on disaster tweets)
- **Tweet Cleaning & Severity Extraction**
- **Location Extraction + Geocoding**
- **Image Damage Analysis (CV-based)**
- **Multimodal Fusion (Image + Text Severity)**
- **Interactive Folium Map & Heatmap Generation**

---

## 🏗 System Architecture

The core system processes image and text inputs in parallel before fusing the severity scores:
```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUT                                    │
│            (Upload Image + Enter Tweet/Description)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  DUAL PIPELINE SPLIT   │
              └───────┬────────┬───────┘
                      │        │
        ┌─────────────┘        └─────────────┐
        │                                     │
        ▼                                     ▼
┌───────────────────┐              ┌──────────────────────┐
│   TEXT PIPELINE   │              │   IMAGE PIPELINE     │
└───────────────────┘              └──────────────────────┘
        │                                     │
        ▼                                     ▼
┌───────────────────┐              ┌──────────────────────┐
│  1. Text Cleaning │              │  1. Preprocessing    │
│  - Remove URLs    │              │  - Resize 224×224    │
│  - Remove @       │              │  - Normalize         │
│  - Remove #       │              │  - To Tensor         │
└─────────┬─────────┘              └──────────┬───────────┘
          │                                   │
          ▼                                   ▼
┌───────────────────┐              ┌──────────────────────┐
│  2. Disaster Type │              │  2. CNN Auth Check   │
│  Classification   │              │  EfficientNet-B0     │
│  ───────────────  │              │  Real/Fake probs     │
│  A) DistilBERT    │              └──────────┬───────────┘
│     (Primary)     │                         │
│     - 66M params  │                         ▼
│     - 4 classes   │              ┌──────────────────────┐
│     - Confidence  │              │  3. Heuristic Check  │
│                   │              │  - Dimensions        │
│  B) Keyword-based │              │  - Sensor noise      │
│     (Fallback)    │              │  - Color analysis    │
│     if conf<70%   │              │  - Texture analysis  │
│                   │              │  - 9/16 threshold    │
│  Classes:         │              └──────────┬───────────┘
│  - Earthquake     │                         │
│  - Flood          │                         ▼
│  - Wildfire       │              ┌──────────────────────┐
│  - Hurricane      │              │  4. Auth Decision    │
└─────────┬─────────┘              │  Combined CNN +      │
          │                        │  Heuristics Vote     │
          ▼                        │  ─────────────────   │
┌───────────────────┐              │  REAL / FAKE /       │
│  3. Location      │              │  VERIFIED            │
│  Extraction       │              └──────────┬───────────┘
│  ─────────────    │                         │
│  - 150+ locations │              ┌──────────┴──────────┐
│  - Pattern match  │              │                     │
│  - Return lat/lng │              ▼                     ▼
└─────────┬─────────┘    ┌────────────────┐    ┌────────────────┐
          │              │  If FAKE:      │    │  If REAL/      │
          ▼              │  ─────────     │    │  VERIFIED:     │
┌───────────────────┐    │  Display:      │    │  ─────────     │
│  4. Text Severity │    │  - Warning     │    │  Continue      │
│  Keyword Analysis │    │  - Stop here   │    │  Processing    │
│  ─────────────    │    └────────────────┘    └────────┬───────┘
│  Keywords:        │                                   │
│  - HIGH: massive, │                                   ▼
│    destroyed,     │                        ┌──────────────────────┐
│    deaths, etc    │                        │  5. Damage Analysis  │
│  - LOW: minor,    │                        │  (Computer Vision)   │
│    contained      │                        │  ───────────────     │
│  - Else: MEDIUM   │                        │  - Canny edges       │
└─────────┬─────────┘                        │  - Texture variance  │
          │                                  │  - Edge density      │
          │                                  │  - Damage severity   │
          │                                  └──────────┬───────────┘
          │                                             │
          │                                             ▼
          │                                  ┌──────────────────────┐
          │                                  │  6. Image Severity   │
          │                                  │  Edge-based:         │
          │                                  │  ───────────         │
          │                                  │  >15%: HIGH          │
          │                                  │  >8%:  MEDIUM        │
          │                                  │  <8%:  LOW           │
          │                                  └──────────┬───────────┘
          │                                             │
          └─────────────────┬───────────────────────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  SEVERITY FUSION     │
                 │  ────────────────    │
                 │  Average:            │
                 │  (Text Sev + Image   │
                 │   Severity) / 2      │
                 │                      │
                 │  Final: HIGH/MED/LOW │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   OUTPUT GENERATION  │
                 └──────────┬───────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
              ▼             ▼             ▼
    ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
    │  Emergency  │ │    Damage    │ │ Interactive  │
    │    Alert    │ │   Heatmap    │ │     Map      │
    │  ─────────  │ │  ──────────  │ │  ──────────  │
    │  HIGH:      │ │  3-Panel:    │ │  Folium:     │
    │  Evacuate!  │ │  - Original  │ │  - RED zone  │
    │             │ │  - Intensity │ │  - ORANGE    │
    │  MEDIUM:    │ │  - Overlay   │ │  - YELLOW    │
    │  Stay alert │ │              │ │              │
    │             │ │  Colormap:   │ │  Marker at   │
    │  LOW:       │ │  Green→Red   │ │  epicenter   │
    │  Monitor    │ │              │ │              │
    └──────┬──────┘ └──────┬───────┘ └──────┬───────┘
           │               │                │
           └───────────────┼────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   DISPLAY RESULTS    │
                │  ────────────────    │
                │  Streamlit Interface │
                │  ───────────────     │
                │  - Authenticity      │
                │  - Disaster type     │
                │  - Location + coords │
                │  - Severity level    │
                │  - Damage heatmap    │
                │  - Interactive map   │
                │  - Emergency alert   │
                └──────────────────────┘
                           │
                           ▼
                         [END]

═══════════════════════════════════════════════════════════════════

LEGEND:
━━━━━━ : Main Flow
┌────┐ : Process/Operation
◇      : Decision Point
[    ] : Terminal (Start/End)
═══════ : Section Separator

═══════════════════════════════════════════════════════════════════

KEY COMPONENTS:

TEXT PIPELINE:
├─ DistilBERT Classifier (Primary)
│  ├─ Model: distilbert-base-uncased
│  ├─ Parameters: 66M
│  ├─ Training: Fine-tuned on disaster tweets
│  └─ Output: 4-class prediction with confidence
│
└─ Keyword Classifier (Fallback)
   ├─ Triggers: When DistilBERT confidence < 70%
   └─ Method: Rule-based pattern matching

IMAGE PIPELINE:
├─ CNN Classifier
│  ├─ Model: EfficientNet-B0
│  ├─ Training: 480 real + 480 fake images
│  └─ Output: Real/Fake probabilities
│
└─ Heuristic Detector
   ├─ Method: Computer vision analysis
   ├─ Checks: 6 indicators (dimensions, noise, etc)
   └─ Threshold: 9/16 indicators = FAKE

FUSION LAYER:
└─ Combines text severity + image severity
   └─ Generates final output with all components

═══════════════════════════════════════════════════════════════════
                               
```

---

## 🔍 Key Features

### 1. Image Authenticity Detection
**Two-Stage Verification System:**

#### Stage 1: CNN-Based Classification
- **Backbone:** EfficientNet-B0 (`timm`)
- **Output:** `REAL` / `FAKE` confidence scores
- **Training data:**
    - Real images: Natural Disaster Image Dataset (Kaggle) — earthquake, flood, wildfire, cyclone classes
    - Fake images: AI-generated via Stable Diffusion v1.5 (120 per class)
- **Approach:** Custom classification head on EfficientNet features, mixed precision training, augmentations
- **Performance:** 96.5% accuracy on test set
- **Artifacts saved:** `models/real_fake_classifier.pt`, `outputs/confusion_matrix_real_fake.png`, `outputs/training_history_real_fake.png`

#### Stage 2: Heuristic AI Artifact Detection
**Purpose:** Detect AI-generated images that bypass CNN classifier (reduces false negatives)

**Detection Techniques:**
1. **Dimensional Analysis:** Checks for AI-typical sizes (512×512, 768×768, 1024×1024)
2. **Sensor Noise Analysis:** Real cameras have sensor noise; AI images lack authentic noise patterns
3. **Color Analysis:** 
   - Oversaturation detection (AI often creates unnaturally vivid colors)
   - Channel correlation analysis (AI images have unusual RGB correlations)
4. **Texture Analysis:** Detects overly smooth/uniform textures typical of AI generation
5. **Contrast Analysis:** Identifies unnaturally high contrast patterns
6. **Edge Consistency:** Analyzes edge sharpness uniformity (AI images show specific patterns)

**Decision Logic:**
- Both CNN and heuristics vote on authenticity
- Heuristics can override CNN for high-confidence fake detections
- Threshold: 9/16 indicators needed for fake classification
- Final output: `REAL`, `FAKE`, or `VERIFIED`

**Why Dual-Stage?**
- CNN alone: 96.5% accurate but can miss sophisticated AI images
- Heuristics: Catches edge cases by detecting physical image properties
- Combined: More robust against evolving AI generation techniques

### 2. Text Classification (DistilBERT)
- **Model:** `distilbert-base-uncased` fine-tuned for 4-way disaster classification
- **Classes:** `earthquake`, `flood`, `wildfire`, `hurricane`
- **Datasets used:**
    - Kaggle `nlp-getting-started` (real disaster tweets, filtered `target==1`)
    - CrisisLex-style curated/augmented disaster samples (to balance & increase variety)
- **Training details:** Tokenization max_length=128, stratified train/test split, Trainer API (Hugging Face), 3 epochs
- **Artifacts saved:** `models/distilbert_disaster_classifier/` (tokenizer + model files)

### 3. Hybrid / Fallback Logic
- **Primary:** DistilBERT prediction with confidence score
- **Fallback:** Keyword-based classifier when DistilBERT confidence < threshold (e.g., 0.7)
- *This ensures robust predictions for short/ambiguous tweets.*

### 4. Location Extraction & Geocoding
- **Strategy:** 23+ predefined known locations (fast path) + regex pattern matching + Geopy (Nominatim) fallback
- **Returns:** Location name, latitude, longitude, confidence

### 5. Severity Estimation & Damage Analysis
- **Text severity:** Keyword scoring (HIGH/MEDIUM/LOW)
- **Image severity:** Edge density (Canny), texture variance, color cues (fire/flood indicators)
- **Fusion:** Average / rule-based fusion of text and image severity for final alert level

### 6. Mapping & Alerts
- **Interactive maps:** Folium with radius zones (`RED` / `ORANGE` / `YELLOW`) based on severity
- **Heatmaps:** 3-panel visualization (original, intensity map, overlay)
- **Alerts:** Severity-based messages (`Evacuate` / `Prepare` / `Monitor`)

---

## 🗂 Project Structure
```
DisasterScopeAI/
├── app.py                                # Main Streamlit application
├── requirements.txt
├── README.md
├── models/
│   ├── real_fake_classifier.pt           # EfficientNet-B0 checkpoint
│   └── distilbert_disaster_classifier/   # Tokenizer + model (optional)
├── outputs/
│   ├── confusion_matrix_real_fake.png
│   ├── training_history_real_fake.png
│   └── disaster_map.html
├── data/
│   ├── raw/
│   │   └── disaster_images/              # Real disaster dataset
│   └── fake_disaster_images/             # AI-generated fakes
├── src/
│   ├── real_fake_classifier.py           # CNN training script
│   ├── text_classifier.py                # DistilBERT training script
│   ├── heuristic_detector.py             # AI artifact detection
│   └── utils.py
└── notebooks/
    └── experiments.ipynb
```

> **Note:** Avoid uploading full datasets or large model files (>100MB) to GitHub. Use Git LFS, Hugging Face Hub, or GitHub Releases for large artifacts.

---

## 📦 Installation
```bash
git clone <repo-url>
cd DisasterScopeAI
python -m venv venv
# activate venv: Windows: venv\Scripts\activate  |  Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

**Suggested `requirements.txt` highlights:**
```
torch
torchvision
timm
transformers
datasets
Pillow
numpy
opencv-python
folium
streamlit
streamlit-folium
geopy
scikit-learn
matplotlib
seaborn
```

---

## 🚀 Run (Local Demo)

Start the Streamlit interface:
```bash
streamlit run app.py
```

**Outputs:**
- Real / Fake prediction + confidence (with dual-stage verification)
- Disaster type + confidence (DistilBERT)
- Severity (text, image, fused)
- Damage heatmap and interactive Folium map

---

## 🧪 Model Training Notes

### CNN Image Classifier
- **Architecture:** EfficientNet-B0 with custom 3-layer classification head
- **Training:** Mixed precision, CosineAnnealingLR scheduler, 10 epochs
- **Data augmentation:** Random crop, horizontal flip, color jitter
- **Dataset:** 480 real + 480 AI-generated (balanced)
- **Performance:** 96.5% test accuracy, 96.5% F1 score
- **Training script:** `src/real_fake_classifier.py`

### Heuristic Detector
- **Type:** Rule-based computer vision analysis
- **No training required:** Analyzes physical image properties
- **Techniques:** Noise analysis, color correlation, texture uniformity, dimensional checks
- **Integration:** Works alongside CNN for ensemble detection
- **Implementation:** `src/heuristic_detector.py`

### DistilBERT Text Classifier
- **Base model:** `distilbert-base-uncased` (66M parameters)
- **Fine-tuning:** 3 epochs on disaster tweet corpus
- **Dataset:** Combined Kaggle disaster tweets + augmented crisis samples
- **Performance:** Evaluated with classification report
- **Training script:** `src/text_classifier.py`
- **Artifacts:** Saved at `models/distilbert_disaster_classifier/`

### Data Generation (Fake Images)
- **Tool:** Stable Diffusion v1.5
- **Prompts:** Realistic disaster scenes per category
- **Generation:** 120 images per class (earthquake, flood, wildfire, hurricane)
- **Purpose:** Train CNN to detect AI-generated misinformation

---

## 🎯 System Performance

| Component | Metric | Value |
|-----------|--------|-------|
| **CNN Classifier** | Test Accuracy | 96.5% |
| | F1 Score | 96.5% |
| | Real Precision | 98.6% |
| | Fake Recall | 98.6% |
| **Heuristic Detector** | Coverage | Catches CNN false negatives |
| | Threshold | 9/16 indicators |
| **Combined System** | False Negative Reduction | Significant improvement |
| **Text Classifier** | Accuracy | 99.4% |
| **Response Time** | End-to-end | < 2 seconds |

---

## 🧾 Reproducibility & Artifacts

Included in repo (or via release / HF link if file size large):

- Training scripts (`src/*.py`) and notebooks
- Trained checkpoint(s) (or download link)
- Evaluation artifacts: confusion matrices, training curves, classification reports
- Heuristic detection implementation with documented thresholds
- `README` with quick-start and dataset descriptions

---

## 🔬 Technical Highlights

### Image Authenticity Innovation
The dual-stage detection system represents a **defense-in-depth** approach to AI misinformation:

1. **CNN Stage:** Fast, learned features from training data
2. **Heuristic Stage:** Physics-based analysis resistant to adversarial examples

This architecture is inspired by production ML systems where:
- Primary models handle common cases efficiently
- Fallback systems catch edge cases and concept drift
- Ensemble methods improve robustness

### Why Heuristics Matter
- **Model-agnostic:** Works on any AI generator (Stable Diffusion, DALL-E, Midjourney)
- **Explainable:** Each indicator has clear physical meaning
- **Adaptable:** Thresholds can be tuned without retraining
- **Complementary:** Catches what CNNs miss

---

## 👥 Contributors

- **Arpita Sethi** - Lead Developer
- **Manya Singh** - Project Partner

**Course:** UML501 — Thapar Institute of Engineering and Technology  
**Instructor:** Dr. Anjula Mehto

---

## 📝 License

Academic use only. For commercial use contact the authors.

---

## 🙏 Acknowledgments

- **Datasets:** Kaggle Natural Disaster Dataset, Real Disaster Tweets Dataset (Under getting started with nlp Competition)
- 
- **Models:** Hugging Face Transformers, timm (PyTorch Image Models)
- **AI Generation:** Stable Diffusion v1.5
- **Inspiration:** Real-world need for disaster misinformation detection

---

## 📧 Contact

For questions or collaboration: [Your Email/GitHub]

---

*Built with ❤️ for rapid disaster response and misinformation detection*
