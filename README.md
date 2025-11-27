# 🌪️ DisasterScope AI v2
### Multimodal Disaster Detection • Image Authenticity Verification • Geolocated Severity Mapping

---

## 📘 Project Overview

**DisasterScope AI v2** is an end-to-end multimodal disaster intelligence system that analyzes both **images** and **tweets** to detect real disasters, counter misinformation, and generate severity-based geolocated alerts.

The system performs:
- **Real vs Fake Image Detection** (EfficientNet-B0)
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

```
IMAGE PIPELINE                             TEXT PIPELINE
```

┌────────────────────────┐               ┌──────────────────────────┐
│ Image Upload           │               │ Tweet Text Input         │
└───────────────┬────────┘               └───────────────┬──────────┘
↓                                        ↓
Preprocessing (PIL / OpenCV)                 Text Cleaning
↓                                        ↓
Authenticity Check (EfficientNet)         DistilBERT Classification
↓                                        ↓
Damage Analysis (edges / texture)           Severity Extraction
↓                                        ↓
Image Severity                           Location Extraction
└──────────────┬───────────────┬────────┘
\              /
\            /
\          /
↓
Severity Fusion
↓
Final Severity • Heatmap • Interactive Folium Map

```


---

## 🔍 Key Features

### 1. Image Authenticity Detection
- **Backbone:** EfficientNet-B0 (`timm`)
- **Output:** `REAL` / `FAKE` (and `UNCERTAIN` threshold)
- **Training data:**
    - Real images: Natural Disaster Image Dataset (Kaggle) — earthquake, flood, wildfire, cyclone classes
    - Fake images: AI-generated via Stable Diffusion v1.5 (120 per class)
- **Approach:** custom classification head on EfficientNet features, mixed precision training, augmentations
- **Artifacts saved:** `models/real_fake_classifier.pt`, `outputs/confusion_matrix_real_fake.png`, `outputs/training_history_real_fake.png`

### 2. Text Classification (DistilBERT)
- **Model:** `distilbert-base-uncased` fine-tuned for 4-way disaster classification
- **Classes:** `earthquake`, `flood`, `wildfire`, `hurricane`
- **Datasets used:**
    - Kaggle `nlp-getting-started` (real disaster tweets, filtered `target==1`)
    - CrisisLex-style curated/augmented disaster samples (to balance & increase variety)
- **Training details:** tokenization max_length=128, stratified train/test split, Trainer API (Hugging Face), 3 epochs (example)
- **Artifacts saved:** `models/distilbert_disaster_classifier/` (tokenizer + model files)

### 3. Hybrid / Fallback Logic
- **Primary:** DistilBERT prediction with confidence score
- **Fallback:** keyword-based classifier when DistilBERT confidence < threshold (e.g., 0.7)
- *This ensures robust predictions for short/ambiguous tweets.*

### 4. Location Extraction & Geocoding
- **Strategy:** 23+ predefined known locations (fast path) + regex pattern matching + Geopy (Nominatim) fallback
- **Returns:** location name, latitude, longitude, confidence

### 5. Severity Estimation & Damage Analysis
- **Text severity:** keyword scoring (HIGH/MEDIUM/LOW)
- **Image severity:** edge density (Canny), texture variance, color cues (fire/flood indicators)
- **Fusion:** average / rule-based fusion of text and image severity for final alert level

### 6. Mapping & Alerts
- **Interactive maps:** Folium with radius zones (`RED` / `ORANGE` / `YELLOW`) based on severity
- **Heatmaps:** 3-panel visualization (original, intensity map, overlay)
- **Alerts:** severity-based messages (`Evacuate` / `Prepare` / `Monitor`)

---

## 🗂 Project Structure (recommended)

```

DisasterScopeAI/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── real\_fake\_classifier.pt
│   └── distilbert\_disaster\_classifier/  \# tokenizer + model (optional)
├── outputs/
│   ├── confusion\_matrix\_real\_fake.png
│   └── training\_history\_real\_fake.png
├── data/
│   ├── raw/
│   └── fake\_disaster\_images/
├── src/
│   ├── real\_fake\_classifier.py
│   ├── text\_classifier.py
│   └── utils.py
└── notebooks/

````

> **Note:** Avoid uploading full datasets or large model files (>100MB) to GitHub. Use Git LFS, Hugging Face Hub, or GitHub Releases for large artifacts.

---

## 📦 Installation

```bash
git clone <repo-url>
cd DisasterScopeAI
python -m venv venv
# activate venv: Windows: venv\Scriptsctivate  |  Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
````

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
geopy
scikit-learn
matplotlib
seaborn
```

## 🚀 Run (Local Demo)

Start the Streamlit interface:

```bash
streamlit run app.py
```

**Outputs:**

  - Real / Fake prediction + confidence
  - Disaster type + confidence (DistilBERT)
  - Severity (text, image, fused)
  - Damage heatmap and interactive Folium map

-----

## 🧪 Model Training Notes (summary for reviewer)

  - **DistilBERT:** fine-tuned on combined Kaggle & curated crisis dataset; training script included at `src/text_classifier.py`. Example metrics from fine-tuning shown in notebook.
  - **Real vs Fake CNN:** trained EfficientNet-B0 backbone with custom head; training script at `src/real_fake_classifier.py`; checkpoint saved as `models/real_fake_classifier.pt`.
  - Limited GPU access constrained fake image generation to 480 samples; balanced sampling and augmentation used during training to mitigate class-size differences. Larger-scale re-training is possible and documented in `notebooks/`.

-----

## 🧾 Reproducibility & Artifacts

Included in repo (or via release / HF link if file size large):

  - Training scripts (`src/*.py`) and notebooks
  - Trained checkpoint(s) (or download link)
  - Evaluation artifacts: confusion matrices, training curves, classification reports
  - `README` with quick-start and dataset descriptions

-----

## 👥 Contributors

  - Arpita Sethi
  - Manya Singh

Course: UML501 — Thapar Institute of Engineering and Technology

-----

## 📝 License

Academic use only. For commercial use contact the authors.
