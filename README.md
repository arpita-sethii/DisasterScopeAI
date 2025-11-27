# 🌪️ DisasterScope AI v2  
### Multimodal Disaster Detection, Image Authenticity Verification & Geolocated Severity Mapping

---

## 📘 Project Overview
**DisasterScope AI v2** is an end-to-end multimodal disaster analysis system that processes both **disaster images** and **tweet text** to generate rapid, reliable, and interpretable insights during disaster events.  
It is designed to counter misinformation, detect real disasters, and provide map‑based severity alerts.

The system performs:

- Real vs Fake Image Detection (EfficientNet‑B0)  
- Disaster Type Classification (Transformer-based)  
- Tweet Cleaning & Severity Extraction  
- Location Extraction using dictionary, regex, and geocoding  
- Image-Based Damage Assessment  
- Fusion of text + image severity  
- Interactive Folium Map Generation  
- Severity-Based Alerts  

---

## 🏗 System Architecture

```
Image → Preprocessing → Authenticity Check → Damage Analysis → Image Severity
Tweet → Cleaning → Disaster Type → Severity + Location → Text Severity
                      ↓
               Severity Fusion
                      ↓
       Final Severity + Heatmap + Interactive Map
```

---

## 🔍 Key Features

### **1. Image Authenticity Detection**
- Backbone: **EfficientNet‑B0**
- Output: REAL / FAKE / UNCERTAIN  
- Test Accuracy: **96.5%**
- Balanced dataset (480 real + 480 fake images)

### **2. Transformer-Based Text Classification**
Two modes are supported:
- **Zero‑shot classification** using `facebook/bart-large-mnli`
- **Optional DistilBERT mini‑trained classifier (.pt)**

Labels:
- Earthquake  
- Flood  
- Wildfire  
- Hurricane  
- Unknown  

### **3. Text Cleaning & Severity Extraction**
Severity Levels:
- **HIGH**  
- **MEDIUM**  
- **LOW**

Based on keyword scoring.

### **4. Location Extraction**
Uses:
- Predefined dictionary of 23+ global locations  
- Regex-based extraction  
- Fallback: GeoPy Nominatim geocoding  

### **5. Image Damage Analysis**
Computer vision techniques:
- Canny edge density  
- Texture variance  
- Color region detection (fire/smoke/water indicators)  

Produces:
- Damage severity  
- Heatmap visualization  

### **6. Multimodal Fusion**
Fuses:
- Image severity  
- Text severity  

Final Output:
- Final severity (H/M/L)  
- Interactive map  
- Alert description  

---

## 🗺 Features in the Output UI

- **Interactive Folium Map**
  - Red Zone → Critical  
  - Orange Zone → Warning  
  - Yellow Zone → Caution  
  - Marker showing coordinates, severity, and disaster type  

- **Heatmap Visualization**
  - Original image  
  - Damage intensity  
  - Overlay  

- **Alert Generator**
  - Critical / Warning / Advisory messages

---

## 📂 Project Structure

```
DisasterScopeAI/
│
├── models/
│   ├── real_fake_classifier.pt
│   └── distilbert_disaster.pt (optional)
│
├── data/
│   ├── raw/
│   └── fake_disaster_images/
│
├── outputs/
│   ├── heatmaps/
│   └── disaster_map.html
│
├── app.py  (Streamlit interface)
├── requirements.txt
└── README.md
```

---

## 🧪 Datasets Used

### **Real Images**
- Natural Disaster Image Dataset (Kaggle)
- Classes: Earthquake, Flood, Wildfire, Cyclone
- ~3300 images available, 480 used for balanced training

### **Fake Images**
- Generated using Stable Diffusion v1.5  
- 120 per class → 480 total  

---

## 📦 Installation

### **1. Clone the Repository**
```
git clone <repo-url>
cd DisasterScopeAI
```

### **2. Install Dependencies**
```
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### **Start Streamlit App**
```
streamlit run app.py
```

Uploads:
- A disaster image  
- A corresponding tweet  

System outputs:
- Real/Fake prediction  
- Disaster type + confidence  
- Severity (text, image, combined)  
- Heatmaps  
- Interactive map  

---

## 🧠 Technical Stack

### **Machine Learning**
- PyTorch  
- timm (EfficientNet-B0)  
- Transformers (DistilBERT / BART-MNLI)  

### **Computer Vision**
- OpenCV  
- NumPy  
- Pillow  

### **Mapping**
- Folium  
- GeoPy  

### **Web Interface**
- Streamlit  
- streamlit‑folium  

---

## ⚙️ Future Improvements

- Fine‑tuning DistilBERT on larger disaster datasets  
- Multi-language tweet support  
- Satellite image support  
- Real-time Twitter API integration  
- Mobile app deployment  

---

## 👥 Contributors

- **Arpita Sethi**  
- **Manya Singh**  
- Course: UML501  
- Institution: Thapar Institute of Engineering & Technology  

---

## 📄 License
This project is for academic use only.  
Commercial use requires permission.

---
