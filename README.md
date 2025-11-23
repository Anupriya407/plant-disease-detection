**🌿 Plant Disease Detection (Deep Learning)**

This project identifies plant leaf diseases using a deep learning model.
It works on Tomato, Potato, and Pepper leaves and predicts the exact disease.

The app allows you to upload a leaf image and the model will:

✔ Predict the plant type

✔ Predict the disease name

✔ Show the confidence score

✔ Display Grad-CAM heatmap (infected region)

**🚀 Features**

Detects 15 plant diseases

Uses ResNet-18 (pretrained deep learning model)

99% accuracy

Simple Streamlit web app

Works with images from Google or mobile

Includes Grad-CAM for explainability

**📂 Project Structure**

plant-disease-detection/
│
├── app/
│   ├── streamlit_app.py
│   └── sample.jpg
│
├── data/
│   ├── PlantVillage/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── best_model.pth
│
├── notebooks/
│   ├── 01_explore_data.ipynb
│   └── 02_train_model.ipynb
│
├── results/
│   ├── class_counts.csv
│   └── metrics.json
│
├── src/
│   ├── infer.py
│   ├── gradcam.py
│   ├── split_dataset.py
│   └── __init__.py
│
├── venv/
│
├── README.md
├── requirements.txt
└── .gitignore

**▶️ How to Run**

**1️⃣ Activate virtual environment****
venv\Scripts\activate

**2️⃣ Run Streamlit**
streamlit run app/streamlit_app.py

**3️⃣ Upload a leaf image**

(or click Use sample image)
to view:

Disease prediction

Confidence

Grad-CAM heatmap

**📌 Example Output**
Potato___Late_blight — 1.0000


✔ Meaning: Potato leaf has Late Blight disease
✔ Confidence: 100%

**✨ Future Improvements****

Mobile app version

Real-time camera detection

Support for more crop varieties