🌿 Plant Disease Detection

A deep-learning project that detects plant leaf diseases using ResNet-18 with ≈99% accuracy.
Upload any leaf image and the app will:

Predict the plant type

Predict the disease name

Show confidence score

Display Grad-CAM heatmap for explainability

🚀 How to Run
1️⃣ Activate virtual environment
venv\Scripts\activate

2️⃣ Run Streamlit app
streamlit run app/streamlit_app.py

3️⃣ Upload a leaf image

(or use the sample image provided)

plant-disease-detection/
├── app/
│ └── streamlit_app.py
├── data/
├── models/
│ └── best_model.pth
├── notebooks/
├── results/
├── src/
│ ├── infer.py
│ ├── gradcam.py
│ └── split_dataset.py
└── requirements.txt


✨ Features

15 disease classes

Grad-CAM visual explanation

Works with images from Google or mobile

Clean UI using Streamlit

📌 Example Output
Potato___Late_blight — 1.0000
