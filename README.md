🦵 Knee Osteoarthritis Severity Detection using Swin Transformer

An end-to-end Explainable AI system for automated Knee Osteoarthritis (OA) severity grading from X-ray images using a Swin Transformer V2.
The system predicts five OA severity levels (KL Grades 0–4) along with probability scores and Grad-CAM visual explanations, and is deployed as a real-time Streamlit web application.

🚀 Key Features

✅ Transformer-based medical image analysis (Swin Transformer V2)

✅ Five-class OA severity grading (No OA → Very Severe OA)

✅ Probability-based predictions instead of binary output

✅ Confidence-aware medical decision logic

✅ Grad-CAM heatmaps for model explainability

✅ Interactive Streamlit web application

✅ Deployment-ready (Hugging Face / Streamlit Cloud)

🧠 Problem Statement

Knee Osteoarthritis is a progressive joint disease commonly diagnosed using radiographic (X-ray) imaging.
Manual grading of OA severity is:

Time-consuming

Subjective
📊 Severity Classes (KL Grades)
Class	Description
0	No Osteoarthritis
1	Mild Osteoarthritis
2	Moderate Osteoarthritis
3	Severe Osteoarthritis
4	Very Severe Osteoarthritis 

🧪 Dataset

Source: Public Knee Osteoarthritis Dataset (Kaggle / OAI-based)

Modality: Knee X-ray images

Labels: Severity grades (0–4)

Split: Train / Validation

Preprocessing:

Grayscale → RGB

Resize to 256×256

ImageNet normalization

⚠️ Dataset is not included in this repository due to licensing constraints.

🧠 Model Details

Architecture: Swin Transformer V2 (Tiny)

Input Size: 256 × 256

Loss Function: Cross-Entropy Loss with class weights

Optimizer: Adam

Output: Class-wise probabilities

Explainability: Grad-CAM

⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/<your-username>/knee-oa-severity-swin.git
cd knee-oa-severity-swin

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
streamlit run src/streamlit_app.py

🌐 Live Demo

👉 Hugging Face Spaces:
https://huggingface.co/spaces/PriyanshuRaut96/knee-oa-severity-swin
Dependent on expert availability

This project aims to automate OA severity grading using deep learning while maintaining clinical interpretability and transparency.
