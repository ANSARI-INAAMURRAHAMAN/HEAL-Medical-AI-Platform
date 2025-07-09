# HEAL - Health Evaluation and Lab Analysis

HEAL is a comprehensive disease diagnosis website that uses machine learning models to assist doctors and medical practitioners in the diagnosis of certain diseases and health issues. 

**Developed by:Anshul Bajaj & Ansari Inaamurraheman**

## 🩺 Supported Diagnoses
1. **Brain Tumor Detection** from MRI scans
2. **Pneumonia Detection** from Chest X-Rays
3. **Stroke Risk Prediction**
4. **Early Stage Diabetes Prediction**
5. **Chronic Kidney Disease Diagnosis**
6. **Liver Disease Diagnosis**
7. **Heart Disease Diagnosis**

## 🚀 Key Features
- Automated disease prediction using trained ML/DL models
- User-friendly web interface for uploading medical data and images
- Supports both image-based and tabular data predictions
- Real-time diagnosis results with high accuracy

## 🛠️ Technologies Used
- **Backend**: Python, Flask
- **Machine Learning**: scikit-learn, TensorFlow, Keras
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy, OpenCV
- **Model Management**: Git LFS for large model files

## ⚡ Getting Started

### Prerequisites
- Python 3.7+
- Git LFS

### Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd HEAL
   ```

2. **Install Git LFS:**
   ```bash
   git lfs install
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the website:**
   Open your browser and go to `http://localhost:5000`

## 📸 Screenshots

![HEAL Homepage](https://user-images.githubusercontent.com/92988197/222878687-7fb60431-efd5-4bc8-909f-5c015666d1a1.jpeg)

![Disease Selection](https://user-images.githubusercontent.com/92988197/222878695-f01d6971-87ce-4463-9aaf-86c397982eed.jpeg)

![Diagnosis Results](https://user-images.githubusercontent.com/92988197/222878700-767dc192-dbd6-4fd2-a953-d7b22ce5c6d5.jpeg)

## 🔬 Machine Learning Models
- **Random Forest**: Heart disease, diabetes, kidney disease
- **SVM**: Liver disease diagnosis
- **Decision Tree**: Stroke risk prediction
- **CNN**: Brain tumor and pneumonia detection from medical images

## 👥 Contributors
- **Ansari Inaamurraheman** - Full Stack Development, Machine Learning Implementation
- **Anshul Bajaj** - Machine Learning Models, Data Processing

## 📝 Note
This project uses Git LFS to handle large model files. Make sure to install `git-lfs` on any device where you clone this repository.

---

*This project is designed to assist medical professionals and should not replace professional medical consultation.*
