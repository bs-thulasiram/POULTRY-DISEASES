

# 🐣Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management 

A deep learning–based web application that identifies common poultry diseases such as **Coccidiosis**, **Salmonella**, **Newcastle Disease**, or **Healthy** from chicken feces images using a custom-trained **MobileNetV2** model. The application is built using **Streamlit** and **TensorFlow**.

---



## 🚀 Features

- 🧠 Predicts poultry disease from uploaded fecal images
- ✅ Detects: Coccidiosis, Salmonella, Newcastle, Healthy
- 🎨 Beautiful dark-themed UI
- ⚡ Real-time predictions with confidence score
- 🐍 Built with TensorFlow, Streamlit, PIL, and NumPy

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit (custom HTML/CSS for styling)
- **Model**: MobileNetV2 `.h5` file (pretrained Keras model)
- **Backend**: TensorFlow, Pillow, NumPy

---

## 📂 Project Structure

```

poultry-diseases-identifier/
├── app.py                   # Main Streamlit application
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── static/
│   └── uploaded/
│       └── (uploaded images go here)
├── templates/
│   ├── index.html
│   └── result.html
````

---

## ✅ How to Run the App

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/poultry-diseases-identifier.git
cd poultry-diseases-identifier
````

### 2. Set Up Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
# Or use: source venv/bin/activate   # On Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

Visit: [http://localhost:8501](http://localhost:8501)

---

## 📥 Model File

Make sure `mobilenetv2.h5` is placed inside:

```
model/mobilenetV2/mobilenetv2.h5
```

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Authors

* BS Thulasi Ram - (https://github.com/bs-thulasiram)
---

## 🙏 Acknowledgements

* Streamlit team for the awesome framework
* TensorFlow/Keras for model training
* Open datasets for poultry disease images


"# Classification-of-Poultry-Diseases" 
