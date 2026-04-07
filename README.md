<div align="center">

# 💻 Laptop Price Predictor

**[🔴 Live Demo: Try the Predictor Here](https://laptoppricepredictor1.onrender.com/)**

An end-to-end Machine Learning web application deployed on Render that predicts the market price of a laptop based on its hardware specifications and features.

</div>

---

## 🚀 Overview

Buying a laptop can be overwhelming with countless configurations and fluctuating prices. The **Laptop Price Predictor** utilizes a trained machine learning model to estimate the cost of a laptop based on user-selected specifications like RAM, Processor, GPU, Battery, and Display type. 

## 🛠️ Technologies Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![Render](https://img.shields.io/badge/Render-%46E3B7.svg?style=for-the-badge&logo=render&logoColor=white)

## ✨ Features

* **Accurate Predictions:** Powered by a robust Scikit-Learn pipeline (`pipe.pkl`).
* **Dynamic UI:** Dropdowns and inputs populated dynamically from the dataset (`df.pkl`).
* **Web Interface:** Clean, responsive front-end built with HTML/CSS and served via Python Flask.
* **Live Deployment:** Fully hosted and accessible online via Render.

## 🗂️ Project Structure

* **`main.py`**: The main web server script routing requests and handling model inference.
* **`pipe.pkl`**: The serialized Machine Learning pipeline containing the preprocessor and trained estimator.
* **`df.pkl`**: The serialized Pandas DataFrame used to feed options into the web application's frontend.
* **`templates/`**: Directory containing the HTML structural files for the user interface.
* **`static/css/`**: Directory containing the stylesheets for UI design.
* **`Procfile`**: Configuration file for deploying the application to cloud platforms.
* **`requirements.txt`**: List of all Python dependencies required to run the project.

## ⚙️ Local Installation & Setup

Ensure you have Python installed on your system. 

```bash
git clone [https://github.com/your-username/LaptopPricePredictor.git](https://github.com/your-username/LaptopPricePredictor.git)
cd LaptopPricePredictor
pip install -r requirements.txt
python main.py
