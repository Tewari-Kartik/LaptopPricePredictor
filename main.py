from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

app = Flask(__name__)

df = pickle.load(open('df.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

companies = sorted(df['Company'].unique())
types = sorted(df['TypeName'].unique())
cpu_brands = sorted(df['Cpu brand'].unique())
gpu_brands = sorted(df['Gpu brand'].unique())
os_options = sorted(df['os'].unique())

try:
    X = df[['Company','TypeName','Ram','Cpu brand','Gpu brand','Weight',
            'Touchscreen','Ips','ppi','HDD','SSD','os']]
    y = df['Price']
    y_pred_log = pipe.predict(X)
    y_pred = np.exp(y_pred_log)  # Convert log(price) -> actual price
    model_accuracy = round(r2_score(y, y_pred) * 100, 2)  # in %
except Exception as e:
    model_accuracy = "N/A"

@app.route('/')
def index():
    return render_template('index.html', companies=companies, types=types,
                           cpu_brands=cpu_brands, gpu_brands=gpu_brands,
                           os_options=os_options,
                           model_accuracy=model_accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        company = request.form.get('company')
        type_name = request.form.get('type_name')
        ram = int(request.form.get('ram'))
        cpu = request.form.get('cpu')
        gpu = request.form.get('gpu')
        weight = float(request.form.get('weight'))
        touchscreen = 1 if request.form.get('touchscreen') == 'Yes' else 0
        ips = 1 if request.form.get('ips') == 'Yes' else 0
        ppi = float(request.form.get('ppi'))
        hdd = int(request.form.get('hdd'))
        ssd = int(request.form.get('ssd'))
        os_val = request.form.get('os')  # OS dropdown

        # Create DataFrame for pipeline
        input_df = pd.DataFrame([[company, type_name, ram, cpu, gpu, weight,
                                  touchscreen, ips, ppi, hdd, ssd, os_val]],
                                columns=['Company','TypeName','Ram','Cpu brand','Gpu brand','Weight',
                                         'Touchscreen','Ips','ppi','HDD','SSD','os'])

        # Predict log(price)
        log_prediction = pipe.predict(input_df)[0]

        # Convert log(price) -> actual price
        actual_price = round(np.exp(log_prediction), 2)

        return render_template('index.html', companies=companies, types=types,
                               cpu_brands=cpu_brands, gpu_brands=gpu_brands,
                               os_options=os_options,
                               prediction_text=f"Predicted Laptop Price: ₹{actual_price:,}",
                               model_accuracy=model_accuracy)
    except Exception as e:
        return render_template('index.html', companies=companies, types=types,
                               cpu_brands=cpu_brands, gpu_brands=gpu_brands,
                               os_options=os_options,
                               prediction_text=f"Error: {str(e)}",
                               model_accuracy=model_accuracy)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)