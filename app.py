from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import cv2
import numpy as np
import pickle
import os
import joblib
import pandas as pd

app = Flask(__name__)

model = load_model(r"models\identify_disease\plant_model.h5")

with open(r'D:\crop disease detection\models\dicts\ref_dict.pickle', 'rb') as file:
    class_of_disease = pickle.load(file)
crop_model_filename = r'models\crop yeild\yeild_pred_model.joblib'
crop_loaded_model = joblib.load(crop_model_filename)

with open(r'models\dicts\disease_dict.pkl', 'rb') as file:
    medicine1 = pickle.load(file)
with open(r'models\dicts\disease_info.pkl', 'rb') as file:
    cause = pickle.load(file)

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    return img


@app.route('/')
def index():
    return render_template('front.html')

@app.route('/compare_disease')
def compare_disease():
    return render_template('compare disease.html')


@app.route('/identify_disease_page')
def identify_disease_page():
    return render_template('identify disease.html')


@app.route('/fertigation')
def fertigation():
    return render_template('fertigation.html')

@app.route('/yeild_pred_page')
def yeild_pred_page():
    return render_template('yeild prediction.html')

@app.route('/file')
def file():
    return render_template('file.css')

@app.route('/identify_disease', methods=['POST'])
def identify_disease():

    if request.method == 'POST':
        selected_crop = request.form['cropSelection']
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('identify disease.html', error='No selected file')

        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        image_path = os.path.join(upload_folder, file.filename)
        file.save(image_path)
        input_image = load_and_preprocess_image(image_path)
        predictions = model.predict(np.array([input_image]))
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_of_disease[(predicted_class_index)]
        chemical = medicine1[predicted_class][0]
        medicine = medicine1[predicted_class][1]

        return render_template('identify disease.html', image='../static/pesticideimg/'+medicine+'.jpg',
                               result=predicted_class,
                               chemical=chemical,
                               Medicine=medicine,
                               yesno="NO" if 'health' in predicted_class else 'YES',
                               cause=cause[predicted_class],
                               duration='3 to 4 Days approx.',
                               url=f"https://www.amazon.com/s?k={medicine+str(' fertilizer  ')}")

    return render_template('identify disease.html', result=None, error=None)

@app.route('/crop_yield_prediction', methods=['POST'])
def crop_yield_prediction():
    if request.method == 'POST':
        data = request.json
        cropname = data['Crop'][0]
        croptype = data['Season'][0]
        state = data['State'][0]
        area = float(data['Area'][0])
        production = float(data['Production'][0])
        rainfall = float(data['Annual_Rainfall'][0])
        fertilizer = float(data['Fertilizer'][0])
        pesticide = float(data['Pesticide'][0])

        new_data = pd.DataFrame({
            'Crop': [cropname],
            'Season': [croptype],
            'State': [state],
            'Area': [area],
            'Production': [production],
            'Annual_Rainfall': [rainfall],
            'Fertilizer': [fertilizer],
            'Pesticide': [pesticide]
        })

        prediction_result = crop_loaded_model.predict(new_data)

        return jsonify({'result': (production/area)})
    
    return jsonify({'result': None, 'error': 'Invalid request'})    
if __name__ == '__main__':
    app.run(debug=True, port=8080)
