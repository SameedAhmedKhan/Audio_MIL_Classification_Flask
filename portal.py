import matplotlib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

matplotlib.use('Agg')
import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import librosa
import librosa.display

from tensorflow.keras.models import load_model
from google.cloud import storage
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client.from_service_account_json("sharp-imprint-350519-f953733bff3f.json")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    blob.make_public()
    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


@app.route('/')
@app.route('/form')
def form():
    return render_template('landing_page.html')


@app.route('/data', methods=['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        bucket_name = 'toseef-test-sql-2'
        data_to_pass = {}

        f = request.files['file']
        f.save(secure_filename(f.filename))

        source_file_name = app.config['UPLOAD_FOLDER'] + "/" + f.filename
        print(source_file_name)
        destination_blob_name = source_file_name.split("/")[-1]
        print(source_file_name)
        print(destination_blob_name)
        upload_blob(bucket_name, source_file_name, destination_blob_name)

        data_to_pass['Path'] = "https://storage.googleapis.com/toseef-test-sql-2/" + destination_blob_name

        fig = plt.Figure()
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        file_location = source_file_name
        y, sr = librosa.load(file_location)
        melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)
        p = librosa.display.specshow(melSpec_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
        plt.savefig('spec.png')

        source_file_name = 'spec.png'
        destination_blob_name = 'spec.png'
        print(source_file_name)
        print(destination_blob_name)
        upload_blob(bucket_name, source_file_name, destination_blob_name)

        data_to_pass['SPEC'] = "https://storage.googleapis.com/toseef-test-sql-2/" + destination_blob_name

        im = Image.open(r'spec.png')
        width, height = im.size
        left = 80
        top = 55
        right = 500 + left
        bottom = 370 + top
        im1 = im.crop((left, top, right, bottom))
        im1.save('mil.png')

        source_file_name = 'mil.png'
        destination_blob_name = 'mil.png'
        print(source_file_name)
        print(destination_blob_name)
        upload_blob(bucket_name, source_file_name, destination_blob_name)

        data_to_pass['MEL'] = "https://storage.googleapis.com/toseef-test-sql-2/" + destination_blob_name

        saved_model = load_model('saved_model')
        im = Image.open("mil.png")
        im = im.convert('RGB')
        newsize = (300, 300)
        im1 = im.resize(newsize)

        im2arr = np.array(im1)
        im2arr = im2arr.reshape(1, 300, 300, 3)

        labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        prediction = saved_model.predict(im2arr)
        lables_ind = np.argmax(prediction[0])
        predicted_label = labels[lables_ind]
        print(predicted_label)
        data_to_pass['prediction'] = predicted_label
        return render_template('data.html', form_data=data_to_pass)


app.run(host='localhost', port=5000, debug=True)
