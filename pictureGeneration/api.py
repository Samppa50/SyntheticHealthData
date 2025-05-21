import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from generator import generate

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    # Access the extra fields sent from the other container
    pic_amount = request.form.get('pic-amount', type=int)
    epoch_amount = request.form.get('epoch-amount', type=int)
    session_id = request.form.get('session_id')

    print(f"pic_amount: {pic_amount}, epoch_amount: {epoch_amount}, session_id: {session_id}")

    # Accept multiple files with the key 'images'
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400

    files = request.files.getlist('images')
    print([file.filename for file in files])
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No selected files'}), 400

    saved_files = []
    for file in files:
        i = 1
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
            unique_filename = f"{i}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            saved_files.append(unique_filename)
            i += 1
        else:
            print(f"Skipping invalid file: {file.filename}")

    if not saved_files:
        return jsonify({'error': 'No valid image files provided'}), 400

    generate(session_id, pic_amount, epoch_amount)

    return jsonify({'message': 'Upload successful', 'filenames': saved_files}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0')

