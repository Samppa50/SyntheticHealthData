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

    # Now you can use these variables as needed
    print(f"pic_amount: {pic_amount}, epoch_amount: {epoch_amount}, session_id: {session_id}")

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        generate(session_id, pic_amount, epoch_amount)

        return jsonify({'message': 'Upload successful', 'filename': unique_filename}), 200

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0')

