import os
from flask import Flask, request, jsonify, send_file, redirect
from werkzeug.utils import secure_filename
from datetime import datetime
from generator import generate, get_progress, set_stop
import shutil

app = Flask(__name__)


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
current_id = 404
user_prosessing = False

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    global current_id
    global user_prosessing

    pic_amount = request.form.get('pic-amount', type=int)
    epoch_amount = request.form.get('epoch-amount', type=int)
    session_id = request.form.get('session_id')
    generation_type = request.form.get('generation-type', type=int)

    print("generation_type app.py:", generation_type)

    if user_prosessing and current_id != session_id:
        return jsonify({'error': 'Another generation is in progress. Please wait until it finishes.'}), 403

    user_prosessing = True
    current_id = session_id

    upload_folder_path = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(upload_folder_path, exist_ok=True)

    files = request.files.getlist('images')
    print([file.filename for file in files])
    if not files or all(f.filename == '' for f in files):
        print('No selected files')

    saved_files = []
    i = 1
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{i}_{filename}"
            file_path = os.path.join(upload_folder_path, unique_filename)
            file.save(file_path)
            saved_files.append(unique_filename)
            i += 1
        else:
            print(f"Skipping invalid file: {file.filename}")

    #if not saved_files:
    #    return jsonify({'error': 'No valid image files provided'}), 400

    generate(session_id, pic_amount, epoch_amount, generation_type)
    user_prosessing = False
    return redirect(f"/call_flag")

@app.route('/download/<folder_name>', methods=['GET'])
def download_folder(folder_name):
    folder_path = os.path.join('download', folder_name)
    if not os.path.exists(folder_path):
        return jsonify({'error': 'Folder not found'}), 404

    zip_path = f"{folder_path}.zip"
    # Create a zip of the folder
    shutil.make_archive(folder_path, 'zip', folder_path)

    # Send the zip file
    response = send_file(zip_path, as_attachment=True)

    os.remove(zip_path)

    return response

@app.route('/user/data/delete', methods=['DELETE'])
def delete_user_data():
    global current_id
    current_id = 404
    shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
    gif_path = os.path.join('gifs', 'generated', 'animated.gif')
    if os.path.exists(gif_path):
        os.remove(gif_path)
    print("User data deleted successfully.")
    return jsonify({'message': 'User data deleted successfully.'})



@app.route('/progress', methods=['GET'])
def get_progress_route():
    global progress
    progress = get_progress()
    return jsonify({'progress': progress})

@app.route('/stop', methods=['GET', 'POST'])
def stop_generation():
    global user_prosessing
    user_prosessing = False
    set_stop(True)
    delete_user_data()
    print("Generation stopped.")
    return jsonify({'message': 'Generation stopped successfully.'})

@app.route('/reset/stop', methods=['GET', 'POST'])
def reset_stop():
    set_stop(False)
    print("Stop processing reset.")
    return jsonify({'message': 'Stop processing reset successfully.'})

@app.route('/gif/download', methods=['GET'])
def download_gif():
    gif_path = 'gifs/generated/animated.gif'
    if not os.path.exists(gif_path):
        return jsonify({'error': 'GIF not found'}), 404

    return send_file(gif_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0')

