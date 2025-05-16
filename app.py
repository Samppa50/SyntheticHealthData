from flask import Flask, render_template, url_for, request, redirect, send_from_directory, session
from werkzeug.utils import secure_filename
from synthetic import main, generate_file, get_progress, update_progress
from Correlation_data import correlation
import os
import shutil
import uuid
from flask_session import Session

app = Flask(__name__)

# Configure server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'  # Store session data in the filesystem
app.config['SESSION_FILE_DIR'] = './flask_session/'  # Directory to store session files
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SECRET_KEY'] = 'your_secret_key'  # Required for signing session data

# Initialize Flask-Session
Session(app)

@app.route("/")
def index():
    file_uploaded = 'file_name' in session
    col_categories = session.get('col_categories', [])
    progress = get_progress(session.get('session_id', ''))
    if progress == 100:
        update_progress(session.get('session_id', ''), 0)  # Reset progress for this session
    return render_template("index.html", items=col_categories, file_uploaded=file_uploaded)

@app.route("/progress")
def get_progress_route():
    session_id = session.get('session_id', '')
    return {"progress": get_progress(session_id)}

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files['file']
        if file:
            session_id = session.get('session_id', str(uuid.uuid4()))  # Generate a unique session ID
            session['session_id'] = session_id

            upload_folder = f'Files/uploads/{session_id}'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, secure_filename(file.filename))
            file.save(file_path)

            sanitized_filename = secure_filename(file.filename.replace(" ", "_"))
            session['file_name'] = sanitized_filename
            session['col_categories'] = main(session_id, sanitized_filename)

            return redirect("/")
        else:
            return "Missing file", 400
    return "Invalid request method", 405

@app.route("/submit", methods=["GET", "POST"])
def submit():
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id

    exclude_columns = request.form.to_dict(flat=False).get("exclude_columns", {})
    bool_values = request.form.to_dict(flat=False).get("bool_values", {})
    ignore_zero_values = request.form.to_dict(flat=False).get("ignore_zero_values", {})
    line_amount = request.form.get("line-amount", default=100, type=int)
    epoch_amount = request.form.get("epoch-amount", default=100, type=int)

    col_values = []
    for i in range(len(bool_values)):
        col_values.append((bool_values[i]))

    times = 0
    for index in range(len(ignore_zero_values), len(ignore_zero_values) * 2):
        col_values.append((ignore_zero_values[times]))
        times += 1
    times = 0
    for index in range(len(exclude_columns) * 2, len(exclude_columns) * 3):
        col_values.append((exclude_columns[times]))
        times += 1
    times = 0

    processed_col_values = []
    for i in range(len(col_values)):
        if col_values[i] == "0" and i + 1 < len(col_values) and col_values[i + 1] == "1":
            continue
        processed_col_values.append(col_values[i])

    output_folder = f'Files/downloads/{session_id}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # hide blocks here
    session['col_categories'] = []

    csv_filename = session['file_name']
    csv_filename = os.path.splitext(csv_filename)[0] + ".csv"

    output_file = generate_file(processed_col_values, line_amount, epoch_amount, csv_filename, session_id)
    session['output_file'] = output_file


    return redirect(url_for('review'))


@app.route('/download/<output_file>')
def download_file(output_file):
    session_id = session.get('session_id', '')
    return send_from_directory(f'Files/downloads/{session_id}', output_file, as_attachment=True)

@app.route('/review')
def review():
    session_id = session.get('session_id', '')
    picture = correlation(session_id)
    return render_template("dataReview.html", picture=picture, session_id=session_id)

@app.route('/delete')
def delete():
    session_id = session.get('session_id', '')
    if session_id:
        upload_folder = f'Files/uploads/{session_id}'
        if os.path.exists(upload_folder):
            shutil.rmtree(upload_folder, ignore_errors=True)

        output_folder = f'Files/downloads/{session_id}'
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder, ignore_errors=True)

        image_folder = f'static/images/{session_id}'
        if os.path.exists(image_folder):
            for filename in os.listdir(image_folder):
                file_path = os.path.join(image_folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        session.clear()
    return redirect("/")

app.run(debug=True, port=5001, host='0.0.0.0')