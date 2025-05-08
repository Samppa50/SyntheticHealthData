from flask import Flask, render_template, url_for, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from synthetic import main
import os

app = Flask(__name__)

output_file = "nan"

@app.route("/")
def index():
   return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files['file']
        if file:
            upload_folder = 'Files/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, secure_filename(file.filename))
            file.save(file_path)

            output_file = main(file.filename)
            return redirect(f"/download/{output_file}")
        else:
            return "No file uploaded", 400
    return "Invalid request method", 405


@app.route(f'/download/<output_file>')
def download_file(output_file):
    return send_from_directory('Files/downloads', output_file, as_attachment=True)

app.run(debug=True, port=5001, host='0.0.0.0')