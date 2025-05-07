from flask import Flask, render_template, url_for, request, redirect
from werkzeug.utils import secure_filename
from synthetic import main
import os

app = Flask(__name__)


@app.route("/")
def index():
   return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files['file']
        if file:
            upload_folder = 'SyntheticHealthData/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, secure_filename(file.filename))
            file.save(file_path)

            output_file = main(file.filename)
            
            output_file.save("SyntheticHealthData/downloads")
            
            return redirect("/download")
        else:
            return "No file uploaded", 400
    return "Invalid request method", 405

@app.route("/download")
def download():
    return render_template("download.html", )

app.run(debug=True, port=5001, host='0.0.0.0')