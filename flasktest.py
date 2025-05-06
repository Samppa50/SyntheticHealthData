from flask import Flask, render_template, url_for, request, redirect
from werkzeug.utils import secure_filename
import sythetic
from synthetic import *

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files['file']
    if file:
        file.save(f'uploads/{secure_filename(file.filename)}')
        file.filename
        return redirect('/')
    else:
        return print("No file uploaded")


app.run(debug=True)