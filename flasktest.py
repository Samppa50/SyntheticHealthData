from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route("/")

def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if request.method == "POST":
        file = request.files['file']
        file.save(f'uploads/{secure_filename(file.filename)}')
        return "File uploaded successfully"
    else:
        return "File upload failed"


app.run(debug=True)