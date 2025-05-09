from flask import Flask, render_template, url_for, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from synthetic import main, generate_file
import os

app = Flask(__name__)

output_file = "nan"
col_categories = []
file_name = ""

@app.route("/")
def index():
    global col_categories  # Ensure the global variable is used
    return render_template("index.html", items=col_categories)

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    global col_categories  # Update the global variable
    if request.method == "POST":
        file = request.files['file']
        if file:
            upload_folder = 'Files/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, secure_filename(file.filename)) # Save the file to the upload folder
            file.save(file_path)
            global file_name
            file_name = file.filename

            global col_names
            col_names = main(file.filename)
            col_categories = col_names  # Update the global list with new column names

            global col_amount
            col_amount = len(col_names)
            #tätä ehkä tarvitaan myöhemmin

            return redirect("/")
        else:
            return "No file uploaded", 400
    return "Invalid request method", 405

@app.route("/submit", methods=["GET", "POST"])
def submit():
    global output_file

    bool_values = request.form.to_dict(flat=False).get("bool_values", {})
    ignore_zero_values = request.form.to_dict(flat=False).get("ignore_zero_values", {})
    line_amount = request.form.get("amount", default=100, type=int)

    print("lines wanted:", line_amount)
    col_values = []

    for i in range(len(bool_values)):
        col_values.append((bool_values[i]))

    times = 0
    for index in range(len(ignore_zero_values),len(ignore_zero_values) * 2):
        col_values.append((ignore_zero_values[times]))
        times += 1
    times = 0


    processed_col_values = []
    for i in range(len(col_values)):
        if col_values[i] == "0" and i + 1 < len(col_values) and col_values[i + 1] == "1":
            continue  # Skip the "off" if followed by "on"
        processed_col_values.append(col_values[i])

    print("Processed col_values:", processed_col_values)

    output_file = generate_file(processed_col_values, line_amount ,file_name)
    return redirect(f"/download/{output_file}")


@app.route(f'/download/<output_file>')
def download_file(output_file):
    return send_from_directory('Files/downloads', output_file, as_attachment=True)

app.run(debug=True, port=5001, host='0.0.0.0')