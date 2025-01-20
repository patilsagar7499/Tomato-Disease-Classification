from flask import Flask, render_template, request
import requests

app = Flask(__name__)

FASTAPI_URL = "http://localhost:8000/predict"

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            response = requests.post(FASTAPI_URL, files={"file": file})
            prediction = response.json()
            return render_template("index.html", prediction=prediction)
    
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
