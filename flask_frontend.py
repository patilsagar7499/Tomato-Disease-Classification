from flask import Flask, render_template

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_file():    
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
