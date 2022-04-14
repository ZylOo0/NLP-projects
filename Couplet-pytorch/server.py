from flask import Flask, render_template, request, url_for
import torch
import os

from utils.config import parseArgs
from infer import Inferer

app = Flask(__name__)
app.config["SECRET_KEY"] = "12345"
base_dir = os.environ.get("BASE_DIR", "")

args = parseArgs()
inferer = Inferer(args)


@app.route(f"{base_dir}/v1/index", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html")
    if request.method == "POST":
        if request.form["sentence"] == "":
            return ""
        result = inferer.infer(request.form["sentence"])
        return result

if __name__ == "__main__":
    app.run("0.0.0.0")
