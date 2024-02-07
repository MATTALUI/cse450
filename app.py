from flask import Flask, send_from_directory
import os
import pandas as pd
from scripts.utils import BaseReport
# from scripts.pokemon import report as poke_report
import json
import glob

pokemon = pd.read_csv("./datasets/pokemon.csv")
poke_report = BaseReport(
    pokemon,
    skip_features=["number", "generation"],
)
bank = pd.read_csv("./datasets/bank.csv")
bank_report = BaseReport(
    bank,
)

app = Flask(__name__)

@app.get("/")
def send_root():
    return send_from_directory("static", "index.html")

@app.get("/<path:path>")
def send_static(path):
    return send_from_directory("static", path)

@app.get("/api/test")
def test():
    return "109 in the sky, but the pigs won't quit"

@app.get("/api/pokemon/splot/")
def get_pokemon_splot_data():
    if not os.path.exists("./static/tmp/pokemon/splot"):
        poke_report.write_splot("../static/tmp/pokemon/splot") # from other tmp
    splot_images = {}
    for x_label in poke_report.continuous_features:
        row = []
        for y_label in poke_report.continuous_features:
            img_path = os.path.join("/tmp/pokemon/splot", f"SPLOT-{x_label}-{y_label}.png")
            row.append({ y_label: img_path })
        splot_images[x_label] = row
        row = []
    return json.dumps(splot_images)

@app.get("/api/bank/splot/")
def get_bank_splot_data():
    if not os.path.exists("./static/tmp/bank/splot"):
        bank_report.write_splot("../static/tmp/bank/splot") # from other tmp
    splot_images = {}
    for x_label in bank_report.continuous_features:
        row = []
        for y_label in bank_report.continuous_features:
            img_path = os.path.join("/tmp/bank/splot", f"SPLOT-{x_label}-{y_label}.png")
            row.append({ y_label: img_path })
        splot_images[x_label] = row
        row = []
    return json.dumps(splot_images)

@app.get("/api/datasets")
def get_datasets():
    ds = [f.split("/")[2].split(".")[0] for f in glob.glob("./datasets/*.csv")]

    return json.dumps(ds)

@app.get("/api/datasets/<set_name>/features")
def get_features(set_name):
    features = []
    df = pd.read_csv(f"./datasets/{set_name}.csv")
    for col in df.columns:
        features.append(col)
    print(features)
    return json.dumps(features)

@app.get("/api/datasets/<set_name>/head")
def get_head(set_name):
    df = pd.read_csv(f"./datasets/{set_name}.csv")

    return json.dumps(df.head().to_json())

@app.get("/api/datasets/<set_name>/dqr")
def get_dqr(set_name):
    df = pd.read_csv(f"./datasets/{set_name}.csv")
    br = BaseReport(df)

    return json.dumps({
        "continuous": br.continuouse_quality_report.to_json(),
        "categorical": br.categorical_quality_report.to_json(),
    })