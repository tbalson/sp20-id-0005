import os
from flask import jsonify
import connexion
from flask import Flask, render_template, request

print(os.getcwd())
app = connexion.App(__name__, specification_dir="./")

app.add_api("master.yaml")

@app.route("/")
def home():
    msg = {"msg": "Test basepath to see if it's working!"}
    return jsonify(msg)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
