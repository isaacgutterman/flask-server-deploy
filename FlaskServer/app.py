from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from MSTcoord import ClusterCreator
import os
app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webgl/<path:filename>')
def serve_webgl(filename):
    return send_from_directory('static/webgl', filename)

@app.route('/process', methods=['POST'])
def process_data():
    if 'csv_file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400
    file = request.files['csv_file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join('/tmp', file.filename)
        file.save(filepath)
        max_cluster_depth = 2
        min_nodes = 10
        message = process_csv(filepath, max_cluster_depth, min_nodes)
        return jsonify({"message": message})
    else:
        return jsonify({"message": "Invalid file type, please upload a CSV file"}), 400

def process_csv(csv_file, max_cluster_depth, min_nodes):
    cluster_creator = ClusterCreator(max_cluster_depth, min_nodes)
    cluster_creator.load_skills_data_from_csv(csv_file)
    cluster_creator.make_clusters()
    cluster_creator.create_connections()
    cluster_creator.save_mst_to_csv("/tmp/MST_coord.csv")
    cluster_creator.save_to_files()
    return "Processing complete and files saved."

if __name__ == '__main__':
    app.run(debug=True)
