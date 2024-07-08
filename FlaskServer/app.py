from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from MSTcoord import ClusterCreator

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_data():
    data = request.json
    csv_file = data['csv_file']
    max_cluster_depth = data['max_cluster_depth']
    min_nodes = data['min_nodes']
    message = process_csv(csv_file, max_cluster_depth, min_nodes)
    return jsonify({"message": message})

def process_csv(csv_file, max_cluster_depth, min_nodes):
    cluster_creator = ClusterCreator(max_cluster_depth, min_nodes)
    cluster_creator.load_skills_data_from_csv(csv_file)
    cluster_creator.make_clusters()
    cluster_creator.create_connections()
    cluster_creator.save_mst_to_csv("MST_coord.csv")
    cluster_creator.save_to_files()
    return "Processing complete and files saved."

if __name__ == '__main__':
    app.run(debug=True)
