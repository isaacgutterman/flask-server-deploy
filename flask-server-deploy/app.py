import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import MSTcoord

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = os.path.join('/tmp', file.filename)
        file.save(filename)
        return jsonify({"message": "File uploaded successfully", "filename": filename}), 200

@app.route('/process', methods=['POST'])
def process_data():
    filepath = request.json['filepath']
    max_cluster_depth = request.json.get('max_cluster_depth', 10)
    min_nodes = request.json.get('min_nodes', 5)
    message = process_csv(filepath, max_cluster_depth, min_nodes)
    return jsonify({"message": message})

def process_csv(csv_file, max_cluster_depth, min_nodes):
    cluster_creator = MSTcoord.ClusterCreator(max_cluster_depth, min_nodes)
    cluster_creator.load_skills_data_from_csv(csv_file)
    cluster_creator.make_clusters()
    return "Processing completed successfully"

if __name__ == '__main__':
    app.run(debug=True)
