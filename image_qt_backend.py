from flask import Flask, request, jsonify, Response
from bmd_analysis import bmd_analysis
import os
import json

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    mode = data.get('mode')
    weighted_mode = data.get('weighted_mode')
    det_model_path = data.get('det_model_path')
    det_model_name = data.get('det_model_name')
    reg_model_path = data.get('reg_model_path')
    reg_model_name = data.get('reg_model_name')
    data_path = data.get('data_path')
    excel_path = data.get('excel_path')
    dicom_path = data.get('dicom_path')
    z_threshold = data.get('z_threshold')

    def generate():
        for status_update in bmd_analysis(
            mode=mode,
            weighted_mode=weighted_mode,
            det_model_path=det_model_path,
            det_model_name=det_model_name,
            reg_model_path=reg_model_path,
            reg_model_name=reg_model_name,
            data_path=data_path,
            excel_path=excel_path,
            dicom_path=dicom_path,
            z_threshold=z_threshold
        ):
            yield f"data:{json.dumps(status_update)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
