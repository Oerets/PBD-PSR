from flask import Flask, request, jsonify, Response
from bmd_analysis import bmd_analysis
import json
import sys
import time  # 타임 딜레이 테스트용

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    settings_data = request.json
    if not settings_data:
        return jsonify({"error": "No data provided"}), 400    

    def generate_analysis():
        # 🔹 초기 메시지를 즉시 클라이언트에 전송
        yield f"data: {json.dumps({'status': 'Processing started'})}\n\n"
        sys.stdout.flush()  # 강제 푸시

        # 🔹 진행 상태를 실시간으로 전송
        for status_update in bmd_analysis(settings_data):
            yield f"data: {json.dumps(status_update)}\n\n"
            sys.stdout.flush()  # 강제 푸시
            
    return Response(generate_analysis(), content_type='text/event-stream')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)  # 🔹 threaded=True 추가