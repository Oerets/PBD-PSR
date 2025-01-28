from flask import Flask, request, jsonify
from print import changeinput

app = Flask(__name__)

@app.route('/echo', methods=['POST'])
def echo():
    # 요청으로부터 JSON 데이터 받기
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    print(data)

    # 데이터를 그대로 반환
    response = {
        "received_data": data,
        "message": "Data received successfully!"
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)