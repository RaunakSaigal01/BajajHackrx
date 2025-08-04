from flask import Flask, request, jsonify
from check import answer_from_pdf_url 

app = Flask(__name__)

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    try:
        data = request.get_json()

        if not data or "documents" not in data or "questions" not in data:
            return jsonify({"error": "Missing 'documents' or 'questions' in request."}), 400

        pdf_url = data["documents"]
        questions = data["questions"]

        answers = []
        for question in questions:
            answer = answer_from_pdf_url(pdf_url, question)
            answers.append(answer)

        return jsonify({"answers": answers}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
