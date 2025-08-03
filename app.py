import os
import requests
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from tempfile import NamedTemporaryFile

from check import improved_chunking, build_faiss_index, find_answer

app = Flask(__name__)

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    try:
        data = request.get_json()

        # Validate input
        if not data or "documents" not in data or "questions" not in data:
            return jsonify({"error": "Missing 'documents' or 'questions' in request."}), 400

        pdf_url = data["documents"]
        questions = data["questions"]

        # Step 1: Download PDF
        response = requests.get(pdf_url)
        response.raise_for_status()

        # Step 2: Save PDF to temp file
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            tmp_pdf.write(response.content)
            tmp_pdf_path = tmp_pdf.name

        # Step 3: Extract text
        reader = PdfReader(tmp_pdf_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        # Step 4: Chunk and check if valid
        chunks = improved_chunking(full_text)
        if not chunks:
            os.remove(tmp_pdf_path)
            return jsonify({"error": "No readable text found in PDF."}), 400

        # Step 5: Build FAISS index in memory
        index, _ = build_faiss_index(chunks)

        # Step 6: Answer each question
        answers = [find_answer(q, index, chunks) for q in questions]

        # Step 7: Cleanup
        os.remove(tmp_pdf_path)

        # Step 8: Return response
        return jsonify({"answers": answers}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
