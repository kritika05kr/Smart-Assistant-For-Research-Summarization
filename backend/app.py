import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import google.generativeai as genai

# --- Configuration ---
API_KEY = "AIzaSyB-8SjBKkb3aEjGmNcXTacs5GShzAgy-4U"  # Add your Gemini API Key here
genai.configure(api_key=API_KEY)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                lines = page_text.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        text += f"[Page {page_num + 1}, Line {i + 1}] {line}\n"
                    else:
                        text += "\n"
        print(f"Extracted PDF text length: {len(text)}")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text

def extract_text_from_txt(txt_path):
    text = ""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if line.strip():
                    text += f"[Line {i + 1}] {line}"
                else:
                    text += "\n"
        print(f"Extracted TXT text length: {len(text)}")
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return None
    return text

document_content = {}

def get_gemini_response(prompt, generation_config=None):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        if generation_config:
            response = model.generate_content(prompt, generation_config=generation_config)
        else:
            response = model.generate_content(prompt)
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        return None
    except Exception as e:
        print(f"Error interacting with Gemini API: {e}")
        return None

@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        extracted_text = None
        if filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(filepath)
        elif filename.lower().endswith('.txt'):
            extracted_text = extract_text_from_txt(filepath)

        if not extracted_text:
            return jsonify({'error': 'Failed to extract text from document'}), 500

        document_content[filename] = extracted_text
        summary_prompt = f"Summarize the following document in no more than 150 words. Focus on the main points. Document content: {extracted_text}"
        summary = get_gemini_response(summary_prompt)

        if summary:
            return jsonify({'message': 'File uploaded and summarized successfully', 'summary': summary}), 200
        else:
            return jsonify({'error': 'Failed to generate summary'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('query')
    document_name = data.get('documentName')

    if not query or not document_name:
        return jsonify({'error': 'Missing query or document name'}), 400

    doc_text = document_content.get(document_name)
    if not doc_text:
        return jsonify({'error': 'Document content not found. Please upload the document again.'}), 404

    prompt = f"Based on the following document, answer the question: '{query}'. Provide justification using [Page X, Line Y] or [Line Y] markers.\n\nDocument content: {doc_text}"
    answer = get_gemini_response(prompt)

    if answer:
        return jsonify({'answer': answer}), 200
    else:
        return jsonify({'error': 'Failed to get an answer from the assistant'}), 500

@app.route('/challenge', methods=['POST'])
def generate_challenge_questions():
    data = request.get_json()
    document_name = data.get('documentName')

    if not document_name:
        return jsonify({'error': 'Missing document name'}), 400

    doc_text = document_content.get(document_name)
    if not doc_text:
        return jsonify({'error': 'Document content not found.'}), 404

    prompt = f"Generate three distinct logic-based or comprehension-focused questions based on the following document. Provide them as a JSON array: [\"Q1\", \"Q2\", \"Q3\"]\n\nDocument content: {doc_text}"

    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        }
    }

    questions_json_str = get_gemini_response(prompt, generation_config=generation_config)

    if questions_json_str:
        try:
            questions = json.loads(questions_json_str)
            return jsonify({'questions': questions}), 200
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from LLM: {questions_json_str}")
            return jsonify({'error': 'Invalid JSON from assistant'}), 500
    else:
        return jsonify({'error': 'Failed to generate challenge questions'}), 500

@app.route('/evaluate_challenge', methods=['POST'])
def evaluate_challenge_answers():
    data = request.get_json()
    document_name = data.get('documentName')
    questions = data.get('questions')
    user_answers = data.get('userAnswers')

    if not document_name or not questions or not user_answers:
        return jsonify({'error': 'Missing data'}), 400

    doc_text = document_content.get(document_name)
    if not doc_text:
        return jsonify({'error': 'Document content not found'}), 404

    feedback = {}
    for index, question in enumerate(questions):
        user_answer = user_answers.get(str(index), '')
        prompt = f"Based on the document, evaluate if this answer: '{user_answer}' is correct for the question: '{question}'. Give a verdict (Correct/Partially Correct/Incorrect) and justify using [Page X, Line Y] or [Line Y] markers.\n\nDocument content: {doc_text}"
        evaluation = get_gemini_response(prompt)
        feedback[index] = evaluation or "Evaluation not available."

    return jsonify({'feedback': feedback}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
