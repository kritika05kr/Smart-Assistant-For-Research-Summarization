import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import google.generativeai as genai

# -------------------------------
# SMART RESEARCH SUMMARIZER API
# -------------------------------
#
# This Flask app:
# - Uploads PDF or TXT documents
# - Extracts their text
# - Summarizes documents using Gemini AI
# - Lets users ask questions about uploaded documents
# - Generates comprehension questions based on documents
# - Evaluates user answers to generated questions
#
# Gemini AI is used for summarization, Q&A, and evaluations.
# -------------------------------

# --- CONFIGURATION ---

# Your Gemini API key goes here.
API_KEY = ""  # TODO: add your Gemini API key here
genai.configure(api_key=API_KEY)

# Initialize Flask app
app = Flask(__name__)
# Enable CORS so frontend apps can connect without cross-origin errors
CORS(app)

# Directory where uploaded documents are stored
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Flask configuration settings
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload = 16MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt'}


# -------------------------------
# Helper Functions
# -------------------------------

def allowed_file(filename):
    """
    Check whether a file's extension is allowed (PDF or TXT).

    Parameters:
        filename (str): The name of the file.

    Returns:
        bool: True if allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file. Annotates each extracted line
    with page and line numbers for traceability.

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        str or None: The extracted text, or None if error.
    """
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
    """
    Extract text from a plain TXT file. Annotates each line with
    its line number.

    Parameters:
        txt_path (str): Path to the text file.

    Returns:
        str or None: The extracted text, or None if error.
    """
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


def get_gemini_response(prompt, generation_config=None):
    """
    Send a prompt to Gemini AI and return the response text.

    Parameters:
        prompt (str): The prompt to send.
        generation_config (dict, optional): Any config for structured outputs.

    Returns:
        str or None: The response text from Gemini AI.
    """
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


# Dictionary to store extracted document content
# Key = filename, Value = extracted text
document_content = {}


# -------------------------------
# API ROUTES
# -------------------------------

@app.route('/upload', methods=['POST'])
def upload_document():
    """
    POST /upload

    Uploads a document (PDF or TXT) to the server.

    Steps:
    - Save uploaded file to disk.
    - Extract its text content.
    - Send text to Gemini AI for summarization.
    - Return JSON with summary.

    Request:
        Form-data:
            file: (binary file) PDF or TXT

    Response:
        {
            "message": "File uploaded and summarized successfully",
            "summary": "...summary text..."
        }
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract text based on file type
        extracted_text = None
        if filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(filepath)
        elif filename.lower().endswith('.txt'):
            extracted_text = extract_text_from_txt(filepath)

        if not extracted_text:
            return jsonify({'error': 'Failed to extract text from document'}), 500

        # Save extracted text in memory
        document_content[filename] = extracted_text

        # Generate summary from Gemini
        summary_prompt = (
            f"Summarize the following document in no more than 150 words. "
            f"Focus on the main points. Document content: {extracted_text}"
        )
        summary = get_gemini_response(summary_prompt)

        if summary:
            return jsonify({
                'message': 'File uploaded and summarized successfully',
                'summary': summary
            }), 200
        else:
            return jsonify({'error': 'Failed to generate summary'}), 500
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/ask', methods=['POST'])
def ask_question():
    """
    POST /ask

    Allows users to ask a question about an uploaded document.

    Steps:
    - Accepts question and document name.
    - Sends prompt to Gemini AI to answer using document content.
    - Returns answer with references to document lines.

    Request:
        {
            "query": "What is this document about?",
            "documentName": "filename.pdf"
        }

    Response:
        {
            "answer": "..."
        }
    """
    data = request.get_json()
    query = data.get('query')
    document_name = data.get('documentName')

    if not query or not document_name:
        return jsonify({'error': 'Missing query or document name'}), 400

    doc_text = document_content.get(document_name)
    if not doc_text:
        return jsonify({
            'error': 'Document content not found. Please upload the document again.'
        }), 404

    prompt = (
        f"Based on the following document, answer the question: '{query}'. "
        f"Provide justification using [Page X, Line Y] or [Line Y] markers.\n\n"
        f"Document content: {doc_text}"
    )
    answer = get_gemini_response(prompt)

    if answer:
        return jsonify({'answer': answer}), 200
    else:
        return jsonify({'error': 'Failed to get an answer from the assistant'}), 500


@app.route('/challenge', methods=['POST'])
def generate_challenge_questions():
    """
    POST /challenge

    Generates three challenge questions for comprehension or logic
    based on the uploaded document.

    Steps:
    - Accepts document name.
    - Prompts Gemini to generate questions in JSON array format.
    - Returns the array of questions.

    Request:
        {
            "documentName": "filename.pdf"
        }

    Response:
        {
            "questions": [
                "Question 1 ...",
                "Question 2 ...",
                "Question 3 ..."
            ]
        }
    """
    data = request.get_json()
    document_name = data.get('documentName')

    if not document_name:
        return jsonify({'error': 'Missing document name'}), 400

    doc_text = document_content.get(document_name)
    if not doc_text:
        return jsonify({'error': 'Document content not found.'}), 404

    prompt = (
        f"Generate three distinct logic-based or comprehension-focused questions "
        f"based on the following document. Provide them as a JSON array: "
        f"[\"Q1\", \"Q2\", \"Q3\"]\n\nDocument content: {doc_text}"
    )

    # Tell Gemini to respond in JSON
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
    """
    POST /evaluate_challenge

    Evaluates user answers to the challenge questions.

    Steps:
    - Accepts document name, questions, and user answers.
    - Prompts Gemini to check if each answer is correct and provide justification.
    - Returns feedback for each question.

    Request:
        {
            "documentName": "filename.pdf",
            "questions": ["Q1", "Q2", "Q3"],
            "userAnswers": {
                "0": "Answer to Q1",
                "1": "Answer to Q2",
                "2": "Answer to Q3"
            }
        }

    Response:
        {
            "feedback": {
                "0": "...evaluation text...",
                "1": "...",
                "2": "..."
            }
        }
    """
    data = request.get_json()
    document_name = data.get('documentName')
    questions = data.get('questions')
    user_answers = data.get('userAnswers')

    if not document_name or not questions or not user_answers:
        return jsonify({'error': 'Missing data'}), 400

    doc_text = document_content.get(document_name)
    if not doc_text:
        return jsonify({'error': 'Document content not found.'}), 404

    feedback = {}
    for index, question in enumerate(questions):
        user_answer = user_answers.get(str(index), '')
        prompt = (
            f"Based on the document, evaluate if this answer: '{user_answer}' "
            f"is correct for the question: '{question}'. Give a verdict "
            f"(Correct/Partially Correct/Incorrect) and justify using [Page X, Line Y] or [Line Y] markers.\n\n"
            f"Document content: {doc_text}"
        )
        evaluation = get_gemini_response(prompt)
        feedback[index] = evaluation or "Evaluation not available."

    return jsonify({'feedback': feedback}), 200


# Run the Flask server if script is executed directly
if __name__ == '__main__':
    app.run(debug=True, port=5000)
