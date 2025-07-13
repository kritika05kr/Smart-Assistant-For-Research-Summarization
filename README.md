## Smart Research Assistant
This project develops an AI assistant capable of understanding and reasoning through user-uploaded documents (PDF or TXT format). It goes beyond basic summarization and keyword search by providing contextual question answering and logic-based question generation, with all responses justified by document references.

---

### Features
---
- Document Upload: Upload PDF or TXT documents.

- Auto Summary: Get a concise summary (≤ 150 words) immediately after upload.

- Ask Anything Mode: Ask free-form questions about the document and receive justified answers.

- Challenge Me Mode: Generate logic-based/comprehension questions from the document, answer them, and get evaluated feedback with justifications.

- Contextual Understanding: All answers are strictly grounded in the document content, minimizing hallucination.

- Justification: Every AI response includes a reference to the source document.

- Clean UI/UX: Intuitive web-based interface for seamless interaction.

---
### Getting Started
---
Follow these instructions to set up and run the project on your local machine.

#### Prerequisites
Before you begin, ensure you have the following installed:

### Prerequisites

Before you begin, ensure you have the following installed:

- [Git](https://git-scm.com/downloads): For cloning the repository.
- [Python 3.8+](https://www.python.org/downloads/): For the backend.
- [Node.js & npm](https://nodejs.org/en/download/): For the React frontend.
- [Google Gemini API Key](https://aistudio.google.com/app/u/1/apikey): Required for AI functionalities.


---

### Project Structure
---
```smart-assistant/
├── backend/
│   ├── app.py           # Flask backend application
│   └── uploads/         # Directory for uploaded documents
│   └── requirements.txt # Python dependencies for backend
├── frontend/
│   ├── public/          # Public assets for React app (e.g., index.html)
│   │   └── index.html
│   ├── src/             # React source code
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json     # Node.js/React project configuration
│   └── .env             # Environment variables for frontend (optional, but good practice)
└── README.md            # Project overview and instructions
└── .gitignore           # Specifies files and directories to be ignored by Git

```

### Setup Instructions
---
- Clone the Repository:
    - Open your terminal or command prompt and run:

    ```
    git clone https://github.com/uk-zash/Smart-Research-Assistant
    ```
    ```
    cd smart-research-assistant
    ```
---
### Backend Setup:
---
- Navigate to the backend directory:

```
cd backend
```

- Create and activate a Python virtual environment:

- macOS/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

- Windows (Command Prompt):

```
python -m venv venv
.\venv\Scripts\activate
```

- Windows (PowerShell):

```
python -m venv venv
.\venv\Scripts\Activate.ps1
```

- Install the required Python packages:

```
pip install Flask Flask-Cors PyPDF2 google-generativeai grpcio
```
### Or: 
```
pip install -r requirements.txt
```


## Configure your Google Gemini API Key:
### Open backend/app.py. Find the line 
```
API_KEY = "" 
```
and replace the empty string with your actual Gemini API key.


```
API_KEY = "YOUR_GEMINI_API_KEY_HERE" # Replace with your actual key
genai.configure(api_key=API_KEY)
```

### Note: In a production environment, it's recommended to use environment variables for API keys for security.
---
### Frontend Setup:
---

Navigate back to the project root and then into the frontend directory:
```
cd ../frontend
```

Install the Node.js 
dependencies:
```
npm install
```

### Important: Update public/index.html
Ensure your frontend/public/index.html file includes the Tailwind CSS CDN and Google Fonts link in the <head> section. This is crucial for styling.

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Research Assistant</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Inter font -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet" />
</head>
<body>
    <div id="root"></div>
</body>
</html>

Make sure these lines are removed from frontend/src/App.js if they are present there.
---


### Running the Application
---
You will need two separate terminal windows/tabs to run the backend and frontend concurrently.

#### Start the Backend Server:
---
In your first terminal, ensure you are in the backend directory and your virtual environment is activated.

Run the Flask application:
```
python app.py
```

The backend server will typically run on http://127.0.0.1:5000/.

#### Start the Frontend Development Server:
---
In your second terminal, navigate to the frontend directory.

Start the React development server:
```
npm start
```

This will usually open the application in your default web browser at http://localhost:3000/.

### Architecture / Reasoning Flow
The Smart Research Assistant follows a client-server architecture:

- Frontend (React): Provides the user interface for document upload, displaying summaries, and interacting via chat or challenge modes. It communicates with the backend via RESTful API calls.

- Backend (Flask): Handles core logic, including:

    - Document Processing: Receives PDF/TXT files, extracts text using PyPDF2, and stores the content in memory (for this demo). In a production environment, this would involve more robust storage and potentially vector embeddings.

    - AI Interaction: Integrates with the Google Gemini API (gemini-2.0-flash) to perform:

        - Summarization: Generates concise summaries of uploaded documents.

        - Question Answering: Answers free-form questions by querying the LLM with the document content as context.

        - Challenge Generation: Creates logic-based questions from the document.

        - Answer Evaluation: Assesses user answers to challenge questions, providing feedback and justification.

    - Justification & Context: Prompts are carefully engineered to ensure the LLM's responses are grounded in the provided document content and include explicit justifications (e.g., "This is supported by...").

    - API Endpoints: Exposes endpoints for file upload, asking questions, generating challenges, and evaluating answers.

### Evaluation Criteria Alignment
This project is designed with the following evaluation criteria in mind:

- Response Quality (Accuracy + Justification): Achieved through careful prompt engineering to ground LLM responses in the document and explicit inclusion of justifications.

- Reasoning Mode Functionality: Implemented via the "Challenge Me" mode, which generates logic-based questions and evaluates user answers with detailed feedback.

- UI/UX and Smooth Flow: Provided by a clean, responsive React frontend with clear interaction modes and loading indicators.

- Code Structure & Documentation: Modular backend and frontend codebases, along with this comprehensive README.md, ensure readability and maintainability.

- Creativity / Bonus Features: (Future enhancements could include memory handling for follow-up questions or direct snippet highlighting).

- Minimal Hallucination & Good Context Use: A core focus of the LLM prompting strategy, ensuring responses are strictly document-based.

```
Feel free to contribute or suggest improvements!
```