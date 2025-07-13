import React, { useState, useEffect, useRef } from 'react';

// Main App component
const App = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [summary, setSummary] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [userInput, setUserInput] = useState('');
    const [mode, setMode] = useState('askAnything'); // 'askAnything' or 'challengeMe'
    const [challengeQuestions, setChallengeQuestions] = useState([]);
    const [challengeAnswers, setChallengeAnswers] = useState({});
    const [challengeFeedback, setChallengeFeedback] = useState({});
    const [isLoading, setIsLoading] = useState(false);
    const [message, setMessage] = useState(''); // For user messages/alerts
    const chatContainerRef = useRef(null); // Ref for scrolling chat to bottom

    const API_BASE_URL = 'http://127.0.0.1:5000'; // Your Flask backend URL

    // Effect to scroll chat to the bottom on new messages
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chatHistory]);

    // Function to handle file selection
    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]); // Corrected: get the first file
        setSummary('');
        setChatHistory([]); // Corrected: clear chat history
        setMessage('');
        setChallengeQuestions([]); // Corrected: clear challenge questions
        setChallengeAnswers({});
        setChallengeFeedback({});
    };

    // Function to handle document upload
    const handleUpload = async () => {
        if (!selectedFile) {
            setMessage('Please select a file first.');
            return;
        }

        setIsLoading(true);
        setMessage('Uploading document and generating summary...');

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();

            if (response.ok) {
                setSummary(data.summary);
                setMessage('Document uploaded and summarized successfully!');
                setChatHistory([]); // Corrected: clear chat history after new upload
            } else {
                setMessage(`Error: ${data.error || 'Failed to upload document.'}`); // Corrected syntax
                console.error('Upload error:', data);
            }

        } catch (error) {
            console.error('Error uploading file or getting summary:', error);
            setMessage('Error uploading file or generating summary. Please check console.');
        } finally {
            setIsLoading(false);
        }
    };

    // Function to handle sending a message in "Ask Anything" mode
    const handleSendMessage = async () => {
        if (!userInput.trim()) return;
        if (!selectedFile) {
            setMessage('Please upload a document first to ask questions.');
            return;
        }

        const newUserMessage = { role: 'user', text: userInput };
        setChatHistory((prev) => [...prev, newUserMessage]);
        setUserInput('');
        setIsLoading(true);
        setMessage('Assistant is typing...');

        try {
            const response = await fetch(`${API_BASE_URL}/ask`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: newUserMessage.text, documentName: selectedFile.name }),
            });
            const data = await response.json();

            if (response.ok) {
                setChatHistory((prev) => [...prev, { role: 'assistant', text: data.answer }]);
                setMessage('');
            } else {
                setMessage(`Error: ${data.error || 'Failed to get an answer.'}`); // Corrected syntax
                console.error('Ask error:', data);
            }

        } catch (error) {
            console.error('Error sending message:', error);
            setMessage('Error getting answer. Please check console.');
        } finally {
            setIsLoading(false);
        }
    };

    // Function to handle generating challenge questions
    const handleGenerateChallenge = async () => {
        if (!selectedFile) {
            setMessage('Please upload a document first to generate challenges.');
            return;
        }

        setIsLoading(true);
        setMessage('Generating challenge questions...');
        setChallengeQuestions([]); // Corrected: clear previous questions
        setChallengeAnswers({});
        setChallengeFeedback({});

        try {
            const response = await fetch(`${API_BASE_URL}/challenge`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ documentName: selectedFile.name }),
            });
            const data = await response.json();

            if (response.ok) {
                setChallengeQuestions(data.questions);
                setMessage('Challenge questions generated!');
            } else {
                setMessage(`Error: ${data.error || 'Failed to generate challenge questions.'}`); // Corrected syntax
                console.error('Challenge generation error:', data);
            }

        } catch (error) {
            console.error('Error generating challenge questions:', error);
            setMessage('Error generating challenge questions. Please check console.');
        } finally {
            setIsLoading(false);
        }
    };

    // Function to handle user input for challenge answers
    const handleChallengeAnswerChange = (index, value) => {
        setChallengeAnswers((prev) => ({...prev, [index]: value }));
    };

    // Function to handle evaluating challenge answers
    const handleSubmitChallengeAnswers = async () => {
        if (Object.keys(challengeAnswers).length === 0 || Object.values(challengeAnswers).every(answer => !answer.trim())) { // Corrected syntax
            setMessage('Please provide answers to the challenge questions.');
            return;
        }

        setIsLoading(true);
        setMessage('Evaluating your answers...');
        setChallengeFeedback({});

        try {
            const response = await fetch(`${API_BASE_URL}/evaluate_challenge`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    documentName: selectedFile.name,
                    questions: challengeQuestions,
                    userAnswers: challengeAnswers,
                }),
            });
            const data = await response.json();

            if (response.ok) {
                setChallengeFeedback(data.feedback);
                setMessage('Answers evaluated!');
            } else {
                setMessage(`Error: ${data.error || 'Failed to evaluate answers.'}`); // Corrected syntax
                console.error('Evaluation error:', data);
            }

        } catch (error) {
            console.error('Error evaluating challenge answers:', error);
            setMessage('Error evaluating answers. Please check console.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 font-sans text-gray-800 p-4 sm:p-6 lg:p-8 flex flex-col items-center">
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet" />
            <script src="https://cdn.tailwindcss.com"></script>

            <style>
                {`
                body {
                    font-family: 'Inter', sans-serif;
                }
                /* Custom scrollbar for chat history */
               .chat-history-scroll::-webkit-scrollbar {
                    width: 8px;
                }
               .chat-history-scroll::-webkit-scrollbar-track {
                    background: #f1f1f1;
                    border-radius: 10px;
                }
               .chat-history-scroll::-webkit-scrollbar-thumb {
                    background: #cbd5e1;
                    border-radius: 10px;
                }
               .chat-history-scroll::-webkit-scrollbar-thumb:hover {
                    background: #94a3b8;
                }
                `}
            </style>

            <div className="w-full max-w-4xl bg-white rounded-xl shadow-2xl p-6 sm:p-8 lg:p-10 flex flex-col gap-6">
                <h1 className="text-4xl font-extrabold text-center text-indigo-700 mb-4">
                    Smart Research Assistant
                </h1>

                {/* File Upload Section */}
                <div className="bg-blue-50 p-6 rounded-lg shadow-inner flex flex-col sm:flex-row items-center justify-between gap-4">
                    <div className="flex-grow">
                        <label htmlFor="file-upload" className="block text-lg font-semibold text-gray-700 mb-2">
                            Upload Document (PDF/TXT)
                        </label>
                        <input
                            id="file-upload"
                            type="file"
                            accept=".pdf,.txt"
                            onChange={handleFileChange}
                            className="block w-full text-sm text-gray-500
                                file:mr-4 file:py-2 file:px-4
                                file:rounded-full file:border-0
                                file:text-sm file:font-semibold
                                file:bg-indigo-50 file:text-indigo-700
                                hover:file:bg-indigo-100
                                cursor-pointer"
                        />
                        {selectedFile && (
                            <p className="mt-2 text-sm text-gray-600">Selected: <span className="font-medium">{selectedFile.name}</span></p>
                        )}
                    </div>
                    <button
                        onClick={handleUpload}
                        disabled={isLoading || !selectedFile} // Corrected syntax
                        className={`px-6 py-3 rounded-full font-bold text-white transition-all duration-300
                            ${isLoading || !selectedFile ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 shadow-md hover:shadow-lg'}`}
                    >
                        {isLoading && !summary ? 'Uploading...' : 'Upload & Summarize'} {/* Corrected syntax */}
                    </button>
                </div>

                {/* Status Message */}
                {message && (
                    <div className={`p-3 rounded-lg text-center font-medium ${message.includes('Error') ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'}`}>
                        {message}
                    </div>
                )}

                {/* Summary Display */}
                {summary && (
                    <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200">
                        <h2 className="text-2xl font-bold text-indigo-600 mb-3">Document Summary</h2>
                        <p className="text-gray-700 leading-relaxed">{summary}</p>
                    </div>
                )}

                {/* Interaction Modes */}
                <div className="flex justify-center gap-4 mb-6">
                    <button
                        onClick={() => setMode('askAnything')}
                        className={`px-6 py-3 rounded-full font-bold transition-all duration-300
                            ${mode === 'askAnything' ? 'bg-indigo-600 text-white shadow-lg' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
                    >
                        Ask Anything
                    </button>
                    <button
                        onClick={() => setMode('challengeMe')}
                        className={`px-6 py-3 rounded-full font-bold transition-all duration-300
                            ${mode === 'challengeMe' ? 'bg-indigo-600 text-white shadow-lg' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}
                    >
                        Challenge Me
                    </button>
                </div>

                {/* Ask Anything Mode */}
                {mode === 'askAnything' && (
                    <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 flex flex-col gap-4">
                        <h2 className="text-2xl font-bold text-indigo-600 mb-2">Ask Anything</h2>
                        <div ref={chatContainerRef} className="flex flex-col gap-3 h-64 overflow-y-auto p-3 bg-gray-50 rounded-lg border border-gray-200 chat-history-scroll">
                            {chatHistory.length === 0 && (
                                <p className="text-gray-500 text-center italic">Start by asking a question about your document!</p>
                            )}
                            {chatHistory.map((msg, index) => (
                                <div
                                    key={index}
                                    className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                                >
                                    <div
                                        className={`max-w-[70%] p-3 rounded-lg shadow-sm
                                            ${msg.role === 'user'
                                               ? 'bg-indigo-500 text-white rounded-br-none'
                                                : 'bg-gray-200 text-gray-800 rounded-bl-none'
                                            }`}
                                    >
                                        <p className="font-semibold text-sm mb-1">{msg.role === 'user' ? 'You' : 'Assistant'}</p>
                                        <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                        <div className="flex gap-3">
                            <input
                                type="text"
                                value={userInput}
                                onChange={(e) => setUserInput(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                                placeholder="Type your question here..."
                                className="flex-grow p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-400 focus:border-transparent outline-none"
                                disabled={isLoading || !selectedFile} // Corrected syntax
                            />
                            <button
                                onClick={handleSendMessage}
                                disabled={isLoading || !userInput.trim() || !selectedFile} // Corrected syntax
                                className={`px-5 py-3 rounded-lg font-bold text-white transition-all duration-300
                                    ${isLoading || !userInput.trim() || !selectedFile ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 shadow-md'}`}
                            >
                                Send
                            </button>
                        </div>
                    </div>
                )}

                {/* Challenge Me Mode */}
                {mode === 'challengeMe' && (
                    <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 flex flex-col gap-4">
                        <h2 className="text-2xl font-bold text-indigo-600 mb-2">Challenge Me</h2>
                        <button
                            onClick={handleGenerateChallenge}
                            disabled={isLoading || !selectedFile} // Corrected syntax
                            className={`px-6 py-3 rounded-full font-bold text-white transition-all duration-300 self-start
                                ${isLoading || !selectedFile ? 'bg-gray-400 cursor-not-allowed' : 'bg-purple-600 hover:bg-purple-700 shadow-md hover:shadow-lg'}`}
                        >
                            {isLoading && challengeQuestions.length === 0 ? 'Generating...' : 'Generate New Challenges'}
                        </button>

                        {challengeQuestions.length > 0 && (
                            <div className="flex flex-col gap-6 mt-4">
                                {challengeQuestions.map((q, index) => (
                                    <div key={index} className="bg-purple-50 p-4 rounded-lg border border-purple-200 shadow-sm">
                                        <p className="font-semibold text-lg text-purple-800 mb-2">Question {index + 1}:</p>
                                        <p className="text-gray-700 mb-3">{q}</p>
                                        <textarea
                                            value={challengeAnswers[index] || ''}
                                            onChange={(e) => handleChallengeAnswerChange(index, e.target.value)}
                                            placeholder="Your answer..."
                                            rows="3"
                                            className="w-full p-2 border border-gray-300 rounded-md focus:ring-1 focus:ring-purple-400 focus:border-transparent outline-none resize-y"
                                            disabled={isLoading}
                                        ></textarea>
                                        {challengeFeedback[index] && (
                                            <div className="mt-2 p-2 bg-green-100 text-green-800 rounded-md text-sm whitespace-pre-wrap">
                                                <p className="font-medium">Feedback:</p>
                                                <p>{challengeFeedback[index]}</p>
                                            </div>
                                        )}
                                    </div>
                                ))}
                                <button
                                    onClick={handleSubmitChallengeAnswers}
                                    disabled={isLoading || Object.keys(challengeAnswers).length === 0 || Object.values(challengeAnswers).every(answer => !answer.trim())} // Corrected syntax
                                    className={`px-6 py-3 rounded-full font-bold text-white transition-all duration-300 self-center
                                        ${isLoading || Object.keys(challengeAnswers).length === 0 || Object.values(challengeAnswers).every(answer => !answer.trim()) ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 shadow-md hover:shadow-lg'}`}
                                >
                                    Submit Answers
                                </button>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default App;