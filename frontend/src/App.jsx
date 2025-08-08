import React, { useState, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import 'katex/dist/katex.min.css';

const BACKEND_URL = 'http://localhost:10000';

class ErrorBoundary extends React.Component {
    constructor(props) { super(props); this.state = { hasError: false }; }
    static getDerivedStateFromError(error) { return { hasError: true }; }
    componentDidCatch(error, errorInfo) { console.error("Markdown rendering error:", error, errorInfo); }
    render() {
        if (this.state.hasError) {
            return (
                <div className="p-4 bg-red-100 border border-red-300 rounded-lg text-red-800">
                    <h4 className="font-bold">Rendering Error</h4>
                    <p>There was an issue displaying this content. Please try generating a new question.</p>
                </div>
            );
        }
        return this.props.children;
    }
}

const Background = () => <div className="grid-background"></div>;

const TrueFocus = ({ children, glowColor = 'rgba(167, 139, 250, 0.5)' }) => {
    const [isFocused, setIsFocused] = useState(false);
    const [position, setPosition] = useState({ x: 0, y: 0 });
    const [opacity, setOpacity] = useState(0);
    const handleMouseMove = (e) => {
        if (!e.currentTarget) return;
        const rect = e.currentTarget.getBoundingClientRect();
        setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    };
    const handleFocus = () => { setIsFocused(true); setOpacity(1); };
    const handleBlur = () => { setIsFocused(false); setOpacity(0); };
    return (
        <div className="relative" onMouseMove={handleMouseMove} onFocus={handleFocus} onBlur={handleBlur} onMouseEnter={handleFocus} onMouseLeave={handleBlur}>
            {children}
            <div className="pointer-events-none absolute -inset-px rounded-xl transition-opacity duration-300" style={{ opacity, background: isFocused ? `radial-gradient(600px circle at ${position.x}px ${position.y}px, ${glowColor}, transparent 40%)` : 'transparent' }} />
        </div>
    );
};

const Message = ({ type, message }) => {
    if (!message) return null;
    const baseClasses = "message p-3 rounded-lg mt-4 font-medium";
    const typeClasses = type === "success" ? "bg-emerald-100 text-emerald-800 border border-emerald-300" : "bg-red-100 text-red-800 border border-red-300";
    return <div className={`${baseClasses} ${typeClasses}`}>{message}</div>;
};

const SubjectSelector = ({ subjectCodes, level, onLevelChange, subjectCode, onSubjectCodeChange, idPrefix }) => {
    const handleLevelSelect = (e) => { onLevelChange(e.target.value); onSubjectCodeChange(''); };
    const currentSubjects = subjectCodes?.[level?.toLowerCase()] || {};
    return (
        <>
            <div>
                <label htmlFor={`${idPrefix}-level-select`} className="block text-gray-800 text-sm font-medium mb-2">Exam Level:</label>
                <select id={`${idPrefix}-level-select`} value={level} onChange={handleLevelSelect} required className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-gray-900">
                    <option value="" disabled>-- Select a Level --</option>
                    <option value="igcse">IGCSE</option>
                    <option value="alevel">A-Level</option>
                </select>
            </div>
            {level && (
                <div>
                    <label htmlFor={`${idPrefix}-subject-select`} className="block text-gray-800 text-sm font-medium mb-2">Subject Code:</label>
                    <select id={`${idPrefix}-subject-select`} value={subjectCode} onChange={(e) => onSubjectCodeChange(e.target.value)} required className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-gray-900">
                        <option value="" disabled>-- Select a Subject --</option>
                        {Object.entries(currentSubjects).map(([code, name]) => (
                            <option key={code} value={code}>{code} - {name.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</option>
                        ))}
                    </select>
                </div>
            )}
        </>
    );
};

const PastPaperCompiler = ({ subjectCodes }) => {
    const [level, setLevel] = useState('');
    const [subjectCode, setSubjectCode] = useState('');
    const [componentCodes, setComponentCodes] = useState('');
    const [years, setYears] = useState(() => Array.from({ length: 3 }, (_, i) => (new Date().getFullYear() - i).toString()));
    const [sessions, setSessions] = useState(['s', 'w']);
    const [loading, setLoading] = useState(false);
    const [successMessage, setSuccessMessage] = useState('');
    const [errorMessage, setErrorMessage] = useState('');
    const [removeBlank, setRemoveBlank] = useState(true);
    const [ignoreFirst, setIgnoreFirst] = useState(0);
    const [removePageTypes, setRemovePageTypes] = useState(['formula_sheet', 'instructions']);
    const [includeMarkSchemes, setIncludeMarkSchemes] = useState(false);

    const handleYearChange = (year) => setYears(p => p.includes(year) ? p.filter(y => y !== year) : [...p, year].sort());
    const handleSessionChange = (session) => setSessions(p => p.includes(session) ? p.filter(s => s !== session) : [...p, session].sort());
    
    const handleRemoveTypeChange = (type) => {
        setRemovePageTypes(prev => 
            prev.includes(type) ? prev.filter(t => t !== type) : [...prev, type]
        );
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setSuccessMessage('');
        setErrorMessage('');
        if (!level || !subjectCode || !componentCodes.trim() || years.length === 0 || sessions.length === 0) {
            setErrorMessage('Please fill out all required fields.');
            setLoading(false);
            return;
        }
        try {
            const response = await fetch(`${BACKEND_URL}/generate_past_paper`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    level,
                    subject_code: subjectCode,
                    component_codes: componentCodes,
                    years,
                    sessions,
                    remove_blank_pages: removeBlank,
                    ignore_first_pages: ignoreFirst,
                    remove_page_types: removePageTypes,
                    include_mark_schemes: includeMarkSchemes,
                }),
            });
            if (response.headers.get("content-type")?.includes("application/json")) {
                const data = await response.json();
                if (response.status === 202) {
                    setSuccessMessage(data.message);
                } else {
                    setErrorMessage(data.message || 'Failed to generate paper.');
                }
            } else if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `compiled_papers_${subjectCode}.pdf`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
                const processingMessage = response.headers.get('X-Processing-Message');
                setSuccessMessage(processingMessage || 'Paper compiled and downloaded!');
            } else {
                setErrorMessage(`Download failed: ${await response.text() || response.statusText}`);
            }
        } catch (error) {
            setErrorMessage('A network error occurred.');
        } finally {
            setLoading(false);
        }
    };

    const pageTypesToRemove = [
        { id: 'formula_sheet', label: 'Formula Sheets' },
        { id: 'periodic_table', label: 'Periodic Tables' },
        { id: 'instructions', label: 'Instructions Pages' },
    ];

    return (
        <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl">
            <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Past Paper Compiler</h2>
            <Message type="success" message={successMessage} />
            <Message type="error" message={errorMessage} />
            <form onSubmit={handleSubmit} className="space-y-6">
                <SubjectSelector subjectCodes={subjectCodes} level={level} onLevelChange={setLevel} subjectCode={subjectCode} onSubjectCodeChange={setSubjectCode} idPrefix="compiler" />
                <div>
                    <label htmlFor="component_codes" className="block text-gray-800 text-sm font-medium mb-2">Component Codes (comma-separated):</label>
                    <input type="text" id="component_codes" value={componentCodes} onChange={(e) => setComponentCodes(e.target.value)} placeholder="e.g., 11,12,21" required className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-gray-900" />
                </div>
                
                <div>
                    <label className="block text-gray-800 text-sm font-medium mb-2">Processing Options:</label>
                    <div className="space-y-4 bg-gray-50 p-4 rounded-lg border">
                        <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
                            <label className="flex items-center cursor-pointer">
                                <input type="checkbox" checked={removeBlank} onChange={(e) => setRemoveBlank(e.target.checked)} className="form-checkbox h-5 w-5 text-indigo-600 rounded" />
                                <span className="ml-2 text-gray-900">Remove Blank Pages (AI-enhanced)</span>
                            </label>
                            <div className="flex items-center">
                                <label htmlFor="ignore_first" className="text-gray-900 mr-2 whitespace-nowrap">Ignore first</label>
                                <input type="number" id="ignore_first" value={ignoreFirst} onChange={(e) => setIgnoreFirst(Number(e.target.value))} min="0" className="w-20 p-2 border border-gray-300 rounded-lg shadow-sm text-gray-900" />
                                <span className="ml-2 text-gray-900">pages</span>
                            </div>
                        </div>
                         <div>
                            <p className="text-gray-800 text-sm font-medium mb-2">Remove additional content:</p>
                            <div className="flex flex-wrap gap-x-4 gap-y-2">
                                {pageTypesToRemove.map(pageType => (
                                    <label key={pageType.id} className="flex items-center cursor-pointer">
                                        <input
                                            type="checkbox"
                                            checked={removePageTypes.includes(pageType.id)}
                                            onChange={() => handleRemoveTypeChange(pageType.id)}
                                            className="form-checkbox h-5 w-5 text-indigo-600 rounded"
                                        />
                                        <span className="ml-2 text-gray-900">{pageType.label}</span>
                                    </label>
                                ))}
                            </div>
                        </div>
                        <div className="border-t pt-4">
                             <label className="flex items-center cursor-pointer">
                                <input type="checkbox" checked={includeMarkSchemes} onChange={(e) => setIncludeMarkSchemes(e.target.checked)} className="form-checkbox h-5 w-5 text-indigo-600 rounded" />
                                <span className="ml-2 text-gray-900 font-semibold">Include Mark Schemes</span>
                            </label>
                        </div>
                    </div>
                </div>

                <div>
                    <label className="block text-gray-800 text-sm font-medium mb-2">Years:</label>
                    <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
                        {Array.from({ length: new Date().getFullYear() - 2005 }, (_, i) => new Date().getFullYear() - i).map(year => (
                            <label key={year} className="flex items-center bg-gray-50 p-3 rounded-lg border border-gray-200 cursor-pointer hover:bg-gray-100">
                                <input type="checkbox" value={year} checked={years.includes(year.toString())} onChange={() => handleYearChange(year.toString())} className="form-checkbox h-5 w-5 text-indigo-600 rounded" />
                                <span className="ml-2 text-gray-900">{year}</span>
                            </label>
                        ))}
                    </div>
                </div>
                <div>
                    <label className="block text-gray-800 text-sm font-medium mb-2">Sessions:</label>
                    <div className="flex flex-wrap gap-4">
                        {['m', 's', 'w'].map(session => (
                            <label key={session} className="flex items-center bg-gray-50 p-3 rounded-lg border border-gray-200 cursor-pointer hover:bg-gray-100">
                                <input type="checkbox" value={session} checked={sessions.includes(session)} onChange={() => handleSessionChange(session)} className="form-checkbox h-5 w-5 text-indigo-600 rounded" />
                                <span className="ml-2 text-gray-900">{session === 'm' ? 'March' : session === 's' ? 'May/June' : 'Oct/Nov'}</span>
                            </label>
                        ))}
                    </div>
                </div>
                <div className="text-center">
                    <button type="submit" className="w-full py-3 px-6 bg-indigo-600 text-white font-semibold rounded-lg shadow-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50" disabled={loading}>
                        {loading ? 'Processing...' : 'Download and Merge Papers'}
                    </button>
                </div>
            </form>
        </div>
    );
};

const TopicalQuestionSelector = ({ subjectCodes }) => {
    const [level, setLevel] = useState('');
    const [subjectCode, setSubjectCode] = useState('');
    const [topic, setTopic] = useState('');
    const [question, setQuestion] = useState('');
    const [modelAnswer, setModelAnswer] = useState('');
    const [diagramImage, setDiagramImage] = useState(null);
    const [userAnswer, setUserAnswer] = useState('');
    const [questionLoading, setQuestionLoading] = useState(false);
    const [questionSuccessMessage, setQuestionSuccessMessage] = useState('');
    const [questionErrorMessage, setQuestionErrorMessage] = useState('');
    const [markingLoading, setMarkingLoading] = useState(false);
    const [markingSuccessMessage, setMarkingSuccessMessage] = useState('');
    const [markingErrorMessage, setMarkingErrorMessage] = useState('');
    const [markedScore, setMarkedScore] = useState(null);
    const [feedbackStrengths, setFeedbackStrengths] = useState('');
    const [feedbackImprovements, setFeedbackImprovements] = useState('');
    const [correctedAnswer, setCorrectedAnswer] = useState('');

    const resetAllState = () => {
        setQuestion(''); setModelAnswer(''); setDiagramImage(null); setUserAnswer('');
        setQuestionSuccessMessage(''); setQuestionErrorMessage(''); setMarkedScore(null);
        setFeedbackStrengths(''); setFeedbackImprovements('');
        setCorrectedAnswer(''); setMarkingSuccessMessage(''); setMarkingErrorMessage('');
    };

    const handleGenerateQuestion = useCallback(async (e) => {
        e.preventDefault();
        setQuestionLoading(true);
        resetAllState();

        if (!level || !subjectCode || !topic.trim()) {
            setQuestionErrorMessage('Please select a Level, Subject, and enter a Topic.');
            setQuestionLoading(false); return;
        }

        try {
            const response = await fetch(`${BACKEND_URL}/generate_question`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ subject_code: subjectCode, topic: topic, level: level }),
            });
            const data = await response.json();
            if (response.ok && data.success) {
                setQuestion(data.question || '');
                setModelAnswer(data.model_answer || '');
                setDiagramImage(data.diagram_image || null);
                setQuestionSuccessMessage('Question generated successfully!');
            } else {
                setQuestionErrorMessage(data.message || 'Failed to generate question.');
            }
        } catch (error) {
            setQuestionErrorMessage('A network error occurred.');
        } finally {
            setQuestionLoading(false);
        }
    }, [subjectCode, topic, level]);

    const handleMarkAnswer = useCallback(async () => {
        setMarkingLoading(true);
        setMarkingSuccessMessage(''); setMarkingErrorMessage(''); setMarkedScore(null);
        setFeedbackStrengths(''); setFeedbackImprovements(''); setCorrectedAnswer('');
        if (!userAnswer.trim()) {
            setMarkingErrorMessage('Please provide an answer.');
            setMarkingLoading(false); return;
        }
        try {
            const response = await fetch(`${BACKEND_URL}/mark_answer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, user_answer: userAnswer, model_answer: modelAnswer }),
            });
            const data = await response.json();
            if (response.ok && data.success) {
                setMarkedScore(data.score);
                if (data.feedback) {
                    setFeedbackStrengths(data.feedback.strengths || '');
                    setFeedbackImprovements(data.feedback.improvements || '');
                }
                setCorrectedAnswer(data.corrected_answer || '');
                setMarkingSuccessMessage('Answer marked successfully!');
            } else {
                setMarkingErrorMessage(data.message || 'Failed to mark answer.');
            }
        } catch (error) {
            setMarkingErrorMessage('A network error occurred.');
        } finally {
            setMarkingLoading(false);
        }
    }, [question, userAnswer, modelAnswer]);
    
    const totalMarks = (q) => (q.match(/\[(\d+)\]/g) || []).reduce((sum, mark) => sum + parseInt(mark.slice(1, -1)), 0);

    return (
        <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl">
            <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Topical Question Selector (AI)</h2>
            <Message type="success" message={questionSuccessMessage} />
            <Message type="error" message={questionErrorMessage} />
            <form onSubmit={handleGenerateQuestion} className="space-y-6">
                <SubjectSelector subjectCodes={subjectCodes} level={level} onLevelChange={setLevel} subjectCode={subjectCode} onSubjectCodeChange={setSubjectCode} idPrefix="topical" />
                <div>
                    <label htmlFor="topic" className="block text-gray-800 text-sm font-medium mb-2">Topic:</label>
                    <input type="text" id="topic" value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="e.g., Cell Structure and Function" required className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-gray-900" />
                </div>
                <div className="text-center">
                    <button type="submit" disabled={questionLoading} className="w-full py-3 px-6 bg-purple-600 text-white font-semibold rounded-lg shadow-md hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50 flex items-center justify-center">
                        {questionLoading ? (
                            <>
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                <span>Generating...</span>
                            </>
                        ) : (
                            'Generate Question'
                        )}
                    </button>
                </div>
            </form>

            {question && (
                <div className="mt-8 p-6 bg-indigo-50 border border-indigo-200 rounded-lg shadow-inner">
                    <h3 className="text-xl font-semibold text-indigo-800 mb-4">Generated Question:</h3>
                    <ErrorBoundary>
                        {diagramImage && (
                            <div className="my-4 p-2 bg-white rounded-lg border flex justify-center">
                                <img src={`data:image/png;base64,${diagramImage}`} alt="Diagram for the question" className="max-w-full h-auto rounded" />
                            </div>
                        )}
                        <ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex]} className="markdown-container text-gray-900">
                          {(question || '').replace(/\\n/g, '\n\n')}
                        </ReactMarkdown>
                    </ErrorBoundary>
                    
                    <div className="mt-6">
                        <label htmlFor="user_answer" className="block text-gray-800 text-sm font-medium mb-2">Your Answer:</label>
                        <textarea id="user_answer" value={userAnswer} onChange={(e) => setUserAnswer(e.target.value)} rows="8" placeholder="Type your answer here..." className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 resize-y text-gray-900"></textarea>
                    </div>
                    <div className="text-right mt-4">
                        <button onClick={handleMarkAnswer} disabled={markingLoading} className="py-3 px-6 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-50">
                            {markingLoading ? 'Marking...' : 'Submit for Marking'}
                        </button>
                    </div>
                    <Message type="success" message={markingSuccessMessage} />
                    <Message type="error" message={markingErrorMessage} />
                    {markedScore !== null && (
                        <div className="mt-8 p-6 bg-purple-50 border border-purple-200 rounded-lg shadow-inner">
                            <h3 className="text-xl font-semibold text-purple-800 mb-4">Examiner Feedback:</h3>
                            <ErrorBoundary>
                                <p className="text-2xl font-bold text-purple-700 mb-4">Score: {markedScore} / {totalMarks(question)}</p>
                                {feedbackStrengths && (
                                    <div className="mb-4"><h4 className="font-semibold text-purple-700">Strengths:</h4><ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex]} className="markdown-container text-gray-900">{feedbackStrengths}</ReactMarkdown></div>
                                )}
                                {feedbackImprovements && (
                                    <div><h4 className="font-semibold text-purple-700">Areas for Improvement:</h4><ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex]} className="markdown-container text-gray-900">{feedbackImprovements}</ReactMarkdown></div>
                                )}
                                {modelAnswer && (
                                    <div className="mt-6"><h4 className="font-semibold text-indigo-700">Official Model Answer:</h4><ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex]} className="markdown-container text-gray-900">{modelAnswer}</ReactMarkdown></div>
                                )}
                                {correctedAnswer && (
                                    <div className="mt-6"><h4 className="font-semibold text-purple-700">Corrected Answer:</h4><ReactMarkdown remarkPlugins={[remarkMath, remarkGfm]} rehypePlugins={[rehypeKatex]} className="markdown-container text-gray-900">{correctedAnswer}</ReactMarkdown></div>
                                )}
                            </ErrorBoundary>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

const ExtractedQuestionSelector = ({ subjectCodes }) => {
    const [level, setLevel] = useState('');
    const [subjectCode, setSubjectCode] = useState('');
    const [topic, setTopic] = useState('');
    const [extractedComponentCodes, setExtractedComponentCodes] = useState('');
    const [years, setYears] = useState(() => Array.from({ length: 3 }, (_, i) => (new Date().getFullYear() - i).toString()));
    const [loading, setLoading] = useState(false);
    const [successMessage, setSuccessMessage] = useState('');
    const [errorMessage, setErrorMessage] = useState('');

    const handleYearChange = (year) => setYears(p => p.includes(year) ? p.filter(y => y !== year) : [...p, year].sort());

    const handleExtractQuestion = async (e) => {
        e.preventDefault();
        setLoading(true);
        setSuccessMessage(''); setErrorMessage('');
        if (!level || !subjectCode || !topic.trim() || years.length === 0 || !extractedComponentCodes.trim()) {
            setErrorMessage('Please fill out all fields.');
            setLoading(false); return;
        }
        try {
            const response = await fetch(`${BACKEND_URL}/generate_extracted_paper`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ subject_code: subjectCode, topic, years, component_codes: extractedComponentCodes, level }),
            });
            if (response.headers.get("content-type")?.includes("application/json")) {
                const data = await response.json();
                if (response.status === 202) {
                    setSuccessMessage(data.message);
                } else {
                    setErrorMessage(data.message || 'Failed to generate extracted paper.');
                }
            } else if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `extracted_questions_${subjectCode}_${topic}.pdf`;
                document.body.appendChild(a); a.click(); a.remove();
                window.URL.revokeObjectURL(url);
                const processingMessage = response.headers.get('X-Processing-Message');
                setSuccessMessage(processingMessage || 'Extracted questions PDF downloaded successfully!');
            } else {
                setErrorMessage(`Download failed: ${await response.text() || response.statusText}`);
            }
        } catch (error) {
            setErrorMessage('A network error occurred.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl">
            <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">Generate Extracted Questions PDF (AI)</h2>
            <Message type="success" message={successMessage} />
            <Message type="error" message={errorMessage} />
            <form onSubmit={handleExtractQuestion} className="space-y-6">
                <SubjectSelector subjectCodes={subjectCodes} level={level} onLevelChange={setLevel} subjectCode={subjectCode} onSubjectCodeChange={setSubjectCode} idPrefix="extractor" />
                <div>
                    <label htmlFor="topic_extract" className="block text-gray-800 text-sm font-medium mb-2">Topic for Extraction:</label>
                    <input type="text" id="topic_extract" value={topic} onChange={(e) => setTopic(e.target.value)} placeholder="e.g., Photosynthesis stages" required className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-gray-900" />
                </div>
                <div>
                    <label htmlFor="extracted_component_codes" className="block text-gray-800 text-sm font-medium mb-2">Component Codes (comma-separated):</label>
                    <input type="text" id="extracted_component_codes" value={extractedComponentCodes} onChange={(e) => setExtractedComponentCodes(e.target.value)} placeholder="e.g., 11,21,31" required className="w-full p-3 border border-gray-300 rounded-lg shadow-sm focus:ring-indigo-500 focus:border-indigo-500 text-gray-900" />
                </div>
                <div>
                    <label className="block text-gray-800 text-sm font-medium mb-2">Years to search:</label>
                    <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
                        {Array.from({ length: new Date().getFullYear() - 2005 }, (_, i) => new Date().getFullYear() - i).map(year => (
                            <label key={year} className="flex items-center bg-gray-50 p-3 rounded-lg border border-gray-200 cursor-pointer hover:bg-gray-100">
                                <input type="checkbox" value={year} checked={years.includes(year.toString())} onChange={() => handleYearChange(year.toString())} className="form-checkbox h-5 w-5 text-indigo-600 rounded" />
                                <span className="ml-2 text-gray-900">{year}</span>
                            </label>
                        ))}
                    </div>
                </div>
                <div className="text-center">
                    <button type="submit" disabled={loading} className="w-full py-3 px-6 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50">
                        {loading ? 'Generating PDF...' : 'Generate Extracted Questions PDF'}
                    </button>
                </div>
            </form>
        </div>
    );
};

function App() {
    const [activeTab, setActiveTab] = useState('compiler');
    const [fetchedSubjectCodes, setFetchedSubjectCodes] = useState({ igcse: {}, alevel: {} });
    const [loadingSubjects, setLoadingSubjects] = useState(true);
    const [subjectError, setSubjectError] = useState('');
    const [cacheMessage, setCacheMessage] = useState('');

    const handleClearCache = async () => {
        if (window.confirm("Are you sure you want to clear the server cache? This will force all papers to be re-downloaded and re-processed on the next request.")) {
            try {
                setCacheMessage('Clearing cache...');
                const response = await fetch(`${BACKEND_URL}/clear_cache`, { method: 'POST' });
                const data = await response.json();
                if (data.success) {
                    setCacheMessage('Cache cleared successfully!');
                } else {
                    setCacheMessage(`Failed to clear cache: ${data.message}`);
                }
            } catch (error) {
                setCacheMessage('Error connecting to server to clear cache.');
            }
            setTimeout(() => setCacheMessage(''), 4000);
        }
    };

    useEffect(() => {
        const fetchSubjects = async () => {
            try {
                const response = await fetch(`${BACKEND_URL}/get_subject_codes`);
                if (!response.ok) throw new Error('Failed to load subject codes.');
                const data = await response.json();
                setFetchedSubjectCodes(data);
            } catch (error) {
                setSubjectError('Could not connect to backend. Please ensure the server is running.');
            } finally {
                setLoadingSubjects(false);
            }
        };
        fetchSubjects();
    }, []);

    if (loadingSubjects) {
        return <div className="min-h-screen bg-gray-100 flex items-center justify-center"><p>Loading...</p></div>;
    }

    if (subjectError) {
        return (
            <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
                <p className="text-xl text-red-700 font-semibold mb-4">Error:</p>
                <p className="text-md text-red-600 text-center">{subjectError}</p>
            </div>
        );
    }

    return (
        <>
            <Background />
            <div className="min-h-screen text-gray-100 flex flex-col items-center py-10 px-4 font-inter relative z-10">
                <header className="w-full max-w-4xl text-center mb-8">
                    <h1 className="text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-indigo-700 to-purple-500 mb-4">Cambridge Exam Helper</h1>
                    <p className="text-lg text-gray-300">Your all-in-one tool for IGCSE and A-Level preparation.</p>
                </header>
                <nav className="mb-8 w-full max-w-2xl">
                     <TrueFocus glowColor="rgba(147, 112, 219, 0.4)">
                        <ul className="flex justify-center bg-white/80 backdrop-blur-sm p-2 rounded-xl shadow-md border border-gray-200">
                            <li className="flex-1">
                                <button onClick={() => setActiveTab('compiler')} className={`w-full py-3 px-4 rounded-lg font-semibold transition-all duration-300 ${activeTab === 'compiler' ? 'bg-indigo-600 text-white shadow-lg' : 'text-gray-700 hover:bg-gray-100'}`}>
                                    Past Paper Compiler
                                </button>
                            </li>
                            <li className="flex-1">
                                <button onClick={() => setActiveTab('topical')} className={`w-full py-3 px-4 rounded-lg font-semibold transition-all duration-300 ${activeTab === 'topical' ? 'bg-purple-600 text-white shadow-lg' : 'text-gray-700 hover:bg-gray-100'}`}>
                                    Topical Questions (AI)
                                </button>
                            </li>
                            <li className="flex-1">
                                <button onClick={() => setActiveTab('extracted')} className={`w-full py-3 px-4 rounded-lg font-semibold transition-all duration-300 ${activeTab === 'extracted' ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-700 hover:bg-gray-100'}`}>
                                    Extracted Questions
                                </button>
                            </li>
                        </ul>
                    </TrueFocus>
                </nav>
                <main className="w-full flex justify-center">
                    {activeTab === 'compiler' && <PastPaperCompiler subjectCodes={fetchedSubjectCodes} />}
                    {activeTab === 'topical' && <TopicalQuestionSelector subjectCodes={fetchedSubjectCodes} />}
                    {activeTab === 'extracted' && <ExtractedQuestionSelector subjectCodes={fetchedSubjectCodes} />}
                </main>

                <footer className="mt-12 w-full max-w-2xl text-center">
                    <div className="bg-white/10 backdrop-blur-sm p-4 rounded-xl">
                        <h3 className="font-semibold text-gray-200 mb-2">Cache Management</h3>
                        <p className="text-xs text-gray-400 mb-3">If papers are not processing correctly, clearing the cache may help.</p>
                        <button onClick={handleClearCache} className="py-2 px-4 bg-red-600 text-white font-semibold rounded-lg shadow-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 disabled:opacity-50">
                            Clear Server Cache
                        </button>
                        {cacheMessage && <p className="text-sm mt-2 text-gray-300">{cacheMessage}</p>}
                    </div>
                </footer>
            </div>
        </>
    );
}

export default App;