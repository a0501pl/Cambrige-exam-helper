import os
import re
import io
import json
import time
import uuid
import shutil
import logging
import tempfile
import requests
import hashlib
import concurrent.futures
from functools import lru_cache
import base64
import matplotlib
# Use a non-interactive backend suitable for servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.errors import PdfReadError
from pdf2image import convert_from_path
from PIL import Image as PILImage
import pytesseract
import google.generativeai as genai

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# ==============================================================================
# --- Version & Configuration ---
# ==============================================================================
print("--- Running app.py version: v102 (Final Schema Fix) ---")

class Config:
    """Centralized configuration for the application."""
    API_KEYS = os.environ.get("API_KEYS", "AIzaSyDPPEavwhx1TkkYcYYuevOr1UWEVNOAcPo").split(',')
    TESSERACT_CMD = shutil.which('tesseract')
    MAX_AI_RETRIES = 3
    AI_RETRY_DELAY_SECONDS = 5
    RATE_LIMIT_DELAY_SECONDS = 60
    AI_MODEL_VISION = 'gemini-1.5-flash'
    AI_MODEL_TEXT = 'gemini-1.5-flash'
    PDF_DOWNLOAD_TIMEOUT_SECONDS = 30
    DOWNLOAD_MAX_WORKERS = 8
    PROCESSING_MAX_WORKERS = 4
    OCR_DPI = 200
    PAGES_TO_IGNORE_FOR_PDF_OUTPUT = 0
    BLANK_IMAGE_THRESHOLD = 245
    BLANK_IMAGE_PIXEL_FRACTION = 0.995
    MAX_QUESTIONS_PER_PAPER = 25

# --- App Setup ---
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Configure Tesseract and Google AI ---
if Config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_CMD
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        logging.info(f"Tesseract version {tesseract_version} found and configured.")
    except Exception as e:
        logging.error(f"Tesseract found but failed to get version. Error: {e}")
else:
    logging.error("TESSERACT NOT FOUND. OCR features will be disabled.")

# ==============================================================================
# --- Global State & Cleanup ---
# ==============================================================================
_temp_dirs_to_cleanup = set()

def manage_temp_dir(action='add'):
    if action == 'add':
        temp_dir = tempfile.mkdtemp()
        _temp_dirs_to_cleanup.add(temp_dir)
        return temp_dir
    elif action == 'cleanup':
        logging.info(f"Starting cleanup of {len(_temp_dirs_to_cleanup)} temporary directories...")
        for d in list(_temp_dirs_to_cleanup):
            try:
                if os.path.exists(d): shutil.rmtree(d, ignore_errors=True)
                _temp_dirs_to_cleanup.remove(d)
            except Exception as e:
                logging.warning(f"Cleanup error removing dir {d}: {e}")
        logging.info("Cleanup finished.")

# ==============================================================================
# --- Subject Code Data ---
# ==============================================================================
subject_codes_igcse = {
    "0452": "accounting-igcse", "0985": "accounting-9-1-igcse", "0548": "afrikaans-second-language-igcse",
    "0600": "agriculture-igcse", "0508": "arabic-first-language-igcse", "7184": "arabic-9-1-first-language-igcse",
    "0544": "arabic-foreign-language-igcse", "7180": "arabic-9-1-igcse", "0400": "art-design-igcse",
    "0989": "art-design-9-1-igcse", "0538": "bahasa-indonesia-igcse", "0610": "biology-igcse",
    "0970": "biology-9-1-igcse", "0450": "business-studies-igcse", "0986": "business-studies-9-1-igcse",
    "0620": "chemistry-igcse", "0971": "chemistry-9-1-igcse", "0509": "chinese-first-language-igcse",
    "0523": "chinese-second-language-igcse", "0547": "chinese-mandarin-foreign-language-igcse",
    "0478": "computer-science-igcse", "0984": "computer-science-9-1-igcse", "0445": "design-technology-igcse",
    "0979": "design-technology-9-1-igcse", "0411": "drama-igcse", "0994": "drama-9-1-igcse",
    "0455": "economics-igcse", "0987": "economics-9-1-igcse", "0500": "english-first-language-igcse",
    "0990": "english-first-language-9-1-igcse", "0475": "english-literature-igcse",
    "0992": "english-literature-9-1-igcse", "0511": "english-second-language-count-in-speaking-igcse",
    "0991": "english-second-language-9-1-count-in-speaking-igcse",
    "0510": "english-second-language-speaking-endorsement-igcse",
    "0993": "english-second-language-speaking-endorsement-9-1-igcse", "0454": "enterprise-igcse",
    "0680": "environmental-management-igcse", "0648": "food-nutrition-igcse", "0501": "french-first-language-igcse",
    "0520": "french-foreign-language-igcse", "7156": "french-9-1-igcse", "0460": "geography-igcse",
    "0976": "geography-9-1-igcse", "0505": "german-first-language-igcse", "0525": "german-foreign-language-igcse",
    "7159": "german-9-1-igcse", "0457": "global-perspectives-igcse", "0549": "hindi-second-language-igcse",
    "0470": "history-igcse", "0977": "history-9-1-igcse", "0417": "ict-igcse", "0983": "ict-9-1-igcse",
    "0493": "islamiyat-igcse", "0535": "italian-foreign-language-igcse", "7164": "italian-9-1-igcse",
    "0546": "malay-foreign-language-igcse", "0697": "marine-science-igcse", "0580": "mathematics-igcse",
    "0606": "mathematics-additional-igcse", "0607": "mathematics-international-igcse", "0980": "mathematics-9-1-igcse",
    "0410": "music-igcse", "0978": "music-9-1-igcse", "0448": "pakistan-studies-igcse",
    "0413": "physical-education-igcse", "0995": "physical-education-9-1-igcse", "0652": "physical-science-igcse",
    "0625": "physics-igcse", "0972": "physics-9-1-igcse", "0990": "psychology-igcse",
    "0490": "religious-studies-igcse", "0653": "science-combined-igcse", "0654": "sciences-coordinated-double-igcse",
    "0495": "sociology-igcse", "0502": "spanish-first-language-igcse", "0530": "spanish-foreign-language-igcse",
    "7160": "spanish-9-1-igcse", "0471": "travel-tourism-igcse", "0539": "urdu-second-language-igcse"
}
subject_codes_alevel = {
    "9706": "accounting-alevel", "9680": "arabic-alevel", "9479": "art-design-alevel", "9700": "biology-alevel",
    "9609": "business-alevel", "9701": "chemistry-alevel", "9274": "classical-studies-alevel",
    "9618": "computer-science-alevel", "9705": "design-technology-alevel", "9482": "drama-alevel",
    "9708": "economics-alevel", "9093": "english-language-alevel", "9695": "english-literature-alevel",
    "8021": "english-general-paper-as-alevel", "8291": "environmental-management-as-alevel",
    "9716": "french-alevel", "9696": "geography-alevel", "9717": "german-alevel",
    "9239": "global-perspectives-research-alevel", "9489": "history-alevel", "9626": "information-technology-alevel",
    "9084": "law-alevel", "9693": "marine-science-alevel", "9709": "mathematics-alevel",
    "9231": "further-mathematics-alevel", "9607": "media-studies-alevel", "9483": "music-alevel",
    "9702": "physics-alevel", "9990": "psychology-alevel", "9699": "sociology-alevel", "9718": "portuguese-alevel",
    "9694": "thinking-skills-alevel", "9395": "travel-tourism-alevel"
}
subject_codes = {**subject_codes_igcse, **subject_codes_alevel}

# ==============================================================================
# --- Core AI and Data Processing Utilities ---
# ==============================================================================
_api_keys = app.config['API_KEYS']
_current_key_index = 0
_placeholder_keys = ["", "YOUR_API_KEY_HERE"]

if _api_keys and _api_keys[0] not in _placeholder_keys:
    logging.info(f"Google AI configured with {len(_api_keys)} key(s).")
else:
    logging.error("Google AI API key is not configured or is a placeholder. AI features will be disabled.")

def get_json_from_ai_with_retry(prompt, schema, temperature=0.2, model_type='text'):
    global _current_key_index
    if not _api_keys or _api_keys[0] in _placeholder_keys:
        logging.error("AI call attempted but no valid API key is configured.")
        return None

    # For multi-part prompts (with images), it's safest to ensure it's a tuple.
    if isinstance(prompt, list):
        prompt = tuple(prompt)

    for attempt in range(app.config['MAX_AI_RETRIES']):
        for i in range(len(_api_keys)):
            current_key_index_for_attempt = (_current_key_index + i) % len(_api_keys)
            try:
                logging.info(f"AI JSON call attempt {attempt + 1}, using key index {current_key_index_for_attempt}...")
                genai.configure(api_key=_api_keys[current_key_index_for_attempt])

                model_name = app.config['AI_MODEL_TEXT'] if model_type == 'text' else app.config['AI_MODEL_VISION']
                model = genai.GenerativeModel(model_name)

                generation_config = genai.types.GenerationConfig(
                    temperature=temperature, max_output_tokens=8192,
                    response_mime_type="application/json", response_schema=schema
                )

                response = model.generate_content(prompt, generation_config=generation_config)

                if response.candidates and response.candidates[0].content.parts:
                    _current_key_index = current_key_index_for_attempt
                    return json.loads(response.candidates[0].content.parts[0].text)
                logging.warning(f"AI call with key index {current_key_index_for_attempt} returned invalid content.")
            except Exception as e:
                logging.error(f"Exception on AI call with key index {current_key_index_for_attempt}: {e}")
                logging.error(traceback.format_exc()) # Keep detailed traceback for debugging
                if '429' in str(e) or 'rate limit' in str(e).lower():
                    logging.warning(f"Rate limit on key index {current_key_index_for_attempt}. Trying next key.")

        logging.warning(f"All API keys failed in attempt cycle {attempt + 1}. Waiting {app.config['AI_RETRY_DELAY_SECONDS']}s before next cycle.")
        time.sleep(app.config['AI_RETRY_DELAY_SECONDS'])

    logging.error("All AI call attempts with all API keys failed.")
    return None

def classify_pages_in_batch_ai(image_paths_dict):
    if not image_paths_dict:
        return {}

    try:
        logging.info(f"AI analyzing {len(image_paths_dict)} pages in a single batch...")

        prompt_parts = [
            genai.types.Part.from_text("""
            Analyze the following sequence of exam paper pages. For each page, classify its content into one category:
            - "question": Contains exam questions.
            - "formula_sheet": A list of formulas, data, or constants.
            - "periodic_table": A periodic table.
            - "instructions": Exam instructions.
            - "blank": Blank or only headers/footers.

            Your response must be a JSON object with a single key "classifications", which is an array of objects.
            Each object must have "page_number" (integer) and "page_type" (string category).
            Process all images provided.
            """)
        ]

        for page_num, path in sorted(image_paths_dict.items()):
            try:
                with open(path, "rb") as image_file:
                    # FIX: Pass raw bytes to from_data, not base64 encoded string.
                    image_bytes = image_file.read()
                    prompt_parts.append(genai.types.Part.from_data(
                        data=image_bytes,
                        mime_type="image/png"
                    ))
            except Exception as img_read_err:
                logging.error(f"Failed to read image {path}: {img_read_err}")
                continue

        schema = {
            "type": "OBJECT",
            "properties": {
                "classifications": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "page_number": {"type": "INTEGER"},
                            "page_type": {"type": "STRING"}
                        },
                        "required": ["page_number", "page_type"]
                    }
                }
            },
            "required": ["classifications"]
        }

        result = get_json_from_ai_with_retry(prompt_parts, schema, temperature=0.0, model_type='vision')

        if result and 'classifications' in result:
            classification_map = {item['page_number']: item['page_type'] for item in result['classifications']}
            logging.info(f"AI batch classification successful: {classification_map}")
            return classification_map

    except Exception as e:
        logging.error(f"AI batch page analysis failed: {e}")
        logging.error(traceback.format_exc())

    logging.warning("AI batch analysis failed, falling back to 'unknown' for all pages.")
    return {pageNum: "unknown" for pageNum in image_paths_dict}


def is_blank_image_pixel_based(image_path):
    if not image_path or not os.path.exists(image_path): return True
    with PILImage.open(image_path).convert("L") as img:
        pixels = list(img.getdata())
        if not pixels: return True
        bright_pixels = sum(1 for p in pixels if p > app.config['BLANK_IMAGE_THRESHOLD'])
        return (bright_pixels / len(pixels)) > app.config['BLANK_IMAGE_PIXEL_FRACTION']

def get_potential_pdf_urls(subject_code, year, session, component, level, paper_type='qp'):
    year_short = year[-2:]
    level_map = {"igcse": "Cambridge%20IGCSE", "alevel": "A%20Levels"}
    level_url_part = level_map.get(level, "Miscellaneous")
    subject_name_raw = subject_codes.get(subject_code, "unknown-subject")
    subject_name_clean = subject_name_raw.split('-')[0].replace('-', ' ').title()
    subject_url_part = f"{subject_name_clean.replace(' ', '%20')}%20({subject_code})"
    return [
        f"https://papers.gceguide.com/{level_url_part}/{subject_url_part}/{year}/{subject_code}_{session}{year_short}_{paper_type}_{component}.pdf",
        f"https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/{subject_code}_{session}{year_short}_{paper_type}_{component}.pdf",
    ]

def _download_single_pdf(file_info):
    subject_code, year, session, comp_part, level, paper_type, download_folder = file_info
    filename = f"{subject_code}_{session}{year[-2:]}_{paper_type}_{comp_part}.pdf"
    file_path = os.path.join(download_folder, filename)
    for url in get_potential_pdf_urls(subject_code, year, session, comp_part, level, paper_type):
        try:
            response = requests.get(url, stream=True, timeout=app.config['PDF_DOWNLOAD_TIMEOUT_SECONDS'])
            if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                with open(file_path, 'wb') as f: shutil.copyfileobj(response.raw, f)
                logging.info(f"Successfully downloaded: {filename} from {url}")
                return file_path
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request failed for {url}: {e}")
    logging.error(f"Failed to download {filename} from all known URLs.")
    return None

def extract_text_and_images(pdf_path, image_folder):
    images = {}
    try:
        pages = convert_from_path(pdf_path, output_folder=image_folder, fmt='png', dpi=app.config['OCR_DPI'])
        for i, img in enumerate(pages):
            page_num = i + 1
            img_path = os.path.join(image_folder, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num}.png")
            img.save(img_path)
            images[page_num] = img_path
    except Exception as e:
        logging.error(f"Failed during PDF to image conversion for {pdf_path}: {e}")
    return images

def ocr_specific_pages(image_paths, start_page=0):
    full_text = ""
    for page_num, img_path in sorted(image_paths.items()):
        if page_num > start_page:
             try:
                full_text += f"\n--- PDF Page {page_num} ---\n" + pytesseract.image_to_string(PILImage.open(img_path))
             except Exception as ocr_err:
                logging.warning(f"OCR error on page {page_num} of {os.path.basename(img_path)}: {ocr_err}")
    return full_text

def _download_and_process_pdf_for_extraction(file_info, temp_dir):
    pdf_path = _download_single_pdf(file_info)
    if not pdf_path: return None
    page_images = extract_text_and_images(pdf_path, temp_dir)
    full_text = ocr_specific_pages(page_images, start_page=app.config['PAGES_TO_IGNORE_FOR_PDF_OUTPUT'])
    return {"filename": os.path.basename(pdf_path), "full_text": full_text, "page_images": page_images}

def parse_all_questions_from_text(full_text):
    main_question_start_regex = re.compile(r'^\s*(?:Question\s)?(\d{1,2})[\s\.\(]', re.MULTILINE | re.IGNORECASE)
    matches = list(main_question_start_regex.finditer(full_text))
    if not matches:
        logging.warning("No question numbers found with the regex.")
        return []

    extracted_questions = []
    for i, match in enumerate(matches):
        start_index = match.start()
        end_index = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        question_text_block = full_text[start_index:end_index].strip()

        pages = sorted(list(set(map(int, re.findall(r'--- PDF Page (\d+) ---', question_text_block)))))

        if len(question_text_block) > 25:
            question_number = match.group(1)
            logging.info(f"Parsed question number: {question_number} with text length {len(question_text_block)}")
            extracted_questions.append({
                "question_number": question_number,
                "text": question_text_block,
                "pages": pages if pages else [1]
            })
    return extracted_questions

def get_relevant_question_numbers(full_text, topic, subject_code, level):
    subject_name = subject_codes.get(subject_code, subject_code).replace('-', ' ').title()

    check_prompt = f"""
You are an expert Cambridge {level} {subject_name} examiner acting as a hyper-strict filter.
Your task is to analyze the full OCR text of a past paper and identify ONLY the question numbers that are centrally and primarily about the topic: "{topic}".

**CRITICAL CRITERIA FOR YOUR ANALYSIS:**
1.  **TOPIC RELEVANCE (EXTREMELY STRICT):** The question's main academic focus MUST be "{topic}".
    -   **REJECT** if the question only mentions a keyword from the topic but is fundamentally about something else. For example, if the topic is 'Alternating Current', you MUST REJECT a 'Thermodynamics' question that happens to mention 'current flow'. Similarly, if the topic is 'Waves', you MUST REJECT a question on 'AC Circuits' just because 'frequency' is mentioned. The core concept must match.
    -   **REJECT** if the question is about a related but distinct syllabus topic. Be precise.

2.  **CONTENT COMPLETENESS:** The text for the question must be a full, self-contained question.
    -   **REJECT** if the text is just a fragment, a reference, or clearly incomplete (e.g., "The diagram shows a..." with no further context in the OCR text).

**INSTRUCTIONS:**
-   Analyze the provided text question by question.
-   For each question number, apply the two critical criteria above.
-   Respond with a JSON object containing a single key, "relevant_question_numbers", which is an array of strings.
-   Each string in the array must be a question number that is BOTH relevant AND complete.
-   If no questions meet both criteria, you MUST return an empty array. You will be penalized for including irrelevant or incomplete questions.

**Full OCR Text of the Paper:**
---
{full_text[:24000]}
---
"""

    check_schema = {
        "type": "OBJECT",
        "properties": {
            "relevant_question_numbers": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        },
        "required": ["relevant_question_numbers"]
    }

    logging.info(f"Making a single, strict API call for topic '{topic}'...")
    ai_result = get_json_from_ai_with_retry(check_prompt, check_schema, temperature=0.0, model_type='text')

    if not ai_result or 'relevant_question_numbers' not in ai_result:
        logging.warning(f"AI analysis failed or returned invalid format for topic '{topic}'.")
        return []

    relevant_numbers = ai_result['relevant_question_numbers']
    logging.info(f"AI identified relevant and complete question numbers: {relevant_numbers}")
    return relevant_numbers

# ==============================================================================
# --- API Routes ---
# ==============================================================================
@app.route('/')
def home(): return "AI Exam Helper Backend is running."

@app.route('/get_subject_codes', methods=['GET'])
def get_subject_codes_route():
    return jsonify({"igcse": subject_codes_igcse, "alevel": subject_codes_alevel})

@app.route('/generate_question', methods=['POST'])
def generate_question():
    data = request.get_json()
    if not data or not all(k in data for k in ['subject_code', 'topic', 'level']):
        return jsonify({"success": False, "message": "Missing required parameters."}), 400
    subject_code, topic, level = data['subject_code'], data['topic'], data['level']
    name = subject_codes.get(subject_code, subject_code).replace('-', ' ').title()
    prompt = fr"""
    You are an expert question setter for Cambridge International Examinations (CIE), tasked with creating an authentic, past-paper-style question for {level} {name} on the topic of "{topic}".
    CRITICAL INSTRUCTIONS:
    1. Create a contextual scenario or data stem. This is mandatory.
    2. The question MUST have multiple parts (e.g., (a), (b), (c)) and may include sub-parts (i), (ii).
    3. If the question requires a diagram (like a graph, circuit, or biological drawing), you MUST provide a specification for it in the `diagram_spec` field. If no diagram is needed, this field must be `null`.
       For a graph, the `diagram_spec` object MUST have `type: "graph"`, `title`, `x_label`, `y_label`, and `data` (an array of [x, y] points).
    4. Use standard Cambridge command words (e.g., "State", "Calculate", "Explain", "Plot...").
    5. Assign marks to each sub-part in square brackets, e.g., `[2]`. Total marks should be between 8 and 15.
    6. All mathematical expressions MUST use LaTeX delimiters (`$...$` or `$$...$$`).
    7. The model answer MUST be a point-based marking scheme (e.g., M1, A1, B1).
    Respond in JSON format only.
    """
    # THE FIX: The `type` for `diagram_spec` must be a single string, not a list.
    # The prompt instructs the AI to return `null` for the whole field if not needed,
    # which the JSON parser will handle correctly.
    schema = {
        "type": "OBJECT",
        "properties": {
            "question": {"type": "STRING"},
            "model_answer": {"type": "STRING"},
            "diagram_spec": {
                "type": "OBJECT",
                "nullable": True, # Indicates the object itself can be null
                "properties": {
                    "type": {"type": "STRING"}, "title": {"type": "STRING"}, "x_label": {"type": "STRING"}, "y_label": {"type": "STRING"},
                    "data": {"type": "ARRAY", "items": {"type": "ARRAY", "items": {"type": "NUMBER"}}}
                }
            }
        },
        "required": ["question", "model_answer", "diagram_spec"]
    }
    result = get_json_from_ai_with_retry(prompt, schema, temperature=0.8, model_type='text')
    if not result:
        return jsonify({"success": False, "message": "AI service failed to provide a valid response."}), 500
    
    diagram_image_b64 = None
    diagram_spec = result.get('diagram_spec')
    
    # Check if diagram_spec is a dictionary and has the required keys for plotting
    if isinstance(diagram_spec, dict) and diagram_spec.get('type') == 'graph' and diagram_spec.get('data'):
        try:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            x_vals = [p[0] for p in diagram_spec['data']]
            y_vals = [p[1] for p in diagram_spec['data']]
            ax.plot(x_vals, y_vals, marker='o', linestyle='-')
            ax.set_title(diagram_spec.get('title', ''))
            ax.set_xlabel(diagram_spec.get('x_label', ''))
            ax.set_ylabel(diagram_spec.get('y_label', ''))
            ax.grid(True)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            diagram_image_b64 = base64.b64encode(buf.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Failed to generate diagram from spec: {e}")

    return jsonify({
        "success": True,
        "question": str(result.get('question', '')),
        "model_answer": str(result.get('model_answer', '')),
        "diagram_image": diagram_image_b64
    })

@app.route('/mark_answer', methods=['POST'])
def mark_answer():
    d = request.get_json()
    if not d or not all(k in d for k in ['question', 'user_answer', 'model_answer']):
        return jsonify({"success": False, "message": "Missing required parameters for marking."}), 400
    total_marks_search = re.findall(r'\[(\d+)\]', d['question'])
    max_score = sum(int(m) for m in total_marks_search) if total_marks_search else 10
    prompt = f"""
    As a strict CIE examiner, mark the student's answer based on the provided model answer.
    Question: {d['question']}
    Model Answer: {d['model_answer']}
    Student Answer: {d['user_answer']}
    You must respond in JSON format with these exact keys: "score" (integer out of {max_score}), "feedback" (an object with "strengths" and "improvements" strings), and "corrected_answer" (a string showing the ideal answer).
    """
    schema = {
        "type": "OBJECT", "properties": {
            "score": {"type": "INTEGER"},
            "feedback": {"type": "OBJECT", "properties": {"strengths": {"type": "STRING"}, "improvements": {"type": "STRING"}}, "required": ["strengths", "improvements"]},
            "corrected_answer": {"type": "STRING"}
        }, "required": ["score", "feedback", "corrected_answer"]
    }
    result = get_json_from_ai_with_retry(prompt, schema, model_type='text')
    if not result:
        return jsonify({"success": False, "message": "AI marking service failed."}), 500
    if 'score' in result and isinstance(result['score'], int):
        result['score'] = min(result['score'], max_score)
    return jsonify({"success": True, "max_score": max_score, **result})

@app.route('/generate_past_paper', methods=['POST'])
def generate_past_paper():
    temp_dir = manage_temp_dir('add')
    try:
        data = request.get_json()
        required_keys = ['subject_code', 'component_codes', 'years', 'sessions', 'level']
        if not all(k in data for k in required_keys):
            return jsonify({"success": False, "message": "Missing required parameters (level, subject, etc.)."}), 400

        code = data['subject_code']
        comps = [c.strip() for c in data['component_codes'].split(',') if c.strip()]
        years = data['years']
        sessions = data['sessions']
        level = data['level']

        remove_blank = data.get('remove_blank_pages', False)
        remove_types = data.get('remove_page_types', [])
        ignore_first = data.get('ignore_first_pages', app.config['PAGES_TO_IGNORE_FOR_PDF_OUTPUT'])

        writer = PdfWriter()
        download_tasks = [(code, year, session, comp, level, 'qp', temp_dir) for year in years for session in sessions for comp in comps if not (session == 'm' and not comp.endswith('2'))]

        with concurrent.futures.ThreadPoolExecutor(max_workers=app.config['DOWNLOAD_MAX_WORKERS']) as executor:
            for path in executor.map(_download_single_pdf, download_tasks):
                if path:
                    try:
                        reader = PdfReader(path)
                        start_page = ignore_first if ignore_first < len(reader.pages) else 0

                        if not remove_blank and not remove_types:
                            for i in range(start_page, len(reader.pages)):
                                writer.add_page(reader.pages[i])
                        else:
                            images = extract_text_and_images(path, temp_dir)
                            page_classifications = classify_pages_in_batch_ai(images)

                            for i in range(start_page, len(reader.pages)):
                                page_num = i + 1
                                page_type = page_classifications.get(page_num, "unknown")

                                if page_type == "unknown":
                                    if is_blank_image_pixel_based(images.get(page_num)):
                                        page_type = "blank"

                                keep_page = True
                                if remove_blank and page_type == "blank":
                                    keep_page = False
                                if page_type in remove_types:
                                    keep_page = False

                                if keep_page:
                                    writer.add_page(reader.pages[i])
                                else:
                                    logging.info(f"Removing page {page_num} from {os.path.basename(path)} (type: {page_type})")

                    except PdfReadError as e:
                        logging.warning(f"Could not read PDF {os.path.basename(path)}: {e}")
                    except Exception as e:
                        logging.warning(f"Failed to process {os.path.basename(path)}: {e}")

        if not writer.pages:
            return jsonify({"success": False, "message": "No valid PDF pages could be compiled. Check if the subject and component codes are valid."}), 404

        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)
        return send_file(buf, download_name="compiled_papers.pdf", as_attachment=True, mimetype='application/pdf')
    finally:
        manage_temp_dir('cleanup')

@app.route('/generate_extracted_paper', methods=['POST'])
def generate_extracted_paper():
    temp_dir = manage_temp_dir('add')
    try:
        data = request.get_json()
        required_keys = ['subject_code', 'topic', 'years', 'component_codes', 'level']
        if not all(k in data for k in required_keys):
            return jsonify({"success": False, "message": "Missing required parameters."}), 400

        subject_code, topic, years, component_codes_str, level = data['subject_code'], data['topic'], data['years'], data['component_codes'], data['level']
        component_codes_list = [c.strip() for c in component_codes_str.split(',') if c.strip()]

        download_tasks = [(subject_code, year, session, comp, level, 'qp', temp_dir) for year in years for session in ['s', 'w', 'm'] for comp in component_codes_list if not (session == 'm' and not comp.endswith('2'))]

        processed_papers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=app.config['PROCESSING_MAX_WORKERS']) as executor:
            future_to_task = {executor.submit(_download_and_process_pdf_for_extraction, task, temp_dir): task for task in download_tasks}
            for future in concurrent.futures.as_completed(future_to_task):
                result = future.result()
                if result: processed_papers.append(result)

        if not processed_papers:
            return jsonify({"success": False, "message": "Could not download any past papers."}), 404

        final_questions_for_pdf = []
        unique_question_hashes = set()

        for paper in processed_papers:
            if not paper['full_text']:
                logging.warning(f"Skipping paper {paper['filename']} due to empty OCR text.")
                continue

            relevant_q_numbers = get_relevant_question_numbers(paper['full_text'], topic, subject_code, level)
            if not relevant_q_numbers:
                logging.info(f"No relevant questions found by AI in paper {paper['filename']}.")
                continue

            all_questions_in_paper = parse_all_questions_from_text(paper['full_text'])
            question_map = {q['question_number']: q for q in all_questions_in_paper}

            for q_num in relevant_q_numbers:
                normalized_q_num = str(q_num).strip()
                if normalized_q_num in question_map:
                    q_data = question_map[normalized_q_num]
                    question_hash = hashlib.md5(re.sub(r'\s+', '', q_data['text']).lower().encode()).hexdigest()
                    if question_hash not in unique_question_hashes:
                        logging.info(f"Adding question {q_num} from {paper['filename']} to the final PDF.")
                        final_questions_for_pdf.append({
                            "question_number": q_data['question_number'],
                            "text": q_data['text'],
                            "pages": q_data['pages'],
                            "source_filename": paper['filename'],
                            "page_images": paper['page_images']
                        })
                        unique_question_hashes.add(question_hash)
                else:
                    logging.warning(f"AI returned q_num '{q_num}' which was not found in parsed map of {paper['filename']}. Available keys: {list(question_map.keys())}");

        if not final_questions_for_pdf:
            return jsonify({"success": False, "message": f"No questions found for the topic '{topic}'. The AI filter is very strict."}), 404

        output_pdf_path = os.path.join(temp_dir, f"extracted_{topic}.pdf")
        doc = SimpleDocTemplate(output_pdf_path, pagesize=A4, leftMargin=inch/2, rightMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
        styles = getSampleStyleSheet()
        error_style = ParagraphStyle(name='ErrorStyle', parent=styles['Normal'], textColor='red', alignment=TA_CENTER)
        elements = [Paragraph(f"<b>{topic} - Extracted Questions</b>", styles['Title']), Spacer(1, 0.2 * inch)]

        final_questions_for_pdf.sort(key=lambda x: (x['source_filename'], int(x['question_number'])))

        for q in final_questions_for_pdf:
            elements.append(Paragraph(f"<b>Source: {q['source_filename']} - Question {q['question_number']}</b>", styles['Heading2']))

            all_pages_for_question = sorted(list(set(q['pages'])))
            for p_num in all_pages_for_question:
                img_path = q['page_images'].get(p_num)
                if img_path and os.path.exists(img_path) and not is_blank_image_pixel_based(img_path):
                    elements.append(Spacer(1, 0.1 * inch))
                    try:
                        elements.append(ReportLabImage(img_path, width=A4[0] - 1*inch, height=A4[1] - 1.5*inch, hAlign='CENTER', kind='bound'))
                    except Exception as img_err:
                        logging.error(f"Could not add image {img_path} to PDF: {img_err}")
                        elements.append(Paragraph(f"[Error: Could not render image for page {p_num}]", error_style))

            elements.append(PageBreak())

        if elements and isinstance(elements[-1], PageBreak):
            elements.pop()

        doc.build(elements)
        return send_file(output_pdf_path, as_attachment=True, download_name=f"extracted_{topic.replace(' ', '_')}.pdf", mimetype='application/pdf')
    finally:
        manage_temp_dir('cleanup')

if __name__ == '__main__':
    if not _api_keys or _api_keys[0] in _placeholder_keys:
        logging.fatal("AI features disabled due to missing API key. Please provide a valid key in the environment variables or config.")
    if not Config.TESSERACT_CMD:
        logging.fatal("OCR features disabled because Tesseract is not in PATH.")
    from waitress import serve
    serve(app, host="0.0.0.0", port=10000)
