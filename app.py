# ==============================================================================
# --- Cambridge Exam Helper - AI-Powered Past Paper Application ---
# ==============================================================================
#
# Version:      v140 (Final Stable Synchronous Version)
# Description:  A stable, synchronous backend. This version uses a robust
#               text-delimiter method for AI generation to guarantee valid JSON
#               and fix all formatting and stability issues.
#
# ==============================================================================

# --- 1. IMPORTS ---
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
import base64
import traceback
from typing import List, Dict, Optional, Tuple, Any, Union
from threading import Thread

# --- Third-party library imports ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.errors import PdfReadError
from pdf2image import convert_from_path
from PIL import Image as PILImage
import pytesseract
import google.generativeai as genai

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER

# ==============================================================================
# --- 2. CONFIGURATION & INITIALIZATION ---
# ==============================================================================
print("--- Running app.py version: v140 (Final Stable Synchronous Version) ---")

class Config:
    """Centralized configuration class for the application."""
    API_KEYS: List[str] = os.environ.get("API_KEYS", "AIzaSyC7Cyq4I5p-pV1NLuIllAnfJToEVonK23Q,AIzaSyBzghsTx_tsMbp3GvQRlZ8TLJvEY6ae4Gc").split(',')
    TESSACT_CMD: Optional[str] = shutil.which('tesseract')
    MAX_AI_RETRIES: int = 3
    AI_RETRY_DELAY_SECONDS: int = 5
    API_CALL_INTERVAL_SECONDS: int = 1
    AI_MODEL_VISION: str = 'gemini-1.5-flash'
    AI_MODEL_TEXT: str = 'gemini-1.5-flash'
    PDF_DOWNLOAD_TIMEOUT_SECONDS: int = 30
    DOWNLOAD_MAX_WORKERS: int = 8
    PROCESSING_MAX_WORKERS: int = 8
    OCR_DPI: int = 200
    BLANK_IMAGE_THRESHOLD: int = 245
    BLANK_IMAGE_PIXEL_FRACTION: float = 0.995
    CACHE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, expose_headers=['X-Processing-Message'])
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

if Config.TESSACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = Config.TESSACT_CMD
    try:
        tesseract_version = pytesseract.get_tesseract_version()
        logging.info(f"Tesseract version {tesseract_version} found and configured.")
    except Exception as e:
        logging.error(f"Tesseract found but failed to get version. Error: {e}")
else:
    logging.error("TESSERACT NOT FOUND. OCR features will be disabled.")

_api_keys: List[str] = app.config['API_KEYS']
_current_key_index: int = 0
_placeholder_keys: List[str] = ["", "YOUR_API_KEY_HERE"]

if _api_keys and _api_keys[0] not in _placeholder_keys:
    logging.info(f"Google AI configured with {len(_api_keys)} key(s).")
else:
    logging.error("Google AI API key is not configured or is a placeholder. AI features will be disabled.")

def setup_cache() -> None:
    """Creates the cache directory if it doesn't exist."""
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    logging.info(f"Cache directory is set to: {Config.CACHE_DIR}")

# ==============================================================================
# --- 3. SUBJECT CODE DATA ---
# ==============================================================================
subject_codes_igcse: Dict[str, str] = {
    "0452": "accounting-igcse", "0985": "accounting-9-1-igcse", "0548": "afrikaans-second-language-igcse", "0600": "agriculture-igcse", "0508": "arabic-first-language-igcse", "7184": "arabic-9-1-first-language-igcse", "0544": "arabic-foreign-language-igcse", "7180": "arabic-9-1-igcse", "0400": "art-design-igcse", "0989": "art-design-9-1-igcse", "0538": "bahasa-indonesia-igcse", "0610": "biology-igcse", "0970": "biology-9-1-igcse", "0450": "business-studies-igcse", "0986": "business-studies-9-1-igcse", "0620": "chemistry-igcse", "0971": "chemistry-9-1-igcse", "0509": "chinese-first-language-igcse", "0523": "chinese-second-language-igcse", "0547": "chinese-mandarin-foreign-language-igcse", "0478": "computer-science-igcse", "0984": "computer-science-9-1-igcse", "0445": "design-technology-igcse", "0979": "design-technology-9-1-igcse", "0411": "drama-igcse", "0994": "drama-9-1-igcse", "0455": "economics-igcse", "0987": "economics-9-1-igcse", "0500": "english-first-language-igcse", "0990": "english-first-language-9-1-igcse", "0475": "english-literature-igcse", "0992": "english-literature-9-1-igcse", "0511": "english-second-language-count-in-speaking-igcse", "0991": "english-second-language-9-1-count-in-speaking-igcse", "0510": "english-second-language-speaking-endorsement-igcse", "0993": "english-second-language-speaking-endorsement-9-1-igcse", "0454": "enterprise-igcse", "0680": "environmental-management-igcse", "0648": "food-nutrition-igcse", "0501": "french-first-language-igcse", "0520": "french-foreign-language-igcse", "7156": "french-9-1-igcse", "0460": "geography-igcse", "0976": "geography-9-1-igcse", "0505": "german-first-language-igcse", "0525": "german-foreign-language-igcse", "7159": "german-9-1-igcse", "0457": "global-perspectives-igcse", "0549": "hindi-second-language-igcse", "0470": "history-igcse", "0977": "history-9-1-igcse", "0417": "ict-igcse", "0983": "ict-9-1-igcse", "0493": "islamiyat-igcse", "0535": "italian-foreign-language-igcse", "7164": "italian-9-1-igcse", "0546": "malay-foreign-language-igcse", "0697": "marine-science-igcse", "0580": "mathematics-igcse", "0606": "mathematics-additional-igcse", "0607": "mathematics-international-igcse", "0980": "mathematics-9-1-igcse", "0410": "music-igcse", "0978": "music-9-1-igcse", "0448": "pakistan-studies-igcse", "0413": "physical-education-igcse", "0995": "physical-education-9-1-igcse", "0652": "physical-science-igcse", "0625": "physics-igcse", "0972": "physics-9-1-igcse", "0990": "psychology-igcse", "0490": "religious-studies-igcse", "0653": "science-combined-igcse", "0654": "sciences-coordinated-double-igcse", "0495": "sociology-igcse", "0502": "spanish-first-language-igcse", "0530": "spanish-foreign-language-igcse", "7160": "spanish-9-1-igcse", "0471": "travel-tourism-igcse", "0539": "urdu-second-language-igcse"
}
subject_codes_alevel: Dict[str, str] = {
    "9706": "accounting-alevel", "9680": "arabic-alevel", "9479": "art-design-alevel", "9700": "biology-alevel", "9609": "business-alevel", "9701": "chemistry-alevel", "9274": "classical-studies-alevel", "9618": "computer-science-alevel", "9705": "design-technology-alevel", "9482": "drama-alevel", "9708": "economics-alevel", "9093": "english-language-alevel", "9695": "english-literature-alevel", "8021": "english-general-paper-as-alevel", "8291": "environmental-management-as-alevel", "9716": "french-alevel", "9696": "geography-alevel", "9717": "german-alevel", "9239": "global-perspectives-research-alevel", "9489": "history-alevel", "9626": "information-technology-alevel", "9084": "law-alevel", "9693": "marine-science-alevel", "9709": "mathematics-alevel", "9231": "further-mathematics-alevel", "9607": "media-studies-alevel", "9483": "music-alevel", "9702": "physics-alevel", "9990": "psychology-alevel", "9699": "sociology-alevel", "9718": "portuguese-alevel", "9694": "thinking-skills-alevel", "9395": "travel-tourism-alevel"
}
subject_codes: Dict[str, str] = {**subject_codes_igcse, **subject_codes_alevel}

# ==============================================================================
# --- 4. CORE AI & DATA PROCESSING UTILITIES ---
# ==============================================================================

def get_ai_response(prompt: Union[str, List[Any]], temperature: float = 0.2, model_type: str = 'text', is_json: bool = False) -> Optional[Union[Dict[str, Any], str]]:
    """A robust function to get a response from Google's AI, with retries and key rotation."""
    global _current_key_index
    if not _api_keys or _api_keys[0] in _placeholder_keys:
        logging.error("AI call attempted but no valid API key is configured.")
        return None

    if isinstance(prompt, list):
        prompt = tuple(prompt)

    generation_config_args = {"temperature": temperature, "max_output_tokens": 8192}
    if is_json:
        generation_config_args["response_mime_type"] = "application/json"

    for attempt in range(app.config['MAX_AI_RETRIES']):
        for i in range(len(_api_keys)):
            current_key_index_for_attempt = (_current_key_index + i) % len(_api_keys)
            try:
                logging.info(f"AI call attempt {attempt + 1}, using key index {current_key_index_for_attempt}...")
                genai.configure(api_key=_api_keys[current_key_index_for_attempt])
                model = genai.GenerativeModel(app.config['AI_MODEL_TEXT'] if model_type == 'text' else app.config['AI_MODEL_VISION'])
                
                generation_config = genai.types.GenerationConfig(**generation_config_args)
                response = model.generate_content(prompt, generation_config=generation_config)
                
                _current_key_index = current_key_index_for_attempt
                raw_text = response.candidates[0].content.parts[0].text
                
                if is_json:
                    return json.loads(raw_text)
                else:
                    return raw_text
            except Exception as e:
                logging.error(f"Exception on AI call with key index {current_key_index_for_attempt}: {e}")
                if '429' in str(e) or 'rate limit' in str(e).lower():
                    logging.warning(f"Rate limit on key index {current_key_index_for_attempt}. Trying next key.")
                    continue
        logging.warning(f"All API keys failed in attempt cycle {attempt + 1}. Waiting {app.config['AI_RETRY_DELAY_SECONDS']}s.")
        time.sleep(app.config['AI_RETRY_DELAY_SECONDS'])
    logging.error("All AI call attempts with all API keys failed.")
    return None

def classify_single_paper_ai(filename: str, image_paths_dict: Dict[int, str]) -> Dict[int, str]:
    """Uses AI to classify pages of a single paper, relying on instruction following."""
    if not image_paths_dict: return {}
    logging.info(f"AI analyzing {len(image_paths_dict)} pages for {filename}...")
    prompt_parts = [
        "Analyze the following exam paper pages. Classify each into a single category: 'question', 'formula_sheet', 'periodic_table', 'instructions', or 'blank'.",
        "Respond with a single JSON object where keys are page numbers (as strings) and values are the category.",
        "Example: { \"1\": \"instructions\", \"2\": \"question\", \"3\": \"blank\" }"
    ]
    for page_num, path in sorted(image_paths_dict.items()):
        prompt_parts.extend([f"--- PAGE: {page_num} ---", PILImage.open(path)])
    
    result = get_ai_response(prompt_parts, temperature=0.0, model_type='vision', is_json=True)
    return {int(k): v for k, v in result.items() if k.isdigit()} if isinstance(result, dict) else {}

def identify_relevant_questions_in_bulk_ai(all_papers_text: str, topic: str, subject_code: str, level: str) -> Dict[str, List[str]]:
    """Uses AI to find questions on a specific topic across multiple papers."""
    subject_name = subject_codes.get(subject_code, subject_code).replace('-', ' ').title()
    prompt = f"""You are an expert Cambridge {level} {subject_name} examiner. Analyze the OCR text from multiple past papers and identify questions primarily about the topic: "{topic}". Be extremely strict. Reject questions that only mention keywords but are about a different concept. Respond with a single JSON object where keys are filenames (e.g., "9702_s23_qp_41.pdf") and values are an array of relevant question numbers (as strings). Example: {{"9702_s23_qp_41.pdf": ["5", "7"], "9702_m24_qp_42.pdf": ["4"]}}. Full OCR Text of All Papers:\n---\n{all_papers_text[:1000000]}\n---"""
    logging.info(f"Making a single bulk API call for topic '{topic}' across all papers...")
    ai_result = get_ai_response(prompt, temperature=0.0, model_type='text', is_json=True)
    return ai_result if isinstance(ai_result, dict) else {}

def is_blank_image_pixel_based(image_path: str) -> bool:
    """Determines if an image is blank by checking pixel brightness."""
    if not image_path or not os.path.exists(image_path): return True
    with PILImage.open(image_path).convert("L") as img:
        pixels = list(img.getdata())
        if not pixels: return True
        bright_pixels = sum(1 for p in pixels if p > app.config['BLANK_IMAGE_THRESHOLD'])
        return (bright_pixels / len(pixels)) > app.config['BLANK_IMAGE_PIXEL_FRACTION']

def get_potential_pdf_urls(subject_code: str, year: str, session: str, component: str, level: str, paper_type: str) -> List[str]:
    """Generates a list of possible download URLs for a given past paper."""
    year_short = year[-2:]
    level_url_part = {"igcse": "Cambridge%20IGCSE", "alevel": "A%20Levels"}.get(level, "Miscellaneous")
    subject_name_raw = subject_codes.get(subject_code, "unknown-subject")
    subject_name_clean = subject_name_raw.split('-')[0].replace('-', ' ').title()
    subject_url_part = f"{subject_name_clean.replace(' ', '%20')}%20({subject_code})"
    return [
        f"https://papers.gceguide.com/{level_url_part}/{subject_url_part}/{year}/{subject_code}_{session}{year_short}_{paper_type}_{component}.pdf",
        f"https://pastpapers.papacambridge.com/directories/CAIE/CAIE-pastpapers/upload/{subject_code}_{session}{year_short}_{paper_type}_{component}.pdf",
    ]

def _download_single_pdf(url: str, destination_path: str) -> bool:
    """Downloads a single PDF from a URL to a destination path."""
    try:
        response = requests.get(url, stream=True, timeout=app.config['PDF_DOWNLOAD_TIMEOUT_SECONDS'])
        if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
            with open(destination_path, 'wb') as f: shutil.copyfileobj(response.raw, f)
            logging.info(f"Successfully downloaded to: {destination_path} from {url}")
            return True
    except requests.exceptions.RequestException as e:
        logging.warning(f"Request failed for {url}: {e}")
    return False

def extract_images_from_pdf(pdf_path: str, image_folder: str) -> Dict[int, str]:
    """Converts a PDF to images and saves them."""
    images = {}
    try:
        file_prefix = uuid.uuid4().hex
        pages = convert_from_path(pdf_path, output_folder=image_folder, fmt='png', dpi=app.config['OCR_DPI'], output_file=file_prefix)
        for i, img in enumerate(pages): images[i + 1] = img.filename
    except Exception as e: logging.error(f"Failed during PDF to image conversion for {pdf_path}: {e}")
    return images

def ocr_specific_pages(image_paths: Dict[int, str], start_page: int = 0) -> str:
    """Performs OCR on a dictionary of image paths."""
    full_text = ""
    for page_num, img_path in sorted(image_paths.items()):
        if page_num > start_page:
             try:
                full_text += f"\n--- PDF Page {page_num} ---\n" + pytesseract.image_to_string(PILImage.open(img_path))
             except Exception as ocr_err:
                logging.warning(f"OCR error on page {page_num} of {os.path.basename(img_path)}: {ocr_err}")
    return full_text

def parse_all_questions_from_text(full_text: str) -> List[Dict[str, Any]]:
    """Parses OCR text to identify and split individual questions."""
    main_question_start_regex = re.compile(r'\n\s*(\d{1,2})[\s\.\(]', re.MULTILINE)
    matches = list(main_question_start_regex.finditer(full_text))
    if not matches: return []
    extracted_questions = []
    for i, match in enumerate(matches):
        start_index = match.start()
        end_index = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        question_text_block = full_text[start_index:end_index].strip()
        pages = sorted(list(set(map(int, re.findall(r'--- PDF Page (\d+) ---', question_text_block)))))
        if len(question_text_block) > 25:
            extracted_questions.append({"question_number": match.group(1), "text": question_text_block, "pages": pages})
    return extracted_questions

# ==============================================================================
# --- 5. CACHING & BACKGROUND PROCESSING ---
# ==============================================================================

def _ensure_pdf_is_cached(file_info: Tuple[str, ...], pdf_path: str) -> bool:
    """Handles the caching logic for a single PDF file."""
    if os.path.exists(pdf_path):
        logging.info(f"CACHE HIT (PDF): Found {os.path.basename(pdf_path)}")
        return True
    
    logging.info(f"CACHE MISS (PDF): Downloading {os.path.basename(pdf_path)}")
    subject_code, year, session, comp_part, level, paper_type = file_info
    for url in get_potential_pdf_urls(subject_code, year, session, comp_part, level, paper_type):
        if _download_single_pdf(url, pdf_path):
            return True
    
    logging.error(f"Failed to download {os.path.basename(pdf_path)} from all sources.")
    return False

def _ensure_images_are_cached(pdf_path: str, images_dir: str) -> Dict[int, str]:
    """Handles the caching logic for PDF page images."""
    if os.path.exists(images_dir) and os.listdir(images_dir):
        logging.info(f"CACHE HIT (Images): Loading images for {os.path.basename(pdf_path)}")
        return {int(re.search(r'page_(\d+)', f).group(1)): os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')}

    logging.info(f"CACHE MISS (Images): Extracting images for {os.path.basename(pdf_path)}")
    os.makedirs(images_dir, exist_ok=True)
    page_images = {}
    with tempfile.TemporaryDirectory() as temp_conv_dir:
        temp_page_images = extract_images_from_pdf(pdf_path, temp_conv_dir)
        for page_num, temp_path in temp_page_images.items():
            final_path = os.path.join(images_dir, f"page_{page_num}.png")
            shutil.move(temp_path, final_path)
            page_images[page_num] = final_path
    return page_images

def _ensure_ocr_is_cached(ocr_path: str, page_images: Dict[int, str]) -> str:
    """Handles the caching logic for OCR text."""
    if os.path.exists(ocr_path):
        logging.info(f"CACHE HIT (OCR): Loading OCR text from {os.path.basename(ocr_path)}")
        with open(ocr_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    logging.info(f"CACHE MISS (OCR): Performing OCR and saving to {os.path.basename(ocr_path)}")
    full_text = ocr_specific_pages(page_images)
    with open(ocr_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    return full_text

def _ensure_classification_is_cached(classification_path: str, filename_base: str, page_images: Dict[int, str]) -> Dict[int, str]:
    """Handles the caching logic for AI page classifications."""
    if os.path.exists(classification_path):
        logging.info(f"CACHE HIT (Classification): Loading from {os.path.basename(classification_path)}")
        with open(classification_path, 'r', encoding='utf-8') as f:
            return {int(k): v for k, v in json.load(f).items()}

    logging.info(f"CACHE MISS (Classification): Running AI classification for {filename_base}")
    non_blank_images = {p: path for p, path in page_images.items() if not is_blank_image_pixel_based(path)}
    ai_classifications = classify_single_paper_ai(filename_base, non_blank_images)
    
    classifications = {p: 'blank' for p in page_images if p not in non_blank_images}
    classifications.update(ai_classifications)
    
    with open(classification_path, 'w', encoding='utf-8') as f:
        json.dump(classifications, f)
    return classifications

def get_or_create_paper_data(file_info: Tuple[str, ...], required_data: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    """Orchestrates the process of retrieving all necessary paper data, utilizing caching at each step."""
    subject_code, year, session, comp_part, level, paper_type = file_info
    filename_base = f"{subject_code}_{session}{year[-2:]}_{paper_type}_{comp_part}"
    paper_cache_dir = os.path.join(Config.CACHE_DIR, filename_base)
    os.makedirs(paper_cache_dir, exist_ok=True)

    pdf_path = os.path.join(paper_cache_dir, f"{filename_base}.pdf")
    images_dir = os.path.join(paper_cache_dir, 'images')
    ocr_path = os.path.join(paper_cache_dir, 'ocr.json')
    classification_path = os.path.join(paper_cache_dir, 'classification.json')

    if not _ensure_pdf_is_cached(file_info, pdf_path):
        return None

    page_images = {}
    if any(item in required_data for item in ['images', 'ocr', 'classification']):
        page_images = _ensure_images_are_cached(pdf_path, images_dir)
        if not page_images: return None

    full_text = ""
    if 'ocr' in required_data:
        full_text = _ensure_ocr_is_cached(ocr_path, page_images)

    classifications = {}
    if 'classification' in required_data:
        classifications = _ensure_classification_is_cached(classification_path, filename_base, page_images)

    return {"filename": f"{filename_base}.pdf", "pdf_path": pdf_path, "page_images": page_images, "full_text": full_text, "classifications": classifications}

def process_paper_in_background(file_info: Tuple[str, ...], required_data: Tuple[str, ...]) -> None:
    """Target function for a background thread to process a single paper."""
    logging.info(f"BACKGROUND: Starting to process {file_info}")
    try:
        get_or_create_paper_data(file_info, required_data)
        logging.info(f"BACKGROUND: Successfully processed {file_info}")
    except Exception as e:
        logging.error(f"BACKGROUND: Error processing {file_info}: {e}")

# ==============================================================================
# --- 6. HELPER FUNCTIONS FOR API ROUTES ---
# ==============================================================================

def _create_diagram_from_spec(diagram_spec: Dict[str, Any]) -> Optional[str]:
    """Generates a base64 encoded PNG image from a diagram specification."""
    try:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        x_vals, y_vals = zip(*diagram_spec['data'])
        ax.plot(x_vals, y_vals, marker='o', linestyle='-')
        ax.set(title=diagram_spec.get('title', ''), xlabel=diagram_spec.get('x_label', ''), ylabel=diagram_spec.get('y_label', ''))
        ax.grid(True)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Failed to generate diagram from spec: {e}")
        return None

def _build_compiled_pdf(writer: PdfWriter, papers_to_add: List[Dict[str, Any]], ignore_first: int, remove_blank: bool, remove_types: List[str]) -> None:
    """Helper to add pages from a list of processed papers to a PdfWriter object."""
    for pdf_data in papers_to_add:
        try:
            reader = PdfReader(pdf_data['pdf_path'])
            start_page = ignore_first if ignore_first < len(reader.pages) else 0
            for i in range(start_page, len(reader.pages)):
                page_num = i + 1
                page_type = pdf_data.get('classifications', {}).get(page_num, "unknown")
                if not (remove_blank and page_type == "blank") and (page_type not in remove_types):
                    writer.add_page(reader.pages[i])
                else:
                    logging.info(f"Removing page {page_num} from {pdf_data['filename']} (type: {page_type})")
        except Exception as e:
            logging.warning(f"Failed to process {pdf_data['filename']}: {e}")

def _build_extracted_pdf(output_path: str, topic: str, questions: List[Dict[str, Any]]) -> None:
    """Builds the final PDF for extracted questions using ReportLab."""
    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=inch/2, rightMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
    styles = getSampleStyleSheet()
    error_style = ParagraphStyle(name='ErrorStyle', parent=styles['Normal'], textColor='red', alignment=TA_CENTER)
    elements = [Paragraph(f"<b>{topic} - Extracted Questions</b>", styles['Title']), Spacer(1, 0.2 * inch)]
    
    for q in sorted(questions, key=lambda x: (x['source_filename'], int(x['question_number']))):
        elements.append(Paragraph(f"<b>Source: {q['source_filename']} - Question {q['question_number']}</b>", styles['Heading2']))
        for p_num in sorted(list(set(q['pages']))):
            if (img_path := q['page_images'].get(p_num)) and os.path.exists(img_path) and not is_blank_image_pixel_based(img_path):
                elements.append(Spacer(1, 0.1 * inch))
                try:
                    elements.append(ReportLabImage(img_path, width=A4[0] - inch, height=A4[1] - 1.5*inch, hAlign='CENTER', kind='bound'))
                except Exception as e:
                    elements.append(Paragraph(f"[Error: Could not render image for page {p_num}: {e}]", error_style))
        elements.append(PageBreak())

    if elements and isinstance(elements[-1], PageBreak): elements.pop()
    doc.build(elements)

def _validate_request(data: Optional[Dict[str, Any]], required_keys: List[str]) -> Optional[str]:
    """Validates incoming request data."""
    if not data:
        return "Missing request body."
    for key in required_keys:
        if key not in data or not data[key]:
            return f"Missing required parameter: '{key}'."
    return None

# ==============================================================================
# --- 7. API ROUTES ---
# ==============================================================================

@app.route('/')
def home() -> str:
    """A simple endpoint to confirm the backend is running."""
    return "AI Exam Helper Backend is running."

@app.route('/health')
def health_check() -> Response:
    """Health check endpoint for monitoring."""
    return jsonify({"status": "ok"}), 200

@app.route('/get_subject_codes', methods=['GET'])
def get_subject_codes_route() -> Response:
    """Provides the list of supported subject codes to the frontend."""
    return jsonify({"igcse": subject_codes_igcse, "alevel": subject_codes_alevel})

@app.route('/clear_cache', methods=['POST'])
def clear_cache_route() -> Response:
    """Endpoint to clear the entire paper cache."""
    try:
        if os.path.exists(Config.CACHE_DIR):
            shutil.rmtree(Config.CACHE_DIR)
            logging.info("Cache directory removed.")
        setup_cache()
        return jsonify({"success": True, "message": "Cache cleared successfully."}), 200
    except Exception as e:
        logging.error(f"Error clearing cache: {e}")
        return jsonify({"success": False, "message": "Failed to clear cache."}), 500

@app.route('/generate_question', methods=['POST'])
def generate_question() -> Response:
    """Generates a new exam question using AI."""
    try:
        data = request.get_json()
        if error_msg := _validate_request(data, ['subject_code', 'topic', 'level']):
            return jsonify({"success": False, "message": error_msg}), 400

        subject_code, topic, level = data['subject_code'], data['topic'], data['level']
        name = subject_codes.get(subject_code, subject_code).replace('-', ' ').title()

        prompt = fr"""
        You are an expert Cambridge examiner for {level} {name}.
        Your first task is to validate if "{topic}" is a real, central topic in this subject.
        - If the topic is INVALID, respond with the single word "INVALID" and nothing else.
        - If the topic is VALID, create an authentic, multi-part exam question.

        CRITICAL INSTRUCTIONS FOR QUESTION GENERATION:
        - **Formatting:** Each part of the question (e.g., (a), (b), (i)) MUST start on a new line.
        - **Marks:** Marks for each part MUST be in square brackets at the end of the line, e.g., [2]. The total marks should be between 8 and 15.
        - **Delimiter:** After the entire question is finished, you MUST include the exact delimiter '|||ANSWER|||' on its own line.
        - **Model Answer:** After the delimiter, provide a point-based marking scheme, with each point on a new line.
        """
        
        raw_response = get_ai_response(prompt, temperature=0.8, model_type='text', is_json=False)

        if not raw_response:
            return jsonify({"success": False, "message": "AI service failed to provide a valid response."}), 500
        
        if "INVALID" in raw_response or '|||ANSWER|||' not in raw_response:
            return jsonify({"success": False, "message": f"The AI determined that '{topic}' is not a valid topic for this subject."}), 400

        question_text, model_answer_text = raw_response.split('|||ANSWER|||', 1)

        return jsonify({
            "success": True,
            "question": question_text.strip(),
            "model_answer": model_answer_text.strip(),
            "diagram_spec": None # Diagram generation can be re-added later if needed
        })
    except Exception as e:
        logging.error(f"Unhandled error in generate_question: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": "An unexpected server error occurred."}), 500

@app.route('/mark_answer', methods=['POST'])
def mark_answer() -> Response:
    """Marks a user's answer using AI."""
    try:
        data = request.get_json()
        if error_msg := _validate_request(data, ['question', 'user_answer', 'model_answer']):
            return jsonify({"success": False, "message": error_msg}), 400
        
        max_score = sum(int(m) for m in re.findall(r'\[(\d+)\]', data['question'])) or 10
        prompt = f"""As a strict CIE examiner, mark the student's answer based on the provided model answer. Question: {data['question']}\nModel Answer: {data['model_answer']}\nStudent Answer: {data['user_answer']}\nRespond in JSON with "score" (integer out of {max_score}), "feedback" (object with "strengths" and "improvements"), and "corrected_answer"."""
        schema = {"type": "OBJECT", "properties": {"score": {"type": "INTEGER"}, "feedback": {"type": "OBJECT", "properties": {"strengths": {"type": "STRING"}, "improvements": {"type": "STRING"}}, "required": ["strengths", "improvements"]}, "corrected_answer": {"type": "STRING"}}, "required": ["score", "feedback", "corrected_answer"]}
        
        result = get_ai_response(prompt, schema=schema, model_type='text', is_json=True)
        if not result: return jsonify({"success": False, "message": "AI marking service failed."}), 500
        
        if 'score' in result and isinstance(result.get('score'), int): result['score'] = min(result['score'], max_score)
        return jsonify({"success": True, "max_score": max_score, **result})
    except Exception as e:
        logging.error(f"Unhandled error in mark_answer: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": "An unexpected server error occurred."}), 500

@app.route('/generate_past_paper', methods=['POST'])
def generate_past_paper() -> Response:
    """Compiles multiple past papers and optionally their mark schemes into a single PDF."""
    try:
        data = request.get_json()
        if error_msg := _validate_request(data, ['subject_code', 'component_codes', 'years', 'sessions', 'level']):
            return jsonify({"success": False, "message": error_msg}), 400

        code, comps, years, sessions, level = data['subject_code'], [c.strip() for c in data['component_codes'].split(',') if c.strip()], data['years'], data['sessions'], data['level']
        remove_blank = data.get('remove_blank_pages', False)
        remove_types = data.get('remove_page_types', [])
        ignore_first = data.get('ignore_first_pages', 0)
        include_ms = data.get('include_mark_schemes', False)

        tasks = []
        for year in years:
            for session in sessions:
                for comp in comps:
                    if not (session == 'm' and not comp.endswith('2')):
                        tasks.append((code, year, session, comp, level, 'qp'))
                        if include_ms:
                            tasks.append((code, year, session, comp, level, 'ms'))
        
        ready_papers, pending_papers = [], []
        for task in tasks:
            filename_base = f"{task[0]}_{task[2]}{str(task[1])[-2:]}_{task[5]}_{task[3]}"
            pdf_path = os.path.join(Config.CACHE_DIR, filename_base, f"{filename_base}.pdf")
            if os.path.exists(pdf_path):
                paper_data = get_or_create_paper_data(task, ('classification',))
                if paper_data:
                    ready_papers.append(paper_data)
            else:
                pending_papers.append(f"{filename_base}.pdf")
                thread = Thread(target=process_paper_in_background, args=(task, ('classification',)))
                thread.daemon = True
                thread.start()

        writer = PdfWriter()
        _build_compiled_pdf(writer, sorted(ready_papers, key=lambda p: p['filename']), ignore_first, remove_blank, remove_types)

        if not writer.pages and not pending_papers:
            return jsonify({"success": False, "message": "No valid PDF pages could be found or compiled."}), 404

        if not writer.pages and pending_papers:
             return jsonify({
                "success": True,
                "message": f"No papers were ready in the cache. The following {len(pending_papers)} papers are now being processed in the background. Please try your request again in a few minutes.",
                "processing_files": pending_papers
            }), 202

        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)
        
        response = send_file(buf, download_name="compiled_papers.pdf", as_attachment=True, mimetype='application/pdf')
        if pending_papers:
            response.headers['X-Processing-Message'] = f"Compiled {len(ready_papers)} papers. The following {len(pending_papers)} papers are being processed in the background: {', '.join(pending_papers)}"
        
        return response
    except Exception as e:
        logging.error(f"Unhandled error in generate_past_paper: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": "An unexpected server error occurred."}), 500

@app.route('/generate_extracted_paper', methods=['POST'])
def generate_extracted_paper() -> Response:
    """Extracts questions on a specific topic into a new PDF."""
    try:
        data = request.get_json()
        if error_msg := _validate_request(data, ['subject_code', 'topic', 'years', 'component_codes', 'level']):
            return jsonify({"success": False, "message": error_msg}), 400

        subject_code, topic, years, comps_str, level = data['subject_code'], data['topic'], data['years'], data['component_codes'], data['level']
        comps = [c.strip() for c in comps_str.split(',') if c.strip()]
        tasks = [(subject_code, year, session, comp, level, 'qp') for year in years for session in ['s', 'w', 'm'] for comp in comps if not (session == 'm' and not comp.endswith('2'))]

        processed_papers_map = {}
        pending_papers = []
        for task in tasks:
            filename_base = f"{task[0]}_{task[2]}{str(task[1])[-2:]}_{task[5]}_{task[3]}"
            pdf_path = os.path.join(Config.CACHE_DIR, filename_base, f"{filename_base}.pdf")
            if os.path.exists(pdf_path):
                paper_data = get_or_create_paper_data(task, ('images', 'ocr'))
                if paper_data:
                    processed_papers_map[paper_data['filename']] = paper_data
            else:
                pending_papers.append(f"{filename_base}.pdf")
                thread = Thread(target=process_paper_in_background, args=(task, ('images', 'ocr')))
                thread.daemon = True
                thread.start()

        if not processed_papers_map:
            message = f"No papers were ready in the cache to search for the topic '{topic}'. The required papers are now being processed in the background. Please try again in a few minutes."
            return jsonify({"success": True, "message": message, "processing_files": pending_papers}), 202

        combined_text = "\n\n".join(f"--- START PAPER: {fname} ---\n{pdata['full_text']}\n--- END PAPER: {fname} ---" for fname, pdata in processed_papers_map.items())
        relevant_questions_map = identify_relevant_questions_in_bulk_ai(combined_text, topic, subject_code, level)

        if not relevant_questions_map: return jsonify({"success": False, "message": f"No questions found for the topic '{topic}' in the available papers."}), 404

        final_questions_for_pdf = []
        unique_question_hashes = set()
        for filename, q_numbers in relevant_questions_map.items():
            if filename not in processed_papers_map: continue
            paper_data = processed_papers_map[filename]
            question_map = {q['question_number']: q for q in parse_all_questions_from_text(paper_data['full_text'])}
            for q_num in q_numbers:
                if q_data := question_map.get(str(q_num).strip()):
                    question_hash = hashlib.md5(re.sub(r'\s+', '', q_data['text']).lower().encode()).hexdigest()
                    if question_hash not in unique_question_hashes:
                        final_questions_for_pdf.append({**q_data, "source_filename": filename, "page_images": paper_data['page_images']})
                        unique_question_hashes.add(question_hash)

        if not final_questions_for_pdf: return jsonify({"success": False, "message": f"No questions found for the topic '{topic}' after filtering."}), 404

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp: output_pdf_path = tmp.name
        _build_extracted_pdf(output_pdf_path, topic, final_questions_for_pdf)
        
        response = send_file(output_pdf_path, as_attachment=True, download_name=f"extracted_{topic.replace(' ', '_')}.pdf", mimetype='application/pdf')
        if pending_papers:
            response.headers['X-Processing-Message'] = f"Extracted questions from {len(processed_papers_map)} papers. An additional {len(pending_papers)} papers are being processed in the background for future requests."

        return response
    except Exception as e:
        logging.error(f"Unhandled error in generate_extracted_paper: {e}\n{traceback.format_exc()}")
        return jsonify({"success": False, "message": "An unexpected server error occurred."}), 500

# ==============================================================================
# --- 8. MAIN EXECUTION ---
# ==============================================================================
if __name__ == '__main__':
    setup_cache()
    if not _api_keys or _api_keys[0] in _placeholder_keys:
        logging.fatal("AI features disabled due to missing API key.")
    if not Config.TESSACT_CMD:
        logging.fatal("OCR features disabled because Tesseract is not in PATH.")
    
    from waitress import serve
    logging.info("Starting production server with Waitress on http://0.0.0.0:10000")
    serve(app, host="0.0.0.0", port=10000)