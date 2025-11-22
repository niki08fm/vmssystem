
# python_service/app.py
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow React (or any frontend) to call this directly

# --- CONFIG ---
# UPDATE THESE PATHS TO YOUR ACTUAL PATHS
YOLO_PATH = r'models\yolov8n.pt'
PLATE_MODEL_PATH = r'models\license_plate_detector.pt'

# --- GLOBAL INIT ---
print("Loading models... (this may take a while)")
reader = easyocr.Reader(['en'], gpu=False)
coco_model = YOLO(YOLO_PATH)
license_plate_detector = YOLO(PLATE_MODEL_PATH)
print("Models loaded.")

# --- MAPPING HELPER FUNCTIONS ---
dict_char_to_int = {
    'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6',
    'S': '5', 'Z': '2', 'B': '8', 'T': '1', 'L': '1',
    'Q': '0', 'D': '0'
}
dict_int_to_char = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'J',
    '4': 'A', '6': 'G', '5': 'S', '8': 'B', '7': 'T'
}


def preprocess_plate(img):
    """Enhance the cropped plate for OCR"""
    if img is None or img.size == 0:
        return img
    scaling_factor = 4
    new_width = max(1, int(img.shape[1] * scaling_factor))
    new_height = max(1, int(img.shape[0] * scaling_factor))
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    return sharpened


def fix_indian_plate(text):
    """
    Try to normalise OCR text to an Indian-like 10-char plate.
    This uses simple heuristics (mapping ambiguous chars) from your previous code.
    Returns normalized plate string or None if not confident.
    """
    if not text:
        return None
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    if clean.startswith("IND"):
        clean = clean[3:]
    target_len = 10
    if len(clean) < target_len:
        return None

    best_candidate = ""
    best_score = -1

    for i in range(len(clean) - target_len + 1):
        candidate = clean[i: i + target_len]
        simulated_score = 0
        # scoring heuristics (higher = better)
        if candidate[0] not in dict_char_to_int:
            simulated_score += 1
        if candidate[1] not in dict_char_to_int:
            simulated_score += 1
        if candidate[2].isdigit() or candidate[2] in dict_char_to_int:
            simulated_score += 1
        if candidate[3].isdigit() or candidate[3] in dict_char_to_int:
            simulated_score += 1
        if candidate[4] not in dict_char_to_int:
            simulated_score += 1
        if candidate[5] not in dict_char_to_int:
            simulated_score += 1
        if candidate[6].isdigit() or candidate[6] in dict_char_to_int:
            simulated_score += 1
        if candidate[7].isdigit() or candidate[7] in dict_char_to_int:
            simulated_score += 1
        if candidate[8].isdigit() or candidate[8] in dict_char_to_int:
            simulated_score += 1
        if candidate[9].isdigit() or candidate[9] in dict_char_to_int:
            simulated_score += 1

        if simulated_score > best_score:
            best_score = simulated_score
            best_candidate = candidate

    chars = list(best_candidate)
    # map integers to chars for letter positions (approx)
    for i in [0, 1, 4, 5]:
        if chars[i] in dict_int_to_char:
            chars[i] = dict_int_to_char[chars[i]]
    # map chars to ints for digit positions (approx)
    for i in [2, 3, 6, 7, 8, 9]:
        if chars[i] in dict_char_to_int:
            chars[i] = dict_char_to_int[chars[i]]

    return "".join(chars)


@app.route('/detect', methods=['POST'])
def detect():
    """Main detection endpoint - accepts multipart/form-data with 'image' file"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Read file into OpenCV image
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not decode image'}), 400

    # Vehicle class IDs used by COCO (as in your original): car, motorcycle, bus, truck
    vehicles = [2, 3, 5, 7]
    try:
        vehicle_detections = coco_model(img)[0]
    except Exception as e:
        return jsonify({'error': f'coco_model inference failed: {str(e)}'}), 500

    detected_data = {
        "found": False,
        "text": "",
        "cropped_image": ""
    }

    # iterate over detections
    boxes = []
    try:
        boxes = vehicle_detections.boxes.data.tolist()
    except Exception:
        boxes = []

    for detection in boxes:
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            # safely crop (clamp coordinates)
            h, w = img.shape[:2]
            ix1, iy1 = max(0, int(x1)), max(0, int(y1))
            ix2, iy2 = min(w, int(x2)), min(h, int(y2))
            vehicle_crop = img[iy1:iy2, ix1:ix2]
            if vehicle_crop.size == 0:
                continue

            # detect plates inside vehicle crop
            try:
                plate_detections = license_plate_detector(vehicle_crop)[0]
            except Exception:
                continue

            plates_data = []
            try:
                plates_data = plate_detections.boxes.data.tolist()
            except Exception:
                plates_data = []

            if not plates_data:
                continue

            # choose the highest-scoring plate
            best_plate = max(plates_data, key=lambda x: x[4])
            px1, py1, px2, py2, p_score, p_class_id = best_plate

            # convert plate coords relative to original image
            real_px1, real_py1 = ix1 + int(px1), iy1 + int(py1)
            real_px2, real_py2 = ix1 + int(px2), iy1 + int(py2)

            # clamp
            h, w = img.shape[:2]
            real_px1, real_py1 = max(0, real_px1), max(0, real_py1)
            real_px2, real_py2 = min(w, real_px2), min(h, real_py2)

            plate_crop = img[real_py1:real_py2, real_px1:real_px2]
            if plate_crop.size == 0:
                continue

            processed_plate = preprocess_plate(plate_crop)
            try:
                ocr_result = reader.readtext(processed_plate, detail=1)
            except Exception:
                ocr_result = []

            # sort by left-most x coordinate
            try:
                ocr_result.sort(key=lambda x: x[0][0][0])
            except Exception:
                pass

            raw_text_list = [text for (bbox, text, prob) in ocr_result if prob > 0.25]
            final_text = fix_indian_plate("".join(raw_text_list))

            if final_text:
                _, buffer = cv2.imencode('.jpg', plate_crop)
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')

                detected_data = {
                    "found": True,
                    "text": final_text,
                    "cropped_image": "data:image/jpeg;base64," + jpg_as_text
                }
                return jsonify(detected_data)

    return jsonify(detected_data)


@app.route('/form', methods=['GET'])
def form():
    """Simple HTML form to upload image and display detect() JSON result"""
    return """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>ANPR Test Form</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 30px; }
        .preview { max-width: 400px; margin-top: 10px; }
        .result { margin-top: 20px; padding: 12px; border: 1px solid #ddd; border-radius: 6px; background: #fafafa; }
        label { display:block; margin-bottom:8px; font-weight:600; }
        button { padding: 8px 12px; font-size: 14px; cursor: pointer; }
      </style>
    </head>
    <body>
      <h2>ANPR Test Form</h2>
      <p>Select an image containing vehicles and plates, then click <strong>Detect</strong>.</p>

      <form id="uploadForm">
        <label for="image">Choose image</label>
        <input id="image" name="image" type="file" accept="image/*" required />
        <div>
          <img id="imgPreview" class="preview" src="" alt="" style="display:none;" />
        </div>
        <br/>
        <button type="submit">Detect</button>
      </form>

      <div id="result" class="result" style="display:none;">
        <h3>Result</h3>
        <div id="foundText"></div>
        <div id="croppedImgWrap" style="margin-top:10px;"></div>
        <pre id="rawJson" style="white-space:pre-wrap; margin-top:12px;"></pre>
      </div>

      <script>
        const imageInput = document.getElementById('image');
        const imgPreview = document.getElementById('imgPreview');
        const uploadForm = document.getElementById('uploadForm');
        const resultDiv = document.getElementById('result');
        const foundText = document.getElementById('foundText');
        const croppedImgWrap = document.getElementById('croppedImgWrap');
        const rawJson = document.getElementById('rawJson');

        imageInput.addEventListener('change', (e) => {
          const file = e.target.files[0];
          if (!file) { imgPreview.style.display = 'none'; return; }
          const url = URL.createObjectURL(file);
          imgPreview.src = url;
          imgPreview.style.display = 'block';
        });

        uploadForm.addEventListener('submit', async (e) => {
          e.preventDefault();
          const file = imageInput.files[0];
          if (!file) return alert('Choose an image first.');

          const fd = new FormData();
          fd.append('image', file);

          // show loading state
          foundText.textContent = 'Detecting...';
          croppedImgWrap.innerHTML = '';
          rawJson.textContent = '';
          resultDiv.style.display = 'block';

          try {
            const resp = await fetch('/detect', { method: 'POST', body: fd });
            if (!resp.ok) {
              const text = await resp.text();
              foundText.textContent = 'Server error: ' + resp.status;
              rawJson.textContent = text;
              return;
            }
            const data = await resp.json();
            rawJson.textContent = JSON.stringify(data, null, 2);

            if (data.found) {
              foundText.innerHTML = '<strong>Plate:</strong> ' + data.text;
              if (data.cropped_image) {
                croppedImgWrap.innerHTML = '<h4>Cropped plate</h4><img src="' + data.cropped_image + '" style="max-width:300px;" />';
              }
            } else {
              foundText.innerHTML = '<strong>No plate found</strong>';
            }
          } catch (err) {
            foundText.textContent = 'Error: ' + err.message;
            rawJson.textContent = String(err);
          }
        });
      </script>
    </body>
    </html>
    """


if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=True)
