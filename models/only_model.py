# # # import cv2
# # # import numpy as np
# # # from ultralytics import YOLO
# # # import easyocr

# # # # 1. Initialize OCR reader
# # # # Set gpu=True if you have a customized CUDA setup, otherwise False is safer
# # # reader = easyocr.Reader(['en'], gpu=False)

# # # def preprocess_plate(img):
# # #     """
# # #     Preprocesses the license plate image to improve OCR accuracy.
# # #     1. Grayscale
# # #     2. Upscale (makes text larger/clearer)
# # #     3. Adaptive Threshold (handles shadows/glare)
# # #     """
# # #     # Convert to grayscale
# # #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
# # #     # Resize: Upscale by 3x to make characters distinct
# # #     scaling_factor = 3
# # #     new_width = int(gray.shape[1] * scaling_factor)
# # #     new_height = int(gray.shape[0] * scaling_factor)
# # #     resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
# # #     # Apply Gaussian Blur to reduce noise
# # #     blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    
# # #     # Adaptive Thresholding: Converts to strictly black and white based on local lighting
# # #     thresholded = cv2.adaptiveThreshold(
# # #         blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9
# # #     )
    
# # #     return thresholded

# # # def main():
# # #     # 2. Load Models
# # #     print("Loading models...")
# # #     coco_model = YOLO(r'D:\projects\freelancer\gms\automatic-number-plate-recognition-python-yolov8\models\yolov8n.pt')  # Vehicle detector
# # #     license_plate_detector = YOLO(r'D:\projects\freelancer\gms\automatic-number-plate-recognition-python-yolov8\models\license_plate_detector.pt') # Plate detector

# # #     # 3. Load Image
# # #     image_path = 'test1.jpg'  # <--- CHECK YOUR FILENAME
# # #     img = cv2.imread(image_path)

# # #     if img is None:
# # #         print(f"Error: Could not load image {image_path}")
# # #         return

# # #     # Vehicle class IDs in COCO dataset: car(2), motorcycle(3), bus(5), truck(7)
# # #     vehicles = [2, 3, 5, 7]

# # #     print("Detecting vehicles...")
# # #     vehicle_detections = coco_model(img)[0]
    
# # #     detected_any = False

# # #     for detection in vehicle_detections.boxes.data.tolist():
# # #         x1, y1, x2, y2, score, class_id = detection

# # #         if int(class_id) in vehicles:
# # #             # Crop the vehicle area
# # #             vehicle_crop = img[int(y1):int(y2), int(x1):int(x2)]
            
# # #             if vehicle_crop.size == 0:
# # #                 continue

# # #             # 4. Detect License Plates INSIDE the vehicle crop
# # #             plate_detections = license_plate_detector(vehicle_crop)[0]
            
# # #             # Get list of all plate detections
# # #             plates_data = plate_detections.boxes.data.tolist()

# # #             if not plates_data:
# # #                 continue

# # #             # --- FIX: Select only the BEST plate for this vehicle ---
# # #             # We choose the plate detection with the highest confidence score (index 4)
# # #             # This prevents multiple overlapping boxes from being processed
# # #             best_plate = max(plates_data, key=lambda x: x[4])

# # #             # Extract data from the best plate only
# # #             px1, py1, px2, py2, p_score, p_class_id = best_plate

# # #             # Coordinate transformation:
# # #             real_px1 = int(x1) + int(px1)
# # #             real_py1 = int(y1) + int(py1)
# # #             real_px2 = int(x1) + int(px2)
# # #             real_py2 = int(y1) + int(py2)

# # #             # Crop the license plate
# # #             plate_crop = img[real_py1:real_py2, real_px1:real_px2]
            
# # #             if plate_crop.size == 0:
# # #                 continue

# # #             # 5. Preprocess & Read Text
# # #             processed_plate = preprocess_plate(plate_crop)
            
# # #             ocr_result = reader.readtext(processed_plate, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# # #             if ocr_result:
# # #                 detected_any = True
# # #                 text = "".join(ocr_result).replace(" ", "")
# # #                 print(f"Found Plate: {text} (Confidence: {p_score:.2f})")

# # #                 # --- VISUALIZATION ---
# # #                 cv2.rectangle(img, (real_px1, real_py1), (real_px2, real_py2), (0, 0, 255), 3)
                
# # #                 (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
# # #                 cv2.rectangle(img, (real_px1, real_py1 - 35), (real_px1 + text_w, real_py1), (0, 0, 255), -1)
                
# # #                 cv2.putText(img, text, (real_px1, real_py1 - 10), 
# # #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
# # #                 # Optional: Show the cropped plate
# # #                 # cv2.imshow(f"Processed Plate", processed_plate)

# # #     if not detected_any:
# # #         print("No license plates detected or read.")

# # #     # 6. Save and Show Output
# # #     cv2.imwrite('output_result.jpg', img)
# # #     print("Result saved to output_result.jpg")
    
# # #     cv2.imshow('Final Result', img)
# # #     cv2.waitKey(0)
# # #     cv2.destroyAllWindows()

# # # if __name__ == "__main__":
# # #     main()


# # import cv2
# # import numpy as np
# # from ultralytics import YOLO
# # import easyocr
# # import re

# # # 1. Initialize OCR reader
# # print("Initializing OCR...")
# # reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have CUDA

# # def preprocess_plate(img):
# #     """
# #     Improved preprocessing for OCR.
# #     Instead of binary thresholding (which creates noise), we use
# #     Contrast Limited Adaptive Histogram Equalization (CLAHE) and Denoising.
# #     """
# #     # 1. Resize: Upscale by 4x makes characters distinct for OCR
# #     scaling_factor = 4
# #     new_width = int(img.shape[1] * scaling_factor)
# #     new_height = int(img.shape[0] * scaling_factor)
# #     resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
# #     # 2. Grayscale
# #     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
# #     # 3. Denoise (Bilateral Filter keeps edges sharp but removes grain)
# #     denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    
# #     # 4. Contrast Boosting (CLAHE)
# #     # This is better than standard thresholding for plates with glare/shadows
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# #     enhanced = clahe.apply(denoised)
    
# #     # Optional: Mild sharpening to make edges crisp
# #     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# #     sharpened = cv2.filter2D(enhanced, -1, kernel)

# #     return sharpened

# # def clean_text(text):
# #     """
# #     Cleans OCR result:
# #     - Removes special characters
# #     - Keeps only Alphanumeric (A-Z, 0-9)
# #     - Capitalizes everything
# #     """
# #     # Keep only alphanumeric
# #     cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
# #     return cleaned

# # def main():
# #     # 2. Load Models
# #     # UPDATE THESE PATHS TO YOUR LOCAL FILES
# #     yolo_path = r'D:\projects\freelancer\gms\automatic-number-plate-recognition-python-yolov8\models\yolov8n.pt'
# #     plate_model_path = r'D:\projects\freelancer\gms\automatic-number-plate-recognition-python-yolov8\models\license_plate_detector.pt'
    
# #     print("Loading YOLO models...")
# #     try:
# #         coco_model = YOLO(yolo_path)
# #         license_plate_detector = YOLO(plate_model_path)
# #     except Exception as e:
# #         print(f"Error loading models. Check paths.\n{e}")
# #         return

# #     # 3. Load Image
# #     image_path = 'image.png' 
# #     img = cv2.imread(image_path)

# #     if img is None:
# #         print(f"Error: Could not load image {image_path}")
# #         return

# #     vehicles = [2, 3, 5, 7] # car, motorcycle, bus, truck

# #     print("Processing image...")
    
# #     # Detect Vehicles
# #     vehicle_detections = coco_model(img)[0]
# #     detected_any = False

# #     for detection in vehicle_detections.boxes.data.tolist():
# #         x1, y1, x2, y2, score, class_id = detection

# #         if int(class_id) in vehicles:
# #             vehicle_crop = img[int(y1):int(y2), int(x1):int(x2)]
            
# #             if vehicle_crop.size == 0: continue

# #             # Detect Plates
# #             plate_detections = license_plate_detector(vehicle_crop)[0]
# #             plates_data = plate_detections.boxes.data.tolist()

# #             if not plates_data: continue

# #             # Select BEST plate (highest confidence)
# #             best_plate = max(plates_data, key=lambda x: x[4])
# #             px1, py1, px2, py2, p_score, p_class_id = best_plate

# #             # Calculate real coordinates
# #             real_px1 = int(x1) + int(px1)
# #             real_py1 = int(y1) + int(py1)
# #             real_px2 = int(x1) + int(px2)
# #             real_py2 = int(y1) + int(py2)

# #             # Crop Plate
# #             plate_crop = img[real_py1:real_py2, real_px1:real_px2]
            
# #             if plate_crop.size == 0: continue

# #             # --- IMPROVED OCR PIPELINE ---
# #             processed_plate = preprocess_plate(plate_crop)
            
# #             # debug: save processed plate to see what OCR sees
# #             cv2.imwrite("debug_processed_plate.jpg", processed_plate) 

# #             # Read text (detail=1 gives us position and confidence)
# #             ocr_result = reader.readtext(processed_plate, detail=1)

# #             # Sort results left-to-right (handles split text better)
# #             ocr_result.sort(key=lambda x: x[0][0][0])

# #             full_text_parts = []
# #             for (bbox, text, prob) in ocr_result:
# #                 # Filter out low confidence garbage and very short noise
# #                 if prob > 0.3: 
# #                     cleaned_part = clean_text(text)
# #                     if len(cleaned_part) > 1: # Ignore single noise chars
# #                         full_text_parts.append(cleaned_part)

# #             # Join all parts found (e.g., "IND" + "MH20" + "DV2366")
# #             # Then find the longest continuous string which is usually the plate number
# #             full_text = "".join(full_text_parts)
            
# #             # Logic: If multiple parts, sometimes "IND" is merged. 
# #             # Usually the plate is the longest chunk. 
# #             # For now, we join them. You can refine logic to strip "IND" if needed.
# #             final_text = full_text

# #             if len(final_text) > 4: # A valid plate is usually > 4 chars
# #                 detected_any = True
# #                 print(f" -> Plate Detected: {final_text} (Raw Conf: {p_score:.2f})")

# #                 # Draw Box & Text
# #                 cv2.rectangle(img, (real_px1, real_py1), (real_px2, real_py2), (0, 255, 0), 3)
                
# #                 # Background for text for better visibility
# #                 (text_w, text_h), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
# #                 cv2.rectangle(img, (real_px1, real_py1 - 40), (real_px1 + text_w + 10, real_py1), (0, 255, 0), -1)
                
# #                 cv2.putText(img, final_text, (real_px1 + 5, real_py1 - 10), 
# #                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# #     if not detected_any:
# #         print("No readable plates found.")

# #     cv2.imwrite('output_final.jpg', img)
# #     print("Saved result to output_final.jpg")
    
# #     cv2.imshow('Final Result', img)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     main()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# import easyocr
# import re

# # 1. Initialize OCR
# print("Initializing OCR...")
# reader = easyocr.Reader(['en'], gpu=False)

# # --- MAPPING DICTIONARIES FOR ERROR CORRECTION ---
# # Maps letters that look like numbers TO numbers
# dict_char_to_int = {
#     'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'Z': '2', 
#     'B': '8', 'Q': '0', 'D': '0', 'L': '1'
# }

# # Maps numbers that look like letters TO letters
# dict_int_to_char = {
#     '0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '2': 'Z', '8': 'B'
# }

# def preprocess_plate(img):
#     """
#     Advanced preprocessing: CLAHE + Bilateral Filter + Sharpening
#     """
#     scaling_factor = 4
#     new_width = int(img.shape[1] * scaling_factor)
#     new_height = int(img.shape[0] * scaling_factor)
#     resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     enhanced = clahe.apply(denoised)
    
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     sharpened = cv2.filter2D(enhanced, -1, kernel)
    
#     return sharpened

# def clean_text(text):
#     # Keep alphanumeric only and upper case
#     return re.sub(r'[^A-Z0-9]', '', text.upper())

# def fix_characters(text):
#     """
#     Logic to fix common OCR errors based on Indian Plate Format:
#     Format: SS DD LL NNNN (e.g., CH 01 AN 0001)
    
#     Rules applied:
#     1. First 2 chars are usually State Code (Must be LETTERS)
#     2. Next 2 chars are District Code (Must be NUMBERS)
#     3. The LAST 4 chars are the Unique ID (Must be NUMBERS)
#     """
#     chars = list(text)
#     length = len(chars)
    
#     # Rule 1: The LAST 4 characters must be numbers (handles your 'OOOI' -> '0001' issue)
#     if length >= 4:
#         for i in range(length - 4, length):
#             if chars[i] in dict_char_to_int:
#                 chars[i] = dict_char_to_int[chars[i]]

#     # Rule 2: The FIRST 2 characters must be letters (State Code, e.g., MH, DL, CH)
#     if length >= 2:
#         for i in range(0, 2):
#             if chars[i] in dict_int_to_char:
#                 chars[i] = dict_int_to_char[chars[i]]

#     # Rule 3: Characters 3 and 4 (indices 2,3) are usually numbers (District Code)
#     # Only apply if length is sufficient (standard plates are usually 9-10 chars)
#     if length >= 4:
#         for i in range(2, 4):
#             if chars[i] in dict_char_to_int:
#                 chars[i] = dict_char_to_int[chars[i]]
                
#     return "".join(chars)

# def main():
#     # UPDATE PATHS
#     yolo_path = r'D:\projects\freelancer\gms\automatic-number-plate-recognition-python-yolov8\models\yolov8n.pt'
#     plate_model_path = r'D:\projects\freelancer\gms\automatic-number-plate-recognition-python-yolov8\models\license_plate_detector.pt'
    
#     print("Loading models...")
#     try:
#         coco_model = YOLO(yolo_path)
#         license_plate_detector = YOLO(plate_model_path)
#     except Exception as e:
#         print(f"Error: {e}")
#         return

#     # Load Image
#     image_path = 'test1.jpg'
#     img = cv2.imread(image_path)

#     if img is None:
#         print("Image not found.")
#         return

#     vehicles = [2, 3, 5, 7] # Classes for vehicles

#     print("Scanning...")
#     vehicle_detections = coco_model(img)[0]
#     detected_any = False

#     for detection in vehicle_detections.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = detection

#         if int(class_id) in vehicles:
#             vehicle_crop = img[int(y1):int(y2), int(x1):int(x2)]
#             if vehicle_crop.size == 0: continue

#             plate_detections = license_plate_detector(vehicle_crop)[0]
#             plates_data = plate_detections.boxes.data.tolist()

#             if not plates_data: continue

#             # Get best plate
#             best_plate = max(plates_data, key=lambda x: x[4])
#             px1, py1, px2, py2, p_score, p_class_id = best_plate

#             # Real coords
#             real_px1 = int(x1) + int(px1)
#             real_py1 = int(y1) + int(py1)
#             real_px2 = int(x1) + int(px2)
#             real_py2 = int(y1) + int(py2)

#             plate_crop = img[real_py1:real_py2, real_px1:real_px2]
#             if plate_crop.size == 0: continue

#             # 1. Preprocess
#             processed_plate = preprocess_plate(plate_crop)

#             # 2. OCR
#             ocr_result = reader.readtext(processed_plate, detail=1)
#             ocr_result.sort(key=lambda x: x[0][0][0]) # Sort left to right

#             full_text_parts = []
#             for (bbox, text, prob) in ocr_result:
#                 if prob > 0.2: # Lowered slightly to catch confused chars
#                     cleaned = clean_text(text)
#                     if len(cleaned) > 0:
#                         full_text_parts.append(cleaned)

#             raw_text = "".join(full_text_parts)
            
#             # 3. APPLY THE LOGIC FIX
#             final_text = fix_characters(raw_text)

#             if len(final_text) > 4:
#                 detected_any = True
#                 print(f" -> Original: {raw_text}")
#                 print(f" -> Corrected: {final_text} (Conf: {p_score:.2f})")

#                 # Visuals
#                 cv2.rectangle(img, (real_px1, real_py1), (real_px2, real_py2), (0, 255, 0), 3)
#                 (text_w, text_h), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#                 cv2.rectangle(img, (real_px1, real_py1 - 40), (real_px1 + text_w + 10, real_py1), (0, 255, 0), -1)
#                 cv2.putText(img, final_text, (real_px1 + 5, real_py1 - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

#     if not detected_any:
#         print("No plates detected.")

#     cv2.imwrite('output_corrected.jpg', img)
#     cv2.imshow('Result', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re

# 1. Initialize OCR
print("Initializing OCR...")
reader = easyocr.Reader(['en'], gpu=False) 

# --- MAPPING DICTIONARIES ---
# Maps Letters -> Numbers (for positions 2,3 and 6,7,8,9)
dict_char_to_int = {
    'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'Z': '2', 
    'B': '8', 'T': '1', 'L': '1', 'Q': '0', 'D': '0'
}

# Maps Numbers -> Letters (for positions 0,1 and 4,5)
dict_int_to_char = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A', 
    '6': 'G', '5': 'S', '8': 'B', '7': 'T'
}

def preprocess_plate(img):
    """
    Standard robust preprocessing.
    """
    scaling_factor = 4
    new_width = int(img.shape[1] * scaling_factor)
    new_height = int(img.shape[0] * scaling_factor)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 11, 17, 17)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

def score_plate_pattern(text):
    """
    Scores a 10-char string on how well it fits AA 11 BB 1111
    Higher score = More likely to be the valid plate.
    """
    score = 0
    # Check State Code (First 2 chars should be Letters)
    if text[0].isalpha(): score += 1
    if text[1].isalpha(): score += 1
    
    # Check RTO Code (Next 2 chars should be Numbers)
    if text[2].isdigit(): score += 1
    if text[3].isdigit(): score += 1
    
    # Check Series (Next 2 chars should be Letters)
    if text[4].isalpha(): score += 1
    if text[5].isalpha(): score += 1
    
    # Check Unique ID (Last 4 chars should be Numbers)
    if text[6].isdigit(): score += 1
    if text[7].isdigit(): score += 1
    if text[8].isdigit(): score += 1
    if text[9].isdigit(): score += 1
    
    return score

def fix_indian_plate(text):
    """
    Extracts the most likely 10-character Indian Plate from a noisy string.
    Handles: MHZODK23661 -> MH20DK2366
    """
    # 1. Basic Cleanup (AlphaNumeric Only)
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Remove specific "IND" noise common at the start
    if clean.startswith("IND"): clean = clean[3:]
    
    # 2. SLIDING WINDOW SELECTION
    # If length > 10, we try every 10-char substring and see which fits the format best.
    target_len = 10
    
    if len(clean) < target_len:
        return None # Too short to be a full plate
        
    best_candidate = ""
    best_score = -1
    
    # If exactly 10, loops once. If 11, loops twice (index 0 and 1).
    for i in range(len(clean) - target_len + 1):
        candidate = clean[i : i + target_len]
        
        # We must "simulate" the correction to score it fairly
        # (e.g., Z in pos 2 is actually a Number 2, so we count it as a match)
        simulated_score = 0
        
        # Check AA
        if candidate[0] not in dict_char_to_int: simulated_score += 1
        if candidate[1] not in dict_char_to_int: simulated_score += 1
        
        # Check 11 (Allow letters that map to numbers)
        if candidate[2].isdigit() or candidate[2] in dict_char_to_int: simulated_score += 1
        if candidate[3].isdigit() or candidate[3] in dict_char_to_int: simulated_score += 1
        
        # Check BB
        if candidate[4] not in dict_char_to_int: simulated_score += 1
        if candidate[5] not in dict_char_to_int: simulated_score += 1
        
        # Check 1111
        if candidate[6].isdigit() or candidate[6] in dict_char_to_int: simulated_score += 1
        if candidate[7].isdigit() or candidate[7] in dict_char_to_int: simulated_score += 1
        if candidate[8].isdigit() or candidate[8] in dict_char_to_int: simulated_score += 1
        if candidate[9].isdigit() or candidate[9] in dict_char_to_int: simulated_score += 1
        
        if simulated_score > best_score:
            best_score = simulated_score
            best_candidate = candidate
            
    # 3. APPLY CORRECTION TO THE WINNER
    chars = list(best_candidate)
    
    # State (0,1) -> Force Letters
    for i in [0, 1]:
        if chars[i] in dict_int_to_char: chars[i] = dict_int_to_char[chars[i]]
            
    # RTO (2,3) -> Force Numbers
    for i in [2, 3]:
        if chars[i] in dict_char_to_int: chars[i] = dict_char_to_int[chars[i]]
            
    # Series (4,5) -> Force Letters
    for i in [4, 5]:
        if chars[i] in dict_int_to_char: chars[i] = dict_int_to_char[chars[i]]
            
    # ID (6,7,8,9) -> Force Numbers
    for i in range(6, 10):
        if chars[i] in dict_char_to_int: chars[i] = dict_char_to_int[chars[i]]
            
    return "".join(chars)

def main():
    # CONFIG
    yolo_path = r'C:\Users\mekap\OneDrive\Desktop\brahma\GMS_client-52-\models\yolov8n.pt'
    plate_model_path = r'C:\Users\mekap\OneDrive\Desktop\brahma\GMS_client-52-\models\license_plate_detector.pt'
    image_path = 'image.png' 
    
    print("Loading models...")
    try:
        coco_model = YOLO(yolo_path)
        license_plate_detector = YOLO(plate_model_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    img = cv2.imread(image_path)
    if img is None: return

    vehicles = [2, 3, 5, 7] 

    print("Detecting...")
    vehicle_detections = coco_model(img)[0]
    detected_any = False

    for detection in vehicle_detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection

        if int(class_id) in vehicles:
            vehicle_crop = img[int(y1):int(y2), int(x1):int(x2)]
            if vehicle_crop.size == 0: continue

            plate_detections = license_plate_detector(vehicle_crop)[0]
            plates_data = plate_detections.boxes.data.tolist()
            if not plates_data: continue

            best_plate = max(plates_data, key=lambda x: x[4])
            px1, py1, px2, py2, p_score, p_class_id = best_plate

            real_px1, real_py1 = int(x1) + int(px1), int(y1) + int(py1)
            real_px2, real_py2 = int(x1) + int(px2), int(y1) + int(py2)

            plate_crop = img[real_py1:real_py2, real_px1:real_px2]
            if plate_crop.size == 0: continue

            # 1. Preprocess
            processed_plate = preprocess_plate(plate_crop)

            # 2. OCR
            ocr_result = reader.readtext(processed_plate, detail=1)
            ocr_result.sort(key=lambda x: x[0][0][0])

            raw_text_list = []
            for (bbox, text, prob) in ocr_result:
                if prob > 0.25: raw_text_list.append(text)
            
            raw_text = "".join(raw_text_list)

            # 3. INTELLIGENT FIX
            final_text = fix_indian_plate(raw_text)

            if final_text:
                detected_any = True
                print(f" -> RAW OCR: {raw_text}")
                print(f" -> FINAL  : {final_text}")

                # Draw
                cv2.rectangle(img, (real_px1, real_py1), (real_px2, real_py2), (0, 255, 0), 3)
                (text_w, text_h), _ = cv2.getTextSize(final_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(img, (real_px1, real_py1 - 40), (real_px1 + text_w + 10, real_py1), (0, 255, 0), -1)
                cv2.putText(img, final_text, (real_px1 + 5, real_py1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    if not detected_any:
        print("No valid plates detected.")

    cv2.imwrite('output_final_reliable.jpg', img)
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()