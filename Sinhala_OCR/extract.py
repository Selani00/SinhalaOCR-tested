import sys
from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor

def main():
    image_path = "sinhala_text.jpg"
    print(f"Loading image from {image_path}...")
    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    print("Initializing DetectionPredictor and RecognitionPredictor...")
    # These classes load the models onto GPU (CUDA) automatically if available.
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()

    print("Running OCR (Detection + Recognition) on the image...")
    # Predictions will be a list of OCRResult (one per input image)
    predictions = rec_predictor([image], det_predictor=det_predictor)

    print("\n--- Extracted Text ---")
    if not predictions:
        print("No text detected.")
        return

    for page_idx, pred in enumerate(predictions):
        print(f"Page {page_idx + 1}:")
        for line_idx, line in enumerate(pred.text_lines):
            print(f"Line {line_idx + 1} (Confidence: {line.confidence:.2f}): {line.text}")

if __name__ == "__main__":
    main()
