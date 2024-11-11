import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import argparse
from pathlib import Path
import hashlib
from PIL import ImageEnhance, ImageFilter
import itertools
import time

class OCROptimizer:
    """Handles automatic parameter tuning for OCR optimization."""
    
    def __init__(self):
        # Define different optimization strategies
        self.strategies = {
            'default': {
                'resize_factor': 2.0,
                'sharpen': 1.5,
                'contrast': 1.2,
                'brightness': 1.0,
                'denoise': True,
                'threshold': False,
                'psm': 3,
                'oem': 3,
                'dpi': 300,
                'confidence': 60,
            },
            'high_quality': {
                'resize_factor': 3.0,
                'sharpen': 2.0,
                'contrast': 1.5,
                'brightness': 1.2,
                'denoise': True,
                'threshold': False,
                'psm': 3,
                'dpi': 400,
                'confidence': 70,
            },
            'noisy_document': {
                'resize_factor': 2.0,
                'sharpen': 1.2,
                'contrast': 1.3,
                'brightness': 1.1,
                'denoise': True,
                'threshold': True,
                'psm': 6,
                'dpi': 300,
                'confidence': 50,
            },
            'low_quality': {
                'resize_factor': 4.0,
                'sharpen': 2.5,
                'contrast': 1.8,
                'brightness': 1.3,
                'denoise': True,
                'threshold': True,
                'psm': 6,
                'dpi': 400,
                'confidence': 40,
            },
            'single_column': {
                'resize_factor': 2.0,
                'sharpen': 1.5,
                'contrast': 1.2,
                'brightness': 1.0,
                'denoise': False,
                'threshold': False,
                'psm': 4,
                'dpi': 300,
                'confidence': 60,
            },
            'single_line': {
                'resize_factor': 2.5,
                'sharpen': 1.8,
                'contrast': 1.4,
                'brightness': 1.1,
                'denoise': True,
                'threshold': False,
                'psm': 7,
                'dpi': 350,
                'confidence': 50,
            }
        }
        
        # Define PSM modes to try
        self.psm_modes = [3, 4, 6, 7, 13]
        
        # Define preprocessing combinations
        self.preprocessing_combinations = [
            {'resize_factor': 2.0, 'sharpen': 1.5, 'contrast': 1.2},
            {'resize_factor': 3.0, 'sharpen': 2.0, 'contrast': 1.5},
            {'resize_factor': 4.0, 'sharpen': 2.5, 'contrast': 1.8},
        ]
    
    def get_next_settings(self, attempt):
        """Get the next set of settings based on the attempt number."""
        # First try the predefined strategies
        if attempt < len(self.strategies):
            strategy_name = list(self.strategies.keys())[attempt]
            print(f"\nTrying {strategy_name} strategy...")
            return self.strategies[strategy_name]
        
        # Then try combinations of PSM modes and preprocessing settings
        total_strategies = len(self.strategies)
        combination_index = attempt - total_strategies
        
        if combination_index < len(self.psm_modes) * len(self.preprocessing_combinations):
            psm_index = (combination_index // len(self.preprocessing_combinations)) % len(self.psm_modes)
            prep_index = combination_index % len(self.preprocessing_combinations)
            
            settings = self.strategies['default'].copy()
            settings.update(self.preprocessing_combinations[prep_index])
            settings['psm'] = self.psm_modes[psm_index]
            
            print(f"\nTrying custom combination - PSM: {settings['psm']}, "
                  f"Resize: {settings['resize_factor']}, "
                  f"Sharpen: {settings['sharpen']}, "
                  f"Contrast: {settings['contrast']}")
            
            return settings
        
        # Finally, return None to indicate no more combinations to try
        return None

class ImagePreprocessor:
    """Handle image preprocessing with various enhancement options."""
    @staticmethod
    def enhance_image(image, settings):
        """Apply various image enhancements based on settings."""
        if settings.get('resize_factor', 1) != 1:
            new_size = tuple(int(dim * settings['resize_factor']) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        if settings.get('sharpen', 0) > 0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(settings['sharpen'])
        
        if settings.get('contrast', 1) != 1:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(settings['contrast'])
        
        if settings.get('brightness', 1) != 1:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(settings['brightness'])
        
        if settings.get('denoise', False):
            image = image.filter(ImageFilter.MedianFilter(size=3))
        
        if settings.get('threshold', False):
            image = image.convert('L')
            image = image.point(lambda x: 0 if x < 128 else 255, '1')
        
        return image

def get_tesseract_configs(settings):
    """Generate Tesseract configuration based on settings."""
    config = []
    
    if 'psm' in settings:
        config.append(f'--psm {settings["psm"]}')
    
    if 'oem' in settings:
        config.append(f'--oem {settings["oem"]}')
    
    if 'lang' in settings:
        config.append(f'-l {settings["lang"]}')
    
    if 'dpi' in settings:
        config.append(f'--dpi {settings["dpi"]}')
    
    return ' '.join(config)

def calculate_hash(text):
    """Calculate SHA-256 hash of the text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def extract_text_from_image(image, ocr_settings):
    """Extract text from a single image with specified settings."""
    preprocessor = ImagePreprocessor()
    processed_image = preprocessor.enhance_image(image, ocr_settings)
    config = get_tesseract_configs(ocr_settings)
    return pytesseract.image_to_string(processed_image, config=config)

def process_pdf(pdf_path, ocr_settings):
    """Process PDF file and extract text from all pages."""
    images = convert_from_path(pdf_path)
    full_text = ""
    for i, image in enumerate(images):
        text = extract_text_from_image(image, ocr_settings)
        full_text += f"Page {i+1}\n{text}\n\n"
    return full_text

def process_image(image_path, ocr_settings):
    """Process image file and extract text."""
    image = Image.open(image_path)
    text = extract_text_from_image(image, ocr_settings)
    return text

def perform_triple_ocr(input_path, optimizer):
    """Perform OCR three times and compare results, trying different settings until success."""
    file_extension = Path(input_path).suffix.lower()
    attempt = 0
    
    while True:
        # Get next set of settings
        settings = optimizer.get_next_settings(attempt)
        if settings is None:
            print("\nExhausted all optimization strategies without finding matching results.")
            return None
        
        attempt += 1
        print(f"\nOptimization attempt {attempt}")
        print("Current settings:", settings)
        
        results = []
        hashes = []
        
        try:
            for i in range(3):
                print(f"  Performing OCR {i+1}/3...")
                if file_extension == '.pdf':
                    text = process_pdf(input_path, settings)
                elif file_extension in ['.png', '.jpg', '.jpeg']:
                    text = process_image(input_path, settings)
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")
                
                results.append(text)
                hashes.append(calculate_hash(text))
            
            if len(set(hashes)) == 1:
                print("\nSuccess! All three OCR results match with current settings:")
                for key, value in settings.items():
                    print(f"  {key}: {value}")
                return results[0]
            else:
                print("Results don't match. Trying next optimization strategy...")
        
        except Exception as e:
            print(f"Error with current settings: {str(e)}")
            continue

def extract_text(input_path, output_txt_path=None):
    """Extract text with automatic parameter optimization."""
    try:
        optimizer = OCROptimizer()
        
        print("Starting OCR with automatic parameter optimization...")
        full_text = perform_triple_ocr(input_path, optimizer)
        
        if full_text:
            if output_txt_path:
                with open(output_txt_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                print(f"\nText extracted and saved to {output_txt_path}")
            return full_text
        else:
            print("\nFailed to achieve consistent OCR results after trying all optimization strategies.")
            return None
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract text from PDF or image files with automatic parameter optimization.')
    parser.add_argument('input_path', help='Path to the input file (PDF, PNG, JPG, or JPEG)')
    parser.add_argument('-o', '--output', help='Path to save the extracted text (optional)')
    
    args = parser.parse_args()
    
    if not args.output:
        input_path = Path(args.input_path)
        args.output = str(input_path.with_suffix('.txt'))
    
    result = extract_text(args.input_path, args.output)
    
    if result and not args.output:
        print("\nExtracted Text:")
        print("--------------")
        print(result)

if __name__ == "__main__":
    main()