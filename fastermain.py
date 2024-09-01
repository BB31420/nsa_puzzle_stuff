import numpy as np
import cv2
import os
from itertools import product
import logging
from PIL import Image

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CustomSudoku:
    def __init__(self, n):
        self.n = n
        self.grid = np.zeros((n, n), dtype=int)
        self.horizontal_constraints = [['' for _ in range(n-1)] for _ in range(n)]
        self.vertical_constraints = [['' for _ in range(n)] for _ in range(n-1)]

    def add_constraint(self, pos1, pos2, constraint_type):
        row1, col1 = pos1
        row2, col2 = pos2
        if row1 == row2:  # Horizontal constraint
            self.horizontal_constraints[row1][min(col1, col2)] = constraint_type
        else:  # Vertical constraint
            self.vertical_constraints[min(row1, row2)][col1] = constraint_type

    def is_valid(self, num, pos):
        row, col = pos

        # Check row
        if num in self.grid[row]:
            return False

        # Check column
        if num in self.grid[:, col]:
            return False

        # Check box (assuming 4x4 grid with 2x2 boxes)
        box_size = 2
        box_x = col // box_size
        box_y = row // box_size
        box = self.grid[box_y*box_size:(box_y+1)*box_size, box_x*box_size:(box_x+1)*box_size]
        if num in box:
            return False

        # Check horizontal constraints
        if col > 0 and self.horizontal_constraints[row][col-1] == 'black_dot':
            if not self.check_black_dot_constraint(num, self.grid[row][col-1]):
                return False
        if col < self.n - 1 and self.horizontal_constraints[row][col] == 'black_dot':
            if not self.check_black_dot_constraint(num, self.grid[row][col+1]):
                return False

        # Check vertical constraints
        if row > 0 and self.vertical_constraints[row-1][col] == 'black_dot':
            if not self.check_black_dot_constraint(num, self.grid[row-1][col]):
                return False
        if row < self.n - 1 and self.vertical_constraints[row][col] == 'black_dot':
            if not self.check_black_dot_constraint(num, self.grid[row+1][col]):
                return False

        return True

    def check_black_dot_constraint(self, num1, num2):
        if num2 == 0:  # If the other cell is not filled yet
            return True
        return num1 == 2*num2 or num2 == 2*num1

    def solve(self):
        empty = self.find_empty()
        if not empty:
            return True
        row, col = empty

        for num in range(1, self.n + 1):
            if self.is_valid(num, (row, col)):
                self.grid[row][col] = num
                if self.solve():
                    return True
                self.grid[row][col] = 0

        return False

    def find_empty(self):
        for i, j in product(range(self.n), range(self.n)):
            if self.grid[i][j] == 0:
                return (i, j)
        return None

def preprocess_image(image_path):
    logger.debug(f"Attempting to process image: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"The image file '{image_path}' does not exist.")
        raise FileNotFoundError(f"The image file '{image_path}' does not exist.")
    
    if not os.path.isfile(image_path):
        logger.error(f"'{image_path}' is not a file.")
        raise ValueError(f"'{image_path}' is not a file.")
    
    file_size = os.path.getsize(image_path)
    if file_size == 0:
        logger.error(f"The file '{image_path}' is empty (0 bytes).")
        raise ValueError(f"The file '{image_path}' is empty (0 bytes).")
    
    logger.debug(f"File exists and is non-empty. Size: {file_size} bytes")
    
    if not os.access(image_path, os.R_OK):
        logger.error(f"No read permissions for '{image_path}'.")
        raise PermissionError(f"No read permissions for '{image_path}'.")
    
    logger.debug(f"File is readable. Attempting to read with cv2.imread()...")
    img = cv2.imread(image_path)
    
    if img is None:
        logger.error(f"cv2.imread() returned None for '{image_path}'.")
        raise ValueError(f"Unable to read the image file '{image_path}'. It may be corrupted or in an unsupported format.")
    
    logger.debug(f"Image read successfully. Shape: {img.shape}")
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logger.debug("Converted image to grayscale successfully.")
    except cv2.error as e:
        logger.error(f"Error in cv2.cvtColor: {str(e)}")
        raise ValueError(f"Error converting image to grayscale. Original error: {str(e)}")
    
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    logger.debug("Applied thresholding successfully.")
    
    return thresh

def find_grid(preprocessed_img):
    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grid_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(grid_contour)
    return x, y, w, h

def extract_grid_cells(img, grid_rect):
    x, y, w, h = grid_rect
    grid = img[y:y+h, x:x+w]
    cell_h, cell_w = h // 4, w // 4
    cells = []
    for i in range(4):
        for j in range(4):
            cell = grid[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            cells.append(cell)
    return cells

def load_templates():
    templates = {}
    try:
        templates['black_dot_horizontal'] = cv2.imread('black_dot_horizontal.png', 0)
        templates['black_dot_vertical'] = cv2.imread('black_dot_vertical.png', 0)
        if templates['black_dot_horizontal'] is None or templates['black_dot_vertical'] is None:
            raise FileNotFoundError
    except FileNotFoundError:
        logger.error("Error: Template image for black dot not found.")
        exit(1)
    return templates

def recognize_symbols(cells, templates):
    symbols = []
    for cell in cells:
        best_match = None
        best_score = -np.inf
        for symbol, template in templates.items():
            result = cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_match = symbol
        
        symbols.append(best_match if best_score > 0.17 else '') # threshold amount
    
    return symbols

def create_sudoku_from_image(image_path, templates):
    preprocessed = preprocess_image(image_path)
    grid_rect = find_grid(preprocessed)
    cells = extract_grid_cells(preprocessed, grid_rect)
    symbols = recognize_symbols(cells, templates)
    
    sudoku = CustomSudoku(4)  # Assuming 4x4 grid
    
    for i, symbol in enumerate(symbols):
        if symbol:
            row, col = i // 4, i % 4
            logger.info(f"Detected {symbol} at row {row}, column {col}")
            if symbol == 'black_dot_horizontal' and col > 0:
                sudoku.add_constraint((row, col-1), (row, col), 'black_dot')
            elif symbol == 'black_dot_vertical' and row > 0:
                sudoku.add_constraint((row-1, col), (row, col), 'black_dot')
    
    return sudoku, symbols


def visualize_detected_symbols(image_path, symbols):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    preprocessed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    grid_rect = find_grid(preprocessed)
    x, y, w, h = grid_rect
    cell_h, cell_w = h // 4, w // 4
    
    for i, symbol in enumerate(symbols):
        if symbol:
            row, col = i // 4, i % 4
            center_x = x + col * cell_w + cell_w // 2
            center_y = y + row * cell_h + cell_h // 2
            
            color = (0, 0, 255)  # Red color for black dots
            cv2.circle(img, (center_x, center_y), 5, color, -1)
            cv2.putText(img, 'BD', (center_x - 20, center_y + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow('Detected Symbols', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_image_reading(image_path):
    logger.info(f"Testing image reading for: {image_path}")
    
    try:
        pil_img = Image.open(image_path)
        logger.info(f"PIL successfully opened the image. Format: {pil_img.format}, Size: {pil_img.size}, Mode: {pil_img.mode}")
    except Exception as e:
        logger.error(f"PIL failed to open the image. Error: {str(e)}")
    
    cv_img = cv2.imread(image_path)
    if cv_img is None:
        logger.error("OpenCV failed to read the image (returned None)")
    else:
        logger.info(f"OpenCV successfully read the image. Shape: {cv_img.shape}")
    
    try:
        with open(image_path, 'rb') as f:
            raw_bytes = f.read(1024)  # Read first 1024 bytes
        logger.info(f"First few bytes of the file: {raw_bytes[:20].hex()}")
    except Exception as e:
        logger.error(f"Failed to read raw bytes from the file. Error: {str(e)}")

def list_directory_contents(path):
    logger.info(f"Contents of directory '{path}':")
    for item in os.listdir(path):
        logger.info(f"  - {item}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Current directory: {current_dir}")

    list_directory_contents(current_dir)

    image_filename = 'input.jpg'
    image_path = os.path.join(current_dir, image_filename)
    logger.info(f"Full image path: {image_path}")

    if os.path.exists(image_path):
        logger.info(f"Image file found: {image_path}")
    else:
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"The image file '{image_filename}' does not exist in the current directory.")

    try:
        test_image_reading(image_path)
        
        templates = load_templates()
        logger.info("Templates loaded successfully.")
        
        sudoku, symbols = create_sudoku_from_image(image_path, templates)
        logger.info("Sudoku object created from image.")
        
        visualize_detected_symbols(image_path, symbols)
        logger.info("Visualization completed.")

        if sudoku.solve():
            logger.info("Sudoku solved successfully.")
            print("Solved Sudoku:")
            print(sudoku.grid)
        else:
            logger.info("No solution exists for the Sudoku puzzle.")
            print("No solution exists")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")