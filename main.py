import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from PIL import Image, ImageDraw, ImageEnhance
import pytesseract
import cv2
import easyocr
import numpy as np

# Set up Chrome options to avoid prompts and auto-download files
download_dir = os.path.join(os.getcwd(), 'downloads')  # Directory to store screenshots
os.makedirs(download_dir, exist_ok=True)  # Ensure the directory exists

options = Options()
options.headless = False  # Set to True if you don't want the browser window to appear

# Start Chrome with the specified options
driver = webdriver.Chrome(options=options)

driver.set_window_size(576, 600)

def capture_screenshot_of_element(element, screenshot_path):
    """Capture a screenshot of the full banner element."""
    try:
        # Take a screenshot of the element and save it to the specified path
        element.screenshot(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")
    except Exception as e:
        print(f"Error capturing screenshot: {e}")

def detect_significant_regions(image_path, top_ignore=0):
    """Detect significant regions in the image while ignoring the top portion. 
    Automatically detects if the image background is light or dark and inverts if necessary."""
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Convert to grayscale
        grayscale_img = img.convert("L")
        
        # Get pixel values and calculate the average brightness of the image
        pixels = list(grayscale_img.getdata())
        total_pixels = len(pixels)
        light_pixels = sum(1 for pixel in pixels if pixel > 200)  # Light pixel threshold

        # Automatically determine if the image needs to be inverted
        invert = light_pixels > total_pixels / 2  # More than half of the pixels are light
        
        if invert == False:
            grayscale_img = Image.eval(grayscale_img, lambda x: 255 - x)  # Invert image colors

        # Enhance contrast to make significant regions more noticeable
        enhancer = ImageEnhance.Contrast(grayscale_img)
        enhanced_img = enhancer.enhance(3)  # Enhance contrast by a higher factor (adjustable)

        # Convert to a binary image using a higher threshold for dark regions
        threshold = 100  # Lower threshold to detect dark regions (adjustable)
        binary_img = enhanced_img.point(lambda p: p < threshold and 255)  # Select dark regions

        # Convert the binary image to an RGB image so we can draw on it
        binary_img = binary_img.convert("RGB")

        # Draw a red rectangle around significant areas (detected by threshold)
        draw = ImageDraw.Draw(binary_img)

        # Get image dimensions
        width, height = binary_img.size
        
        # Define variables to track the bounding box of the significant regions
        min_x, min_y, max_x, max_y = width, height, 0, 0

        # Loop through all pixels but IGNORE the top `top_ignore` pixels
        for y in range(top_ignore, height):  # Start from `top_ignore` instead of 0
            for x in range(width):
                pixel = binary_img.getpixel((x, y))
                # If pixel is dark (significant region), update bounding box
                if pixel[0] > 200:  # Detect light regions (can adjust this threshold)
                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)

        # If we found significant regions, draw a rectangle around them
        if min_x < max_x and min_y < max_y:
            draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=5)
            print(f"Significant region found: ({min_x}, {min_y}), ({max_x}, {max_y})")

        # Save the new image with the red rectangle
        result_path = os.path.join(os.path.dirname(image_path), 'highlighted_banner_image.png')
        binary_img.save(result_path)
        print(f"Highlighted image saved to {result_path}")
        
        return (min_x, min_y, max_x, max_y)  # Return the coordinates of the significant region

    except Exception as e:
        print(f"Error processing the image: {e}")
        return None
    
def detect_text_bounding_box(image_path, output_path, significant_bbox=None):
    """Detect text in an image and draw a bounding box around all of the text using EasyOCR, excluding text in significant regions."""
    # Load the image with OpenCV
    img = cv2.imread(image_path)

    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])  # Can specify other languages as needed

    # Perform OCR to detect text regions
    results = reader.readtext(img)

    # Variables to store the overall bounding box for all text
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), -float('inf'), -float('inf')

    # Iterate over each detected text region
    for result in results:
        (top_left, top_right, bottom_right, bottom_left) = result[0]
        text = result[1]

        # Convert coordinates to integers
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # If significant region is provided, ignore text inside it
        if significant_bbox:
            # Extract text bounding box coordinates
            text_min_x, text_min_y = top_left
            text_max_x, text_max_y = bottom_right
            bg_min_x, bg_min_y, bg_max_x, bg_max_y = significant_bbox

            # Check for overlap: If the text box overlaps with the significant region, skip it
            if not (text_max_x < bg_min_x or text_min_x > bg_max_x or text_max_y < bg_min_y or text_min_y > bg_max_y):
                continue  # Ignore this text box if it overlaps with the significant region

        # Update the overall bounding box coordinates to include this text region
        min_x = min(min_x, top_left[0])
        min_y = min(min_y, top_left[1])
        max_x = max(max_x, bottom_right[0])
        max_y = max(max_y, bottom_right[1])

    # If we have found any valid text regions, draw the bounding box around all text
    if min_x < max_x and min_y < max_y:
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
        print(f"Bounding box around all text: ({min_x}, {min_y}), ({max_x}, {max_y})")

    # Save the image with the bounding box drawn
    cv2.imwrite(output_path, img)
    print(f"Bounding box image saved to {output_path}")

    return (min_x, min_y, max_x, max_y)  # Return the bounding box coordinates

def check_for_overlap(background_bbox, text_bbox):
    """Check if the significant background area overlaps with the text area."""
    bg_min_x, bg_min_y, bg_max_x, bg_max_y = background_bbox
    text_min_x, text_min_y, text_max_x, text_max_y = text_bbox

    # Check if the two bounding boxes overlap
    if bg_max_x > text_min_x and bg_min_x < text_max_x and bg_max_y > text_min_y and bg_min_y < text_max_y:
        return True  # There is overlap
    else:
        return False  # No overlap

def draw_overlap_on_banner(banner_path, text_bbox, overlap_bbox, download_dir=download_dir):
    """Draw the overlap between two bounding boxes on the banner image with a red box."""
    try:
        # Open the banner image
        banner_img = Image.open(banner_path)
        draw = ImageDraw.Draw(banner_img)

        # Compute the intersection of the two bounding boxes
        x1 = max(text_bbox[0], overlap_bbox[0])
        y1 = max(text_bbox[1], overlap_bbox[1])
        x2 = min(text_bbox[2], overlap_bbox[2])
        y2 = min(text_bbox[3], overlap_bbox[3])

        # Check if there is an actual overlap
        if x1 < x2 and y1 < y2:
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            print(f"Overlap drawn: {x1, y1, x2, y2}")
        else:
            print("No overlap detected.")

        # Save the image with the red rectangle around the overlap
        output_path = os.path.join(download_dir, 'banner_with_overlap.png')
        banner_img.save(output_path)
        print(f"Image with overlap saved to {output_path}")

        return output_path
    except Exception as e:
        print(f"Error drawing overlap: {e}")
        return None

try:
    # 1. Open the website
    driver.get('https://productstore2-us-preview.melaleuca.com/')

    time.sleep(5)

    # Stage login process
    stage_email = driver.find_element(By.NAME, 'loginfmt')
    stage_email.send_keys('')

    stage_email.send_keys(Keys.RETURN)

    time.sleep(3)

    stage_password = driver.find_element(By.NAME, 'passwd')
    stage_password.send_keys('')

    stage_password.send_keys(Keys.RETURN)

    time.sleep(3)

    # 2. Locate the button with the text "SIGN IN" and click it
    sign_in_link = driver.find_element(By.PARTIAL_LINK_TEXT, "SIGN IN")
    sign_in_link.click()

    # 3. Find the element with the name "username" and input "troseenus"
    username_field = driver.find_element(By.NAME, 'username')
    #username_field.send_keys('troseenus')
    username_field.send_keys('travistest')

    # 4. Find the element with the name "password" and input "password1"
    password_field = driver.find_element(By.NAME, 'password')
    #password_field.send_keys('password')
    password_field.send_keys('password')

    # Optionally, you could submit the form if needed:
    password_field.send_keys(Keys.RETURN)
         
    # Wait for the page to load after login
    time.sleep(2)  # Adjust as necessary based on load time

    # Navigate to the product store/supplements page
    driver.get('https://productstore2-us-preview.melaleuca.com/productstore/supplements')

    # Wait for the page to load (adjust this as necessary)
    time.sleep(30)

    fwb_element = driver.find_element("css selector", "a.m-fwBanner") 

    full_banner_path = os.path.join(download_dir, "full_banner.png")
    fwb_element.screenshot(full_banner_path)

    fwb_content_element = driver.find_element("class name", "m-fwBanner__columns")
    driver.execute_script("arguments[0].style.display = 'none';", fwb_content_element)

    no_text_path = os.path.join(download_dir, "no_text_banner.png")
    fwb_element.screenshot(no_text_path)

    driver.execute_script("arguments[0].style.display = 'block';", fwb_content_element)

    # Find the picture and hide it
    image_element = driver.find_element(By.CSS_SELECTOR, 'picture.a-genImg')
    driver.execute_script("arguments[0].style.display = 'none';", image_element)

    div_element = driver.find_element(By.CSS_SELECTOR, 'div.m-fwBanner__row.-media')

    # Use JavaScript to set the background-color to aqua
    driver.execute_script("arguments[0].style.backgroundColor = 'aqua';", div_element)

    text_only_path = os.path.join(download_dir, "text_only_banner.png")
    fwb_element.screenshot(text_only_path)

    # Open the screenshot with PIL
    img = Image.open(no_text_path)

    # Get the original dimensions
    width, height = img.size

    # Crop the top 1 pixel out
    cropped_img = img.crop((0, 1, width, height))  # (left, upper, right, lower)

    cropped_img.save(no_text_path)

    significant_bbox = detect_significant_regions(no_text_path)

    if significant_bbox:
        # Check for overlap between the significant region and the text region
          # Detect text and draw bounding box
        text_bbox = detect_text_bounding_box(full_banner_path, os.path.join(download_dir, "selected_text_banner.png"))
        if check_for_overlap(significant_bbox, text_bbox):
            overlap_path = draw_overlap_on_banner(full_banner_path, text_bbox, significant_bbox)
        else:
            print("No overlap detected between text and significant regions.")
    else:
        print("No significant content detected or error processing image.")
    
    time.sleep(10)

finally:
    # Close the driver after all operations
    driver.quit()
