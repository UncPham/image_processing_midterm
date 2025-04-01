import cv2
import numpy as np

# Load the main image
image_path = 'images/1.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Adaptive Threshold to enhance objects
image_gray = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Scale factors to test (from 50% to 100%)
scale_factors = np.linspace(0.5, 1.0, 50)

# Loop through each template
for i in range(1, 16):
    # Load the template
    template_path = f'images/objects/{i}.jpg'
    template = cv2.imread(template_path)

    # Convert the template to grayscale
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Adaptive Threshold to enhance objects
    template_gray = cv2.adaptiveThreshold(template_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    h, w = template_gray.shape
    best_match = None  # Store the best match result (coordinates, scale, score)
    best_score = -1  # Highest matching score

    for scale in scale_factors:
        resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))

        # Create a mask for the template
        _, mask = cv2.threshold(resized_template, 240, 255, cv2.THRESH_BINARY_INV)

        # Perform template matching
        result = cv2.matchTemplate(
            image_gray, 
            resized_template, 
            cv2.TM_CCOEFF_NORMED, 
            mask=mask
        )
        
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        # Normalize the score to be between 0 and 1
        normalized_score = (maxVal + 1) / 2 

        # Update if a higher score is found
        if normalized_score > best_score:
            best_score = normalized_score
            best_match = (maxLoc, int(w * scale), int(h * scale))

    # If an object is found, draw a rectangle on the image
    if best_match:
        (x, y), w, h = best_match
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image, f"object {i} Score: {best_score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Save the result image
output_path = 'images/results/4.jpg'
cv2.imwrite(output_path, image)
