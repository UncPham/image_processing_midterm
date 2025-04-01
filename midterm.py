import cv2
import numpy as np

# Load ảnh chính và ảnh mẫu
image_path = 'images/1.jpg'
# template_folder = 'images/objects'

image = cv2.imread(image_path)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Adaptive Threshold để tăng cường đối tượng
image_gray = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Các tỷ lệ scale cần thử (từ 50% đến 100%)
scale_factors = np.linspace(0.5, 1.0, 50)
threshold = 0.8
for i in range(1, 16):
    template_path = f'images/objects/{i}.jpg'
    template = cv2.imread(template_path)

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # template_gray = cv2.adaptiveThreshold(template_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    h, w = template_gray.shape
    # threshold = 0.4  # Ngưỡng ban đầu
    best_match = None  # Lưu kết quả tốt nhất (tọa độ, scale, điểm số)
    best_score = -1  # Điểm số tốt nhất

    for scale in scale_factors:
        resized_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))

        _, mask = cv2.threshold(resized_template, 240, 255, cv2.THRESH_BINARY_INV)

        result = cv2.matchTemplate(
            image_gray, 
            resized_template, 
            cv2.TM_CCOEFF_NORMED, 
            mask=mask
        )
        
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)

        # Nếu tìm thấy điểm số cao hơn thì cập nhật
        if maxVal > best_score:
            best_score = maxVal
            best_match = (maxLoc, int(w * scale), int(h * scale))

    # Nếu tìm thấy vật thể, vẽ hình chữ nhật lên ảnh
    if best_match:
        (x, y), w, h = best_match
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image, f"object {i} Score: {best_score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


# Hiển thị kết quả
result_resized = cv2.resize(image, (1000, 800))
cv2.imshow('Improved Multi-Scale Template Matching', result_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()