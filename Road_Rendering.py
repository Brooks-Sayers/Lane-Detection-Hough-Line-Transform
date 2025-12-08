import tkinter as tk
from tkinter import filedialog
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import numpy as np
from moviepy import VideoFileClip

TARGET_WIDTH = 960


def hough_line_transform(masked_canny_image, raw_image, roi_mask):
    # Utilizes Probalistic Hough Line Transform to detect the line segments along the image/frame and draw them ontop of the original image/frame
    height = raw_image.shape[0]

    lines = cv.HoughLinesP(
        masked_canny_image, 
        rho = 1.5,
        theta = np.pi/180,
        threshold = 50,
        lines = np.array([]),
        minLineLength = (height*0.02),
        maxLineGap = (height*0.02)
    )

    if lines is not None: 
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            if abs(x2-x1) > 0: 
                m = (y2-y1)/(x2-x1)

                if  abs(m) > 0.4:
                    cv.line(raw_image, (x1, y1), (x2, y2), (0, 0, 170), 5)
    
    return masked_canny_image, raw_image, roi_mask


def region_of_interest(canny_edges, raw_image):
    # Creates the region of interest that has a shape of a polygon and apply to canny edges- so that only edges inside ROI are kept
    mask = np.zeros_like(canny_edges)

    height, width = canny_edges.shape[:2]

    roi_vertices = np.array([[
        (width*0.05, height),
        (width*0.25, height*0.5),
        (width*0.75, height*0.5),
        (width*0.95, height)]], 
        dtype=np.int32)
    
    cv.fillPoly(mask, roi_vertices, 255)
    masked_canny_image = cv.bitwise_and(canny_edges, mask)

    return hough_line_transform(masked_canny_image, raw_image, mask)


def gaussian_blur_and_canny(combined_mask, raw_image):
    # Apply Gaussian blur and Canny edge to the image/frame
    blurred_image = cv.GaussianBlur(combined_mask, (9,9), 0)
    canny_edges = cv.Canny(blurred_image, 120, 200)

    return region_of_interest(canny_edges, raw_image)


def color_threshold(image_to_edit, raw_image):
    # Apply HLS color thresholding to isolate the white and yellow lane markings that fall into the selected range of HLS
    hls_image = cv.cvtColor(image_to_edit, cv.COLOR_BGR2HLS)
    _, L, S = cv.split(hls_image)

    h_img = L.shape[0]

    road_lightness = L[int(h_img*0.5):, :]
    road_saturation = S[int(h_img*0.5):, :]

    l_thresh = np.percentile(road_lightness, 94)
    l_thresh = int(np.clip(l_thresh, 0, 255))

    s_thresh = np.percentile(road_saturation, 80)
    s_thresh = int(np.clip(s_thresh, 0, 255))

    white_mask = cv.inRange(
        hls_image,
        np.array([0, l_thresh, s_thresh], dtype=np.uint8),
        np.array([180, 255, 140], dtype=np.uint8)
    )

    yellow_mask = cv.inRange(
        hls_image,
        np.array([18, max(80, l_thresh-30), 110], dtype=np.uint8),
        np.array([35, 255, 255], dtype=np.uint8)
    )

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    white_mask = cv.morphologyEx(white_mask, cv.MORPH_CLOSE, kernel)
    yellow_mask = cv.morphologyEx(yellow_mask, cv.MORPH_CLOSE, kernel)

    combined_mask = cv.bitwise_or(white_mask, yellow_mask)

    return gaussian_blur_and_canny(combined_mask, raw_image)


def process_frame(frame):
    # Process each video frame individually from video given: resize, color-threshold, Gaussian blur, Canny edge, apply ROI, Hough transform, and draw lane lines
    height, width = frame.shape[:2]
    scale = TARGET_WIDTH / width
    frame = cv.resize(frame, (TARGET_WIDTH, int(height * scale)))

    frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    raw_image = frame_bgr.copy()
    image_to_edit = frame_bgr.copy()

    _, adjusted_frame, _ = color_threshold(image_to_edit, raw_image)

    return cv.cvtColor(adjusted_frame, cv.COLOR_BGR2RGB)


def plot_image(img):
    # Process an individual image given: resize, color-threshold, Gaussian blur, Canny edge, apply ROI, Hough transform, and draw lane lines
    # Then display the raw image, masked canny image, adjusted image, and the ROI
    height, width = img.shape[:2]
    scale = TARGET_WIDTH / width
    img = cv.resize(img, (TARGET_WIDTH, int(height * scale)))

    raw_image = img.copy()
    image_to_edit = img.copy()

    masked_canny_image, adjusted_image, roi = color_threshold(image_to_edit, raw_image)

    _, axes = plt.subplots(2, 2, figsize=(12,6))

    axes[0,0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    axes[0,0].set_title('Raw Image')

    axes[0,1].imshow(masked_canny_image, cmap='gray')
    axes[0,1].set_title("Canny Edge + Region of Interest Image")

    axes[1,0].imshow(cv.cvtColor(adjusted_image, cv.COLOR_BGR2RGB))
    axes[1,0].set_title("Adjusted Image")

    axes[1,1].imshow(roi)
    axes[1,1].set_title("Region Of Interest")

    plt.tight_layout()
    plt.show()


def select_file():
    # Opens a tkinter file dialog for the user to select an image/video, then process it accordingly
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select an Image or Video File",
        filetypes=[("Media Files", "*.png;*.jpg;*.mp4")]
    )

    root.destroy()

    if not file_path:
        print("No file selected. Program exited!")
        sys.exit(0)

    extension = file_path.lower().split(".")[-1]

    if extension in ['png', 'jpg']:
        img = cv.imread(file_path)

        if img is None:
            print("Couldn't open selected image. Program will exit.")
            sys.exit(0)
        
        plot_image(img)
    
    elif extension in ['mp4']:
        clip = VideoFileClip(file_path)

        if clip is None:
            print("Couldn't open selected video. Program will exit.")
            sys.exit(0)

        processed_clip = clip.image_transform(process_frame)
        output_path = file_path.rsplit(".", 1)[0] + "-Result.mp4"
        processed_clip.write_videofile(output_path, audio=False)

        return
    
    else:
        print("Unsupported file type.")
        sys.exit(0)


if __name__ == '__main__':
    select_file()
