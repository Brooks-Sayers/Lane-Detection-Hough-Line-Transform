import tkinter as tk
from tkinter import filedialog
import cv2 as cv
import matplotlib.pyplot as plt
import sys
import numpy as np
from moviepy import VideoFileClip

'''
Goals for project
Finish by 12/08/25 - started 12/04/25 ----------- After finishing this, in a state im happy with at least. I will say it took many hours, but this was a very new concept to me. It took a lot of studying the libraries documentation and the core principles of Hough Line Transform.
Functional for both images and videos ----------- Obviously I wouldn't throw this in a autonomous driving vehicle and trust my life with it; however, I am very happy with the outcome, and think with more systems added to it could prove very successful.
Clean and well structured code with many comments, so that maybe even someone entirely new to this concept could follow along ----------- I'll admit there might be too many comments, but it was helpful throughout studying this to be able to reread my thought process.
'''

# This function utilizes Hough Line Transform to map the lines along the image/frame
def hough_line_transform(maskedCannyImage, rawImageDupe, imageROI):
    h, w = rawImageDupe.shape[:2]
# Maps all the edge points from given image/frame, with a corresponding rho and theta value measured from the origin. Then esentially tallies a number of votes of edge pixels in a line, and if there is enough then creates the line
    lines = cv.HoughLinesP(
        maskedCannyImage, 
        rho = 1.5, # Distance resolution in pixels of the Hough grid 
        theta = np.pi/180, # Angular resolution in radians of the Hough grid
        threshold = 50, # Minimum number of votes (intersections in Hough grid)
        lines = np.array([]), # Array that holds the positions of lines detected
        minLineLength = (h*0.02), # Minimum number of pixels making up a line
        maxLineGap = (h*0.02)  # Maximum gap in pixels between connectable line segments
    )


    # Cycles through the lines that were detected
    if lines is not None: # Reality check to ensure the array (lines) is created before attempting to cycle through it
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4) # Unpacks line coordinates found in Hough Line Transform
            if abs(x2-x1) > 0: # Reality Check to ensure that the program is never dividing (y2-y1) by 0
                m = (y2-y1)/(x2-x1) # The slope of the line
                if  abs(m) > 0.4: # So far I haven't seen any need to pick up on horizontal lines, so this elimates those
                    cv.line(rawImageDupe, (x1, y1), (x2, y2), (0, 0, 170), 6) # Draws the line detected in a crimson color, and a thickness of 6 pixels
    
    return maskedCannyImage, rawImageDupe, imageROI


# Creates the region of interest
def region_of_interest(cannyEdge, rawImageDupe):
    # Applies a region of interest mask over the nearly finalized image, this will make it so the only noise being detected has to be within the region of the road
    # Creates a blank image to draw the polygon on (the selected region of interest)
    mask = np.zeros_like(cannyEdge)

    # Grabs the height and width of the image/frame provided
    height = cannyEdge.shape[0]
    width = cannyEdge.shape[1]

    roi_vertices = np.array([[
    (width*0.05, height), # Bottom left corner
    (width*0.25, height*0.5), # Top left corner
    (width*0.75, height*0.5), # Top right corner
    (width*0.95, height)]], # Bottom left corner
    dtype=np.int32) # Necessary to specify this arrays elements as 32-bit integers or else openCV throws a fit
    
    # Creates the polygon shape with the given vertices above
    cv.fillPoly(mask, roi_vertices, 255)
    
    # Combines the original cannyEdge image/frame with the region of interest, in order to elimate all edges detected outside of the region of interest
    maskedCannyImage = cv.bitwise_and(cannyEdge, mask)

    return hough_line_transform(maskedCannyImage, rawImageDupe, mask)


# This function applies Gaussian blur and canny edge to the image/frame
def gaussian_blur_canny_edge(combinedMask, rawImageDupe):
    # Applies a blur to the photo to elimate some noise
    blurredImage = cv.GaussianBlur(combinedMask, (9,9), 0)

    # Applies Canny edge to image/frame which detects all pixels along edges in image/frame
    cannyEdge = cv.Canny(blurredImage, 120, 200)

    return region_of_interest(cannyEdge, rawImageDupe)
    

# This function was a huge upgrade from the simple grayscale function I was working with before. Despite openCV documentation suggesting grayscale for projects like this one, I find color_thresholding much more effective
def color_threshold_image(imageToEdit, rawImageDupe):
    hls_image = cv.cvtColor(imageToEdit, cv.COLOR_BGR2HLS)
    H, L, S = cv.split(hls_image)

    # For the dynamic thresholding below, I found that pulling the lighting and saturation levels from the lower half of the image/frame (the road) gave the best results
    h_img = L.shape[0]
    road_L = L[int(h_img*0.5):, :]
    road_S = S[int(h_img*0.5):, :]

    # Dynamic thresholding- I don't want the renderings to be ruined by a difference in weather (gloomy cloudy vs sunny), so I make it so that the lighting and saturation varies image/frame
    L_thresh = np.percentile(road_L, 94)
    L_thresh = int(np.clip(L_thresh, 0, 255))
    S_thresh = np.percentile(road_S, 80)
    S_thresh = int(np.clip(S_thresh, 0, 255))

    ## This color thresholding is purposed to elimate any noise in the image/frame that isn't in the given range of yellow or white (colors of the lines on the road)
    ## Meaning colors like red or black for example wouldn't be picked up in the rendering
    # White mask range
    white_mask = cv.inRange(
        hls_image,
        np.array([0, L_thresh, S_thresh], dtype=np.uint8),
        np.array([180, 255, 140], dtype=np.uint8)
    )

    # Yellow mask range
    yellow_mask = cv.inRange(
        hls_image,
        np.array([18, max(80, L_thresh-30), 110], dtype=np.uint8),
        np.array([35, 255, 255], dtype=np.uint8)
    )

    # Very helpful in filling in the spots of what are to believed detected edges
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    white_mask = cv.morphologyEx(white_mask, cv.MORPH_CLOSE, kernel)
    yellow_mask = cv.morphologyEx(yellow_mask, cv.MORPH_CLOSE, kernel)

    # Combine the masks
    combinedMask = cv.bitwise_or(white_mask, yellow_mask)
    return gaussian_blur_canny_edge(combinedMask, rawImageDupe)


# This function is used whenever the program is given a video, processes each individual frame of the video and returns the adjusted frame
def process_frame(frame):
    # Resize given frame to a set ratio, where given (height, width) becomes (height * (960 / width), 960)
    TARGET_WIDTH = 960
    h, w = frame.shape[:2]
    scale = TARGET_WIDTH / w
    frame = cv.resize(frame, (TARGET_WIDTH, int(h * scale)))

    frame_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    rawImageDupe = frame_bgr.copy()
    imageToEdit = frame_bgr.copy()

    _, adjustedImage, _ = color_threshold_image(imageToEdit, rawImageDupe)

    return cv.cvtColor(adjustedImage, cv.COLOR_BGR2RGB)


# Plots the images using matplotlib, which I found helpful especially in creating the region of interest, as the plot has built in coordinates
def plot_image(img):
    # Resize given image to a set ratio, where given (height, width) becomes (height * (960 / width), 960)
    TARGET_WIDTH = 960
    h, w = img.shape[:2]
    scale = TARGET_WIDTH / w
    img = cv.resize(img, (TARGET_WIDTH, int(h * scale)))

    # Needed copies of the original image so that I could display each notable step throughout the images being processed
    # This proved very helpful during calibrating the multitude of values that changed which edges would be picked up
    rawImageDupe = img.copy()
    imageToEdit = img.copy()

    # Constructs the plot objects with 1 row, 2 columns, and size of window as (16,8)
    fig, axes = plt.subplots(2,2,figsize=(12,6))

    # Defines the maskedCannyImage, adjustedImage, and RegionOfInterest images I want to display. Pulls them from the functions above after processing the images
    maskedCannyImage, adjustedImage, regionOfInterest = color_threshold_image(imageToEdit, rawImageDupe)

    ## img is the completely raw, no edits image
    ## cannyImage is heavily edited image- color threshold, gaussian blur, canny edge, and region of interested applied
    ## adjustedImage is nearly the raw image; however, it has the lines drawn on it using the information gathered from the cannyImage and Hough Line Transform
    ## regionOfInterest is self explanatory, it just displays the region of interest that is selected

    # Create the individual plots for the raw image and the edited images
    axes[0,0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)) # Raw image
    axes[0,0].set_title('Raw Image') # Raw Image
    axes[0,1].imshow(maskedCannyImage, cmap='gray') # Heavily edited, canny image
    axes[0,1].set_title("Canny Edge + Region of Interest Image") # Heavily edited, canny image
    axes[1,0].imshow(cv.cvtColor(adjustedImage, cv.COLOR_BGR2RGB)) # Raw image with lines highlighted
    axes[1,0].set_title("Adjusted Image") # Raw image with lines highlighted
    axes[1,1].imshow(regionOfInterest) # Region of Interest
    axes[1,1].set_title("Region Of Interest") # Region of Interest

    # Creates the matplotlib window to display plots
    plt.tight_layout()
    plt.show()


# This function opens users files on computer and prompts them to select a image or video to attempt Hough Line Transform technique on to dectect lanes on a road
def select_image():
    
    # Create the root/window for tkinter to allow user to access their files easily upon running program
    root = tk.Tk()
    root.withdraw()

    # Opens files on computer to allow user to easily select image or video to process
    file_path = filedialog.askopenfilename(
        title="Select an Image File",
        filetypes=[("Image Files", "*.png;*.jpg;*.mp4")]
    )

    # Remove file dialog once done
    root.destroy()

    # If no file path was selected exit program
    if not file_path:
        print("No file selected. Program exited!")
        sys.exit(0)

    # Determines the extension of the file given
    ext = file_path.lower().split(".")[-1]

    # If the given file is an image then continue
    if ext in ['png', 'jpg']:
        # Grabs image from files with given file path
        img = cv.imread(file_path)

        # If openCV can't find or read image exits program
        if img is None:
            tk.messagebox.showerror("Error", "Couldn't open selected image. Program will exit.")
            sys.exit(0)
        
        # Once openCV has read the image file- start processing the image
        plot_image(img)
    
    # If the given file is a video then continue
    elif ext in ['mp4']:
        # Grabs video from files with given file path
        clip = VideoFileClip(file_path)

        # If moviepy can't find or read video exits program
        if clip is None:
            tk.messagebox.showerror("Error", "Couldn't open selected image. Program will exit.")
            sys.exit(0)

        # This goes thru and process each individual frame of the given video and then saves it to the folder the original video was found on the device
        processed_clip = clip.image_transform(process_frame)
        output_path = file_path.rsplit(".", 1)[0] + "-Result.mp4"
        processed_clip.write_videofile(output_path, audio=False)

        return
    
    # If given a file that's not a video or image, then exit the program
    else:
        print("Unsupported file type.")
        sys.exit(0)


# Runs program for images in order of select_image(), plot_image() - which runs color_threshold_image(), gaussian_blur_canny_edge(), region_of_interest(), hough_line_transform()
# Alternatively for videos runs in order of select_image(), process_frame() - which runs color_threshold_image(), gaussian_blur_canny_edge(), region_of_interest(), hough_line_transform()
if __name__ == '__main__':
    select_image()