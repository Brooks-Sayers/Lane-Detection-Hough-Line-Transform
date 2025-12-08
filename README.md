# Lane Detection - Hough Line Transform
This was my first attempt at building a lane detection model, using the **Probabilistic Hough Line Transform**, alongside **HLS color thresholding**, **Gaussian blur**, **Canny edge detection**, and a **defined region-of-interest mask**. I found the results to be satisfying, especially given how new this concept was to me. It required many hours of reading through OpenCV-Python's and MoviePy's extensive documentation and calibrating detection thresholds, but it helped me develop a deeper understanding of how computers interpret images and video. I know that this would produce better results once layered with other methods of lane detection. Nonetheless, this was a great learning opportunity, and I look forward to researching autonomous driving vehicles further.


## Features
- Detects lane lines in both **images (.png, .jpg)** and **videos (.mp4)**  
- Uses **adaptive color thresholding** to adapt to different brightness/saturation environments  
- Region-of-interest masking removes irrelevant noise outside the road  
- Uses **HoughLinesP** to estimate lane segments and draw them onto the frame  
- Allows user to select files through a simple Tkinter GUI  
- Generates a processed video (-Result.mp4) with visualized lane lines, or displays the images after processing using **Matplotlib.pyplot**


## Examples
| Input Image | Result |
|------------|---------|
| <img src="media/Road-Image2.png" width="400"> | <img src="media/Road-Image2-Result.png" width="550"> |

| Input Video | Result |
|------------|---------|
| <img src="media/Road-Video1.gif" width="400"> | <img src="media/Road-Video1-Result.gif" width="550"> |

