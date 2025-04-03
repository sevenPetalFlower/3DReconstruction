import cv2
import numpy as np
import matplotlib.pyplot as plt


def getMatches(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use BFMatcher to match features
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches, kp1, kp2


def safeMatches(img1, img2, matches, kp1, kp2, numMatches, fileName):
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Create a new image by concatenating the two images side by side
    height = max(img1_color.shape[0], img2_color.shape[0])
    width = img1_color.shape[1] + img2_color.shape[1]
    combined_img = np.zeros((height, width, 3), dtype=np.uint8)

    combined_img[0:img1_color.shape[0], 0:img1_color.shape[1]] = img1_color
    combined_img[0:img2_color.shape[0], img1_color.shape[1]:] = img2_color
    # Define the line thickness you desire
    line_thickness = 3  # Increase this number to draw thicker lines

    # Draw the matches: iterate over the first 100 matches
    for m in matches[:numMatches]:
        # Get the matching keypoints coordinates from the first image
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        # Get the matching keypoints coordinates from the second image,
        # and add the width of the first image since the second image is placed to its right.
        pt2 = (int(kp2[m.trainIdx].pt[0] + img1_color.shape[1]), int(kp2[m.trainIdx].pt[1]))
        # Draw a line between the keypoints with the specified thickness and color (green here)
        cv2.line(combined_img, pt1, pt2, (0, 255, 0), thickness=line_thickness)

    # Convert from BGR (OpenCV format) to RGB (Matplotlib format) for correct display
    combined_img_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB)

    # Plot the combined image with increased line thickness
    plt.figure(figsize=(12, 6))
    plt.imshow(combined_img_rgb)
    plt.title("Feature Matches")
    plt.axis('off')
    plt.savefig(fileName)  # Save the figure to a file
    print("Figure saved as matched_features.png")
