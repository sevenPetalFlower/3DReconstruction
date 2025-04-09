import cv2
from algorithms.structureFromMotion import get_matches, safe_matches
from utils.datasetOperations import get_test_2_images

img1, img2 = get_test_2_images()
imgGray1 = cv2.cvtColor(img1, cv2.IMREAD_GRAYSCALE)
imgGray2 = cv2.cvtColor(img2, cv2.IMREAD_GRAYSCALE)

matches, kp1, kp2 = get_matches(imgGray1, imgGray2)
safe_matches(img1, img2, matches, kp1, kp2, 50, "keyPointsMatch.png")