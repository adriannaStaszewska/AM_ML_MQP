import cv2
import numpy as np
#vis brightness and adjustemnts
ROOT = "/home/azstaszewska/Data/MS Data/Stitched Final/"
OUT = "/home/azstaszewska/Data/MS Data/Viz/"
s = "R6R"
image_path = ROOT + s + ".png"
image = cv2.imread(image_path)

brightness_adj = cv2.addWeighted(image,1.5,np.zeros(image.shape, image.dtype),0,0)
cv2.imwrite(OUT+s+"_1_1.png", brightness_adj)
cropped_img_gray = cv2.cvtColor(brightness_adj, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(cropped_img_gray, 50, 200)
cv2.imwrite(OUT+s+"_1_2.png", edged)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
dilated = cv2.dilate(edged, kernel)
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
polygon = np.zeros(image.shape)
color = [255, 255, 255]
cv2.fillPoly(polygon, contours, color)
cv2.imwrite(OUT+s+"_contours_1.png", polygon)


brightness_adj = cv2.addWeighted(image,1.2,np.zeros(image.shape, image.dtype),0,0)
cv2.imwrite(OUT+s+"_2_1.png", brightness_adj)
cropped_img_gray = cv2.cvtColor(brightness_adj, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(cropped_img_gray, 100, 200)
cv2.imwrite(OUT+s+"_2_2.png", edged)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
dilated = cv2.dilate(edged, kernel)
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


polygon = np.zeros(image.shape)
color = [255, 255, 255]
cv2.fillPoly(polygon, contours, color)

cv2.imwrite(OUT+s+"_contours_2.png", polygon)


brightness_adj = cv2.addWeighted(image,1,np.zeros(image.shape, image.dtype),0,0)
cv2.imwrite(OUT+s+"_3_1.png", brightness_adj)
cropped_img_gray = cv2.cvtColor(brightness_adj, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(cropped_img_gray, 100, 200)
cv2.imwrite(OUT+s+"_3_2.png", edged)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
dilated = cv2.dilate(edged, kernel)
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


polygon = np.zeros(image.shape)
color = [255, 255, 255]
cv2.fillPoly(polygon, contours, color)

cv2.imwrite(OUT+s+"_contours_3.png", polygon)


brightness_adj = cv2.addWeighted(image,1.5,np.zeros(image.shape, image.dtype),0,0)
cv2.imwrite(OUT+s+"_4_1.png", brightness_adj)
cropped_img_gray = cv2.cvtColor(brightness_adj, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(cropped_img_gray, 100, 150)
cv2.imwrite(OUT+s+"_4_2.png", edged)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
dilated = cv2.dilate(edged, kernel)
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


polygon = np.zeros(image.shape)
color = [255, 255, 255]
cv2.fillPoly(polygon, contours, color)

cv2.imwrite(OUT+s+"_contours_4.png", polygon)



brightness_adj = cv2.addWeighted(image,1.25,np.zeros(image.shape, image.dtype),0,0)
cv2.imwrite(OUT+s+"_5_1.png", brightness_adj)
cropped_img_gray = cv2.cvtColor(brightness_adj, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(cropped_img_gray, 150, 200)
cv2.imwrite(OUT+s+"_5_2.png", edged)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
dilated = cv2.dilate(edged, kernel)
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


polygon = np.zeros(image.shape)
color = [255, 255, 255]
cv2.fillPoly(polygon, contours, color)

cv2.imwrite(OUT+s+"_contours_5.png", polygon)



brightness_adj = cv2.addWeighted(image,1.25,np.zeros(image.shape, image.dtype),0,0)
cv2.imwrite(OUT+s+"_6_1.png", brightness_adj)
cropped_img_gray = cv2.cvtColor(brightness_adj, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(cropped_img_gray, 100, 150)
cv2.imwrite(OUT+s+"_6_2.png", edged)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
dilated = cv2.dilate(edged, kernel)
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


polygon = np.zeros(image.shape)
color = [255, 255, 255]
cv2.fillPoly(polygon, contours, color)

cv2.imwrite(OUT+s+"_contours_6.png", polygon)


brightness_adj = cv2.addWeighted(image,1.25,np.zeros(image.shape, image.dtype),0,0)
cv2.imwrite(OUT+s+"_7_1.png", brightness_adj)
cropped_img_gray = cv2.cvtColor(brightness_adj, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(cropped_img_gray, 120, 200)
cv2.imwrite(OUT+s+"_7_2.png", edged)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
dilated = cv2.dilate(edged, kernel)
contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


polygon = np.zeros(image.shape)
color = [255, 255, 255]
cv2.fillPoly(polygon, contours, color)

cv2.imwrite(OUT+s+"_contours_7.png", polygon)
