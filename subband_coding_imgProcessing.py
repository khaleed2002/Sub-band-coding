import cv2
import numpy as np
g0=[0.23037781,0.7148457,0.63088076,-0.02798376,-0.18703481,0.03084138,0.03288301,-0.01059740]
g1=g0.copy()
g1.reverse()
for i in range(len(g1)):
    if not (i % 2==0):
        g1[i]=-1*g1[i]
h0=g0.copy()
h0.reverse()
h1=g1.copy()
h1.reverse()
#Read Image
img = cv2.imread('img1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Define Kernals As numpy array
g0=np.array(g0)
g1=np.array(g1)
h0=np.array(h0)
h1=np.array(h1)
# 2 filters h0_h0(Low Low (Approximation)) & h0_h1(Low High (Vertical details))

# First Filter Along Rows using h0
filtered_h0 = cv2.filter2D(gray, -1, h0, borderType=cv2.BORDER_CONSTANT)
height, width = filtered_h0.shape[:2]
# downsample along rows by factor 2
downsampled_h0 = cv2.resize(filtered_h0, (width, height//2), interpolation=cv2.INTER_AREA)

# Now Get LOW-LOW

# first filter along columns using h0
filtered_h0_h0 = cv2.filter2D(downsampled_h0, -1, h0.T, borderType=cv2.BORDER_CONSTANT)
height, width = filtered_h0_h0.shape[:2]
# downsample along columns by factor 2
downsampled_h0_h0 = cv2.resize(filtered_h0_h0, (width//2, height), interpolation=cv2.INTER_AREA)

# Now Get Low-High

# second filter along columns using h1
filtered_h0_h1 = cv2.filter2D(downsampled_h0, -1, h1.T, borderType=cv2.BORDER_CONSTANT)
# downsample along columns by factor 2
downsampled_h0_h1 = cv2.resize(filtered_h0_h1, (width//2, height), interpolation=cv2.INTER_AREA)

# Now we have LL & LH
LL = downsampled_h0_h0
LH = downsampled_h0_h1

# SECOND Get Horizontal and diagonal details

# 2 filters h1_h0(High Low (Horizontal details)) & h1_h1(High High (Diagonal details))

# First Filter Along Rows using h1
filtered_h1 = cv2.filter2D(gray, -1, h1, borderType=cv2.BORDER_CONSTANT)
height, width = filtered_h1.shape[:2]
# downsample along rows by factor 2
downsampled_h1 = cv2.resize(filtered_h1, (width, height//2), interpolation=cv2.INTER_AREA)

# Now Get High-LOW

# first filter along columns using h0
filtered_h1_h0 = cv2.filter2D(downsampled_h1, -1, h0.T, borderType=cv2.BORDER_CONSTANT)
height, width = filtered_h1_h0.shape[:2]
# downsample along columns by factor 2
downsampled_h1_h0 = cv2.resize(filtered_h1_h0, (width//2, height), interpolation=cv2.INTER_AREA)

# Now Get High-High

# second filter along columns using h1
filtered_h1_h1 = cv2.filter2D(downsampled_h1, -1, h1.T, borderType=cv2.BORDER_CONSTANT)
# downsample along columns by factor 2
downsampled_h1_h1 = cv2.resize(filtered_h1_h1, (width//2, height), interpolation=cv2.INTER_AREA)

# Now we have HL & HH
HL = downsampled_h1_h0
HH = downsampled_h1_h1

cv2.imshow('Original Image', gray)
cv2.waitKey(0)
cv2.imshow('Approximation Image', LL)
cv2.waitKey(0)
cv2.imshow('Vertical details Image', LH)
cv2.waitKey(0)
equalized_LH = cv2.equalizeHist(LH)
cv2.imshow('Vertical details Image(using Equalization)', equalized_LH)
cv2.waitKey(0)
cv2.imshow('Horizontal details Image', HL)
cv2.waitKey(0)
equalized_HL = cv2.equalizeHist(HL)
cv2.imshow('Horizontal details Image (using Equalization)', equalized_HL)
cv2.waitKey(0)
cv2.imshow('Diagonal details Image', HH)
cv2.waitKey(0)
equalized_HH = cv2.equalizeHist(HH)
cv2.imshow('Diagonal details Image (using Equalization)', equalized_HH)
cv2.waitKey(0)
cv2.destroyAllWindows()

