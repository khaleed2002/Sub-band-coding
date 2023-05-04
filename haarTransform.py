import math
import numpy as np

def generate_p_q(n):
    p = []
    for i in range(n):
        p.append(i)
    q = [[0, 1]]
    for i in range(1, len(p)):
        range_up = 2 ** i
        q_tmp = []
        for l in range(1, range_up + 1):
            q_tmp.append(l)
        q.append(q_tmp)
    return p, q


def generate_k(p, q):
    k = []
    for i in p:
        for j in q[i]:
            k.append([i, j])
    return k


def generate_Z(N):
    Z = []
    for i in range(N):
        Z.append(i / N)
    return Z

def generate_haar_matrix(k,Z,N):
    haar_matrix = np.zeros((N,N))
    for i in range(N):
        haar_matrix[0][i] = 1
    for i in range(1,len(k)):
        p = k[i][0]
        q = k[i][1]
        positive = 2 ** (p / 2)
        negative = -1 * positive
        small_positive = (q - 1) / (2 ** p)
        big_positive = (q - 0.5) / (2 ** p)
        small_negative = (q - 0.5) / (2 ** p)
        big_negative = q / (2 ** p)

        for j in range(len(Z)):
            if small_positive <= Z[j] < big_positive:
                haar_matrix[i][j] = positive
            elif  small_negative <= Z[j] < big_negative:
                haar_matrix[i][j] = negative
            else:
                haar_matrix[i][j] = 0
    haar_matrix *= (1/math.sqrt(N))
    return haar_matrix

N = int(input("Enter size of the kernal: "))

n = int(math.log2(N))
p, q = generate_p_q(n)

k = generate_k(p, q)
Z = generate_Z(N)
Haar_transform_matrix = generate_haar_matrix(k,Z,N)
print(Haar_transform_matrix)
# Now perform 2 rows of these matrix in subband coding
import cv2
g0 = Haar_transform_matrix[0]
g1 = Haar_transform_matrix[1]
h0 = np.flip(g0)
h1 = np.flip(g1)
print(h0)
print(h1)
#Read Image
img = cv2.imread('img1.png') # enter the path for image you want to edit
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

