import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

IMG_SIZE = 64

data = []
labels = []

# Load cats
for img in os.listdir("data/cats")[:200]:
    try:
        path = os.path.join("data/cats", img)
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data.append(img_array.flatten())
        labels.append(0)
    except:
        pass

# Load dogs
for img in os.listdir("data/dogs")[:200]:
    try:
        path = os.path.join("data/dogs", img)
        img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        data.append(img_array.flatten())
        labels.append(1)
    except:
        pass

# Convert to numpy
X = np.array(data)
y = np.array(labels)

print("Dataset shape:", X.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.decomposition import PCA

pca = PCA(n_components=100)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Train SVM
model = SVC(kernel='rbf')
print("Training...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# ==============================
# PREDICTION SECTION
# ==============================

import matplotlib.pyplot as plt
import random

print("Running prediction...")   # 🔥 DEBUG LINE

category = random.choice(["cats", "dogs"])
folder = os.path.join("data", category)
img_name = random.choice(os.listdir(folder))
path = os.path.join(folder, img_name)

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray, (64, 64))
flat = resized.flatten().reshape(1, -1)

# Apply PCA if exists
try:
    flat = pca.transform(flat)
except:
    pass

pred = model.predict(flat)[0]
label = "Dog 🐶" if pred == 1 else "Cat 🐱"

print("Prediction:", label)   # 🔥 MUST PRINT

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title(f"Prediction: {label}")
plt.axis("off")
plt.show()