import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# plot config
plt.switch_backend('TkAgg')

# paths
base_dir = "path"
swieze_directory = r'path'
zgnite_directory = r'path'
test_dir = os.path.join(base_dir, "j_testy")

def load_images_with_labels(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(directory, filename))
            img = cv2.resize(img, (64, 64))  # rescaling
            images.append(img)
            labels.append(label)
    return images, labels

swieze_images, swieze_labels = load_images_with_labels(swieze_directory, 1)
zgnite_images, zgnite_labels = load_images_with_labels(zgnite_directory, 0)

all_images = zgnite_images + swieze_images
all_labels = zgnite_labels + swieze_labels

X_train = np.array(all_images)
y_train = np.array(all_labels)

X_test = []
y_test = []
for filename in os.listdir(test_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(test_dir, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64)) # rescaling
        X_test.append(img)
        
        if "FreshApple" in filename:
            label = 1
        elif "RottenApple" in filename:
            label = 0
        else:
            continue  
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

unique_classes_test = np.unique(y_test)
print(f"Unikalne klasy w zbiorze testowym: {unique_classes_test}")

# model layers
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# training
history = model.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

# pred.
y_pred_proba = model.predict(X_test)
y_pred = np.round(y_pred_proba)

# conf. matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# ROC plot
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Krzywa ROC (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Krzywa ROC')
plt.legend(loc="lower right")
plt.show()

plt.plot(history.history['loss'], label='Strata treningowa')
plt.plot(history.history['val_loss'], label='Strata walidacyjna')
plt.xlabel('Epoka')
plt.ylabel('Strata')
plt.legend()
plt.show()

unique_classes_train = np.unique(y_train)
print(f"Unikalne klasy w zbiorze treningowym: {unique_classes_train}")
