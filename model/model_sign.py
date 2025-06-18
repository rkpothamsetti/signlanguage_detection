from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import MobileNetV2

# Feature extraction model (sequential_1)
feature_extractor = Sequential([
    MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
    GlobalAveragePooling2D()
], name="sequential_1")

# Classification head (sequential_3)
classifier = Sequential([
    Dense(100, activation='relu'),
    Dense(3, activation='softmax')
], name="sequential_3")

# Combined model using outer Sequential
model = Sequential([
    feature_extractor,
    classifier
], name="sequential_combined")

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary
model.summary()
