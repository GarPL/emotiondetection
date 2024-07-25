import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Eğitim ve test veri setleri için veri artırma
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',  # Eğitim veri seti yolu
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'dataset/test',  # Test veri seti yolu
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical'
)

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size
# Modelin oluşturulması
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 sınıf: üzgün, öfkeli, mutlu
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Erken durdurma ve en iyi modeli kaydetme callback'leri
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_emotion_detection_model.h5.keras', save_best_only=True, monitor='val_loss')

# Modelin eğitilmesi
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,  # Her epoch'da kaç adım yapılacak
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[early_stopping, model_checkpoint]
  # Her epoch'da kaç doğrulama adımı yapılacak
)

# Modelin kaydedilmesi
model.save('emotion_detection_model.h5')
