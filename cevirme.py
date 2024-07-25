
import tensorflow as tf

model = tf.keras.models.load_model('emotion_detection_model.h5')

# TensorFlow Lite Converter'ı oluşturun
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Modeli optimize etme: Kuantizasyon
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Modeli TensorFlow Lite formatına dönüştürün
tflite_model = converter.convert()

# Modeli .tflite dosyası olarak kaydedin
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)