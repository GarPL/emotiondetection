import cv2
import numpy as np
import tensorflow as tf

# Haarcascade yüz tespit dosyasını yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Eğitilmiş modeli yükle
model = tf.keras.models.load_model('emotion_detection_model.h5')

# Duygu etiketleri
emotion_labels = ['angry', 'happy', 'sad']

# Webcam'den görüntü al
cap = cv2.VideoCapture(0)

while True:
    # Kamera çerçevesini oku
    ret, frame = cap.read()
    
    # Görüntüyü griye dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Yüz bölgesini al
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=0)
        face = face / 255.0
        
        # Duygu tahmini yap
        predictions = model.predict(face)
        max_index = np.argmax(predictions[0])
        emotion = emotion_labels[max_index]
        
        # Tespit edilen yüzler üzerinde dikdörtgen çiz ve duygu etiketini ekle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Görüntüyü göster
    cv2.imshow('Emotion Detection', frame)

    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera ve pencereleri serbest bırak
cap.release()
cv2.destroyAllWindows()
