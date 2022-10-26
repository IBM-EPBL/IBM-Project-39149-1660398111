from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
model=load_model(r'D:\IBM Project\gesture.h5')
categories=['0','1','2','3','4','5']
test_img=image.load_img(r'Downloads\OIP4 image.jpg',target_size=(80,80),color_mode='grayscale')
pixels=image.img_to_array(test_img)
pixels=np.expand_dims(pixels,axis=0)
prediction=model.predict(pixels)
print(np.argmax(pred))
