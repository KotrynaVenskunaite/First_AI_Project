import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import load_model

new_model = keras.models.load_model("AI_Training\Tutorial\chainsawman_model.h5")

image_size = (180, 180)

img = keras.preprocessing.image.load_img(
    "Denji.png", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

predictions = new_model.predict(img_array)
score = float(predictions[0])
print(f"{100 * (1 - score):.2f}% - Unrelated")
print(f"{100 * score:.2f}% -  Denji.")