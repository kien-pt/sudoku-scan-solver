from keras.models import load_model
import numpy as np

model = load_model("./my_model.h5")


def classify_image(image):
    class_inx = int(model.predict_classes(image))
    prob_val = np.amax(model.predict(image))
    return class_inx, prob_val

