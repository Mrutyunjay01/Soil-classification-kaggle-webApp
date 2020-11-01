import numpy as np
from keras.preprocessing.image import img_to_array, load_img

CLASSES = {
    0: "Alluvial Soil:-{ Rice,Wheat,Sugarcane,Maize,Cotton,Soyabean,Jute }",
    1: "Black Soil:-{ Virginia, Wheat , Jowar,Millets,Linseed,Castor,Sunflower} ",
    2: "Clay Soil:-{ Rice,Lettuce,Chard,Broccoli,Cabbage,Snap Beans }",
    3: "Red Soil:{ Cotton,Wheat,Pilses,Millets,OilSeeds,Potatoes }"
}


def model_predict(img_path, model):
    """
    Function to predict the class of input img by model serving via api.
    :param img_path:
    :param model:
    :return:
    """
    print("Predicted")
    img = load_img(img_path, target_size=(150, 150))
    img = img_to_array(img)
    img = img / 255.
    img = np.expand_dims(img, axis=0)

    result = np.argmax(model.predict(img))
    prediction = CLASSES[result]

    if result == 0:
        print("Alluvial.html")

        return "Alluvial", "alluvial.html"
    elif result == 1:
        print("Black.html")

        return "Black", "black.html"
    elif result == 2:
        print("Clay.html")

        return "Clay", "clay.html"
    elif result == 3:
        print("Red.html")

        return "Red", "red.html"
    pass
