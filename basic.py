import gc
import os
from time import gmtime, strftime

import matplotlib.pyplot as plt
from keras.optimizers import Adagrad
from keras.preprocessing.image import ImageDataGenerator

from config import config
from utils import create_models as cm

batch_size = 32
epochs = 40

save_model = True

# Set the image size.
img_height = 96
img_width = 96

################################
# LOADING AND PREPROCESSING DATA
################################

print("[INFO] Loading and preprocessing data")
print("[INFO] Testing on '{}' data".format(len(os.listdir(config.TEST_PATH))))

train_datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x,
                                   rescale=1./255,
                                   width_shift_range=4,
                                   height_shift_range=4,
                                   rotation_range=90,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)

test_datagen = ImageDataGenerator(preprocessing_function=lambda x:(x - x.mean()) / x.std() if x.std() > 0 else x, rescale=1./255)

print("[INFO] Loading and preprocessing training data")
train_generator = train_datagen.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(img_width, img_height),
    color_mode="rgb",
    shuffle=True,
    batch_size=batch_size)

print("[INFO] Loading and preprocessing validation data")
validation_generator = test_datagen.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(img_width, img_height),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size)

print("[INFO] Loading and preprocessing test data")
testing_generator = test_datagen.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(img_width, img_height),
    color_mode="rgb",
    shuffle=False,
    batch_size=batch_size)

################################
# LOADING MODEL
################################

print("[INFO] Loading model")

model = cm.create_mlp((img_height,img_width,3),2)
model.summary()

################################
# TRAINING MODEL
################################

print("[INFO] Training model")
opt = Adagrad(lr=1e-2, decay=1e-2 / epochs)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALIDDATION = validation_generator.n//validation_generator.batch_size

history = model.fit_generator(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=STEP_SIZE_VALIDDATION)

################################
# SAVING MODEL
################################

if (save_model):
    print("[INFO] Saving model")
    model_name = "model__" + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    model_json = model.to_json()
    with open(config.MODELS_PATH + "/" + model_name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(config.MODELS_PATH + "/" + model_name + ".h5")
    print("[INFO] Saved model '{}' to disk".format(model_name))


################################
# EVALUATING MODEL
################################

print("[INFO] Evluating model on testing data")

STEP_SIZE_TESTING = testing_generator.n//testing_generator.batch_size
t_loss, t_acc = model.evaluate_generator(testing_generator, steps=STEP_SIZE_TESTING)
print("[RESULT] LOSS : '{0}', \n\t ACCURACY : '{1}'".format(t_loss, t_acc))

################################
# TRAINING VISUALS
###############################

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epcs = range(1, len(acc) +1)

plt.plot(epcs,acc,'bo',label='Training acc')
plt.plot(epcs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epcs,loss,'bo',label='Training loss')
plt.plot(epcs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#clearing ram, make some free space
gc.collect()

print("[INFO] Done")
