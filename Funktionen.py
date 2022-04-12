from operator import index
import cv2

import matplotlib.pyplot as plt
from os import listdir
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.utils import shuffle #shuffling the data improves the model

# from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten , Activation, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras import optimizers, Sequential
import tensorflow as tf


from tqdm import tqdm_notebook as tqdm
import urllib.request
import tensorflow_probability as tfp

from sklearn.utils import class_weight


###########################################################################
###### Bilder einlesen mit Normalisierung
###########################################################################

def load_data(dir_list, image_size):
    X = []
    y = []
    image_width, image_height = image_size

    for dir in dir_list:
        for filename in listdir(dir):
            image = cv2.imread(dir+'/'+filename)
            image = cv2.resize(image, dsize=(image_width, image_height), interpolation=cv2.INTER_CUBIC)

            # Normalisierung
            image = image / 255


            X.append(image)

            if dir[-3:] == 'yes':
                y.append([1])
            else:
                y.append([0])
       

    X = np.array(X)
    y = np.array(y)


    # Data shuffle
    X, y = shuffle(X, y, random_state=64)

    print(f'Number of examples is: {len(X)}')
    print(f'X shape is: {X.shape}')
    print(f'y shape is: {y.shape}')

    return X, y

###########################################################################
###### Split Data
###########################################################################

def split_data(X, y, test_size=0.2, seed = 128):
       
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=seed) #128
    
    return X_train, y_train, X_val, y_val, X_test, y_test

###########################################################################
###### Scores
###########################################################################

def scores_res(model, pred, pred_proba,X_test, y_test_2, y_test, model_type, ind, base_dir, foldername):
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test,pred)
    prec = precision_score(y_test, pred)
    recall = recall_score(y_test,pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    if model_type == 'RF':
        nll_cross = None
    else: 
        nll_cross = model.evaluate(X_test,y_test_2,verbose=0)[0]


    res = pd.DataFrame(
            {'Acc' : acc, 'f1' : f1, 'Precision' : prec, 'Recall' : recall, 'ROC_AUC' : roc_auc, 'NLL-Cross' : nll_cross}, index=[ind]
    )

    save_scores_csv(base_dir, foldername, res)
    
    return res

###########################################################################
###### GridSearch - Result
###########################################################################

def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))

###########################################################################
###### Save - Modell Parameters
###########################################################################

def save_model_parameters(base_dir, folder_name, parameter_string):
    import os
    # Überprüfe ob Ordner existiert
    isExist = os.path.exists(base_dir + folder_name)

    if not isExist:
        # Erstelle ein neuer Ordner
        os.makedirs(base_dir + folder_name)
        print("Neuer Ordner wurde erstellt!")
    
    # Speichere die Parameter als text file
    with open(base_dir + folder_name + "/Model_Parameter.txt", "w") as text_file:
        text_file.write(parameter_string)
        text_file.close()

###########################################################################
###### Save - Modell Summary
###########################################################################

def save_model_summary(base_dir, folder_name, model):
    import os
    # Überprüfe ob Ordner existiert
    isExist = os.path.exists(base_dir + folder_name)

    if not isExist:
        # Erstelle ein neuer Ordner
        os.makedirs(base_dir + folder_name)
        print("Neuer Ordner wurde erstellt!")
    
    # Speichere die Parameter als text file
    with open(base_dir + folder_name + "/Model_Summary.txt", "w") as text_file:
        model.summary(print_fn=lambda x: text_file.write(x + '\n'))

###########################################################################
###### Save - Data split informations
###########################################################################

def save_data_split_infos(base_dir, folder_name, split_info_string):
    import os
    # Überprüfe ob Ordner existiert
    isExist = os.path.exists(base_dir + folder_name)

    if not isExist:
        # Erstelle ein neuer Ordner
        os.makedirs(base_dir + folder_name)
        print("Neuer Ordner wurde erstellt!")
    
    # Speichere die Parameter als text file
    with open(base_dir + folder_name + "/Data_Split_Info.txt", "w") as text_file:
        text_file.write(split_info_string)
        text_file.close()

###########################################################################
###### Save - Scores as CSV
###########################################################################

def save_scores_csv(base_dir, folder_name, scores_pd):
    import os
    # Überprüfe ob Ordner existiert
    isExist = os.path.exists(base_dir + folder_name)

    if not isExist:
        # Erstelle ein neuer Ordner
        os.makedirs(base_dir + folder_name)
        print("Neuer Ordner wurde erstellt!")

    # Scores abspeichern für das spezifische Modell
    scores_pd.to_csv(base_dir + folder_name + "/scores.csv", sep=";", index = True)
    print("Scores abgspeichert")

    # Überprüfe ob Score File exisiert wo alle Modelle abgespeichert werden
    isExist_csv = os.path.exists(base_dir + "scores_all.csv")

    if not isExist_csv:
        scores_pd.to_csv(base_dir + "scores_all.csv", sep=";", index = True)
        print("Scores_all.csv file erstellt")
    else:
        df = pd.read_csv(base_dir + "scores_all.csv", sep=";", index_col = 0)
        if folder_name in df.index:
            df.update(scores_pd)
            df.to_csv(base_dir + "scores_all.csv", sep=";", index = True)
            print("Scores_all.csv: Modell war bereits vorhanden und die Scores wurden aktualisiert")
        else:
            scores_pd.to_csv(base_dir + "scores_all.csv", sep=";", mode='a', index = True,  header = False)
            print("Scores zur scores_all.csv hinzugefügt")

###########################################################################
###### Plot & Save Confusion Matrix
###########################################################################

def plot_save_confusion_matrix(base_dir, folder_name, y_test, y_pred):
    import os
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.metrics import confusion_matrix

    # Überprüfe ob Ordner existiert
    isExist = os.path.exists(base_dir + folder_name)

    if not isExist:
        # Erstelle ein neuer Ordner
        os.makedirs(base_dir + folder_name)
        print("Neuer Ordner wurde erstellt!")

    labels = ["No Tumor", "Tumor"]
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig("./Results/"+folder_name+"/confusion_matrix.png", facecolor = 'white', transparent=False)
    plt.show()

    print("Plot gespeichert")



###########################################################################
###### Plot Test Images
###########################################################################

def plot_sample_images(X, y, n=40):
    for label in [0,1]:
        # grab the first n images with the corresponding y values equal to label
        images = X[np.argwhere(y == label)]
        n_images = images[:n]
        
        columns_n = 10
        rows_n = int(n/ columns_n)

        plt.figure(figsize=(10, 8))
        
        i = 1 # current plot        
        for image in n_images:
            plt.subplot(rows_n, columns_n, i)
            plt.imshow(image[0])
            
            # remove ticks
            plt.tick_params(axis='both', which='both', 
                            top=False, bottom=False, left=False, right=False,
                           labelbottom=False, labeltop=False, labelleft=False, labelright=False)
            
            i += 1
        
        label_to_str = lambda label: "Yes" if label == 1 else "No"
        plt.suptitle(f"Brain Tumor: {label_to_str(label)}")
        plt.show()

###########################################################################
###### Train Model
###########################################################################

def train_model(model, X_train, y_train, X_val, y_val, class_weights, batch_size, epochs, augmentation = False):
    if augmentation == False:
        history = model.fit(X_train, y_train, 
                        class_weight = class_weights,
                        batch_size=batch_size,
                        epochs=epochs, validation_data=(X_val, y_val),verbose=1)
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        rescale=1./255,
        shear_range=0.05,
        brightness_range=[0.8, 1.5],
        horizontal_flip=True,
        vertical_flip=True
        )

        history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), 
                              steps_per_epoch=len(X_train)/batch_size, 
                              epochs=epochs, 
                              class_weight = class_weights,
                              validation_data=(X_val, y_val),
                              verbose=1)

    return history

###########################################################################
###### Plot Accuracy & Loss Curve & Save
###########################################################################

def plot_save_accuracy_loss_curve(history, foldername):

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,(1))
    plt.plot(history.history['accuracy'],linestyle='-.')
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='lower right')
    plt.subplot(1,2,(2))
    plt.plot(history.history['loss'],linestyle='-.')
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig("./Results/"+foldername+"/accuracy_loss_curve.png", facecolor = 'white', transparent=False)

###########################################################################
###### Plot images - wrong predictions
###########################################################################

def plot_images_wrong_prediction(X_test, y_test, y_pred, foldername):

    from mpl_toolkits.axes_grid1 import ImageGrid

    index = []
    for i in range(0,len(y_test)):
        if y_test[i][0] != y_pred[i]:
            index.append(i)     

    images = []
    for i in range(0,len(index)):
        images.append(X_test[index[i]])


    fig = plt.figure(figsize=(15., 15.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1, len(index)),  # creates 2x2 grid of axes
                    axes_pad=0.2,  # pad between axes in inch.
                    )

    i = 0
    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        pred_label = "Tumor" if y_pred[index[i]] == 1 else "No Tumor"
        ax.set_title('Predicted: {}'.format(pred_label))
        i += 1
    plt.show()
    fig.savefig("./Results/"+foldername+"/wrong_prediction.png", facecolor = 'white')

###########################################################################
###### Prediction CNN
###########################################################################

def cnn_prediction(model, X_test, mc_dropout = False):

    if mc_dropout == True:

        # import tensorflow.keras.backend as k

        model_mc_pred = tf.keras.backend.function([model.input, tf.keras.backend.learning_phase()], [model.output])

        pred_mc=np.zeros((len(X_test),2))
        pred_max_p_mc=np.zeros((len(X_test)))
        pred_std_mc=np.zeros((len(X_test)))
        entropy_mc = np.zeros((len(X_test)))

        for i in tqdm(range(0,len(X_test))):
            multi_img=np.tile(X_test[i],(50,1,1,1))
            preds=model_mc_pred([multi_img,1])
            pred_mc[i]= np.mean(preds,axis=1)
            pred_max_p_mc[i]=np.argmax(np.mean(preds,axis=1))#mean over n runs of every proba class
            pred_std_mc[i]= np.sqrt(np.sum(np.var(preds, axis=1)))
            entropy_mc[i] = -np.sum( pred_mc[i] * np.log2(pred_mc[i] + 1E-14)) #Numerical Stability

        pred=np.array([np.argmax(pred_mc[i]) for i in range(0,len(pred_mc))])
        pred_prob = np.array([np.max(pred_mc[i]) for i in range(0,len(pred_mc))])

    else:
        pred = np.argmax(model.predict(X_test),axis=1)
        pred_prob = model.predict(X_test)[:,1]

    return pred, pred_prob

###########################################################################
###### Random Forest
###########################################################################

def random_forest(X_train, y_train, n_trees = 40, img_width = 32, img_height=32, seed = 128):
    clf = RandomForestClassifier(n_estimators=n_trees)
    clf.fit(X_train.reshape(len(X_train),img_width*img_height*3), np.argmax(y_train,axis=1))

    foldername = "RandomForest_" + str(n_trees) + str(img_width) + str(img_height) + str(seed)

    return clf, foldername

###########################################################################
###### Prediction Random Forest
###########################################################################

def rf_prediction(clf, X_test, img_width=32, img_height=32):

    pred=clf.predict(X_test.reshape(len(X_test),img_width*img_height*3))
    pred_prob = clf.predict_proba(X_test.reshape(len(X_test),img_width*img_height*3))[:,1]

    return pred, pred_prob

###########################################################################
###### Random Forest VGG16
###########################################################################

def random_forest_vgg16(X_train, y_train, X_test, n_trees = 40, img_width = 32, img_height=32, seed = 128):

    # load the pretrained vgg model
    base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,input_shape=(img_width,img_height,3),pooling="avg")

    # extract the vgg features of the images
    X_train_vgg_features=base_model.predict(X_train)
    X_test_vgg_features=base_model.predict(X_test)

    # train a random forest on the vgg features
    clf = RandomForestClassifier(n_estimators=n_trees,random_state=22)
    clf.fit(X_train_vgg_features, np.argmax(y_train,axis=1))


    foldername = "RandomForest_VGG16_" + str(n_trees) + str(img_width) + str(img_height) + str(seed)

    return clf, X_test_vgg_features, foldername


###########################################################################
###### Prediction Random Forest VGG 16
###########################################################################

def rf_prediction_vgg16(clf, X_test_vgg_features):

    pred=clf.predict(X_test_vgg_features)
    pred_prob = clf.predict_proba(X_test_vgg_features)[:,1]

    return pred, pred_prob

###########################################################################
###### Read File mit den Model Parametern
###########################################################################
def read_model_parameter_file(file_dir = "models.xlsx"):
    import pandas as pd
    df = pd.read_excel(file_dir)

    return df

###########################################################################
###### Extrahiere Model Parameter
###########################################################################
def extract_parameters(df, ind):

    img_width = df.loc[ind]['img_width']
    img_height = df.loc[ind]['img_height']
    randomForest = df.loc[ind]['Random Forest']
    vgg16 = df.loc[ind]['vgg16 Model']
    randomForest_vgg16 = df.loc[ind]['Random Forest vgg16']
    batch_size = df.loc[ind]['batch_size']
    epochs = df.loc[ind]['epochs']
    number_conv_layer = df.loc[ind]['number_conv_layer']
    number_conv_series = df.loc[ind]['number_conv_series']


    number_kernels = [int(s) for s in str(df.loc[ind]['number_kernels']).split() if s.isdigit()]


    kernel_size = (df.loc[ind]['kernel_size'],df.loc[ind]['kernel_size'])
    pool_size = (df.loc[ind]['pool_size'],df.loc[ind]['pool_size'])
    padding = df.loc[ind]['padding']
    number_hidden_layers = df.loc[ind]['number_hidden_layers']


    number_neurons = [int(s) for s in str(df.loc[ind]['number_neurons']).split() if s.isdigit()]
    
    dropout_hidden = df.loc[ind]['dropout_hidden']
    dropout_hidden_percentage = df.loc[ind]['dropout_hidden_percentage']
    dropout_conv = df.loc[ind]['dropout_conv']
    dropout_conv_percentage = df.loc[ind]['dropout_conv_percentage']
    augmentation = df.loc[ind]['augmentation']
    seed = df.loc[ind]['seed']
    mc_dropout = df.loc[ind]['mc_dropout']

    return img_width, img_height, randomForest, vgg16, randomForest_vgg16, batch_size, epochs, number_conv_layer, number_conv_series, number_kernels, kernel_size, pool_size, padding, number_hidden_layers, number_neurons, dropout_hidden, dropout_hidden_percentage, dropout_conv, dropout_conv_percentage, augmentation, seed, mc_dropout


###########################################################################
###### Transfer Learning - VGG16
###########################################################################

def transfer_learning_vgg16(number_neurons = [200, 100], img_width = 32, img_height=32, seed = 128):

    # load the pretrained vgg model
    print("1")
    base_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False,input_shape=(img_width,img_height,3))
    print("2")

    x = base_model.output
    x = Flatten()(x)
    x = Dense(number_neurons[0], activation='relu')(x)
    x = Dense(number_neurons[1], activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print("3")

    # freeze the weights of the convolutional part
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    for i, layer in enumerate(model.layers):
        print(i, layer.name,layer.trainable)
    print("4")


    foldername = "VGG16_" + str(number_neurons) + str(img_width) + str(img_height) + str(seed)

    return model, foldername


###########################################################################
###########################################################################
###########################################################################
###### Create CNN Model
###########################################################################
###########################################################################
###########################################################################

def create_cnn_model(img_width = 32, img_height = 32, number_conv_layer = 1, number_conv_series = 1, number_kernels = [16], kernel_size = (5,5), pool_size = (2,2), padding = 'same', 
                    number_hidden_layers = 0, number_neurons = [100], dropout_hidden = False, dropout_hidden_percentage = 0.3, dropout_conv = False, dropout_conv_percentage = 0.3, batch_size = 64, epochs = 40, augmentation = False, seed = 128, mc_dropout = False):

    tf.keras.backend.clear_session()

    from tensorflow.python.framework.ops import disable_eager_execution
    disable_eager_execution()

    ##########################################
    # create model
    ##########################################
    model = Sequential()

    # first layer. convolution
    model.add(Convolution2D(number_kernels[0],kernel_size,activation="relu",padding=padding,input_shape=(img_width,img_height,3)))
    if dropout_conv == True:
        model.add(Dropout(dropout_conv_percentage))


    # for j in range(1,number_conv_series+1):
    #     for i in range(1,number_conv_layer+1):
    #         if i < (number_conv_layer+1):
    #             if j == 1:
    #                 if number_conv_layer == 1:
    #                     continue
    #                 else:
    #                         model.add(Convolution2D(number_kernels[i-1],kernel_size,activation="relu",padding=padding))
    #                         if dropout_conv == True:
    #                             model.add(Dropout(dropout_conv_percentage))
    #             else:
    #                 if number_conv_layer == 1:
    #                     model.add(Convolution2D(number_kernels[i-1],kernel_size,activation="relu",padding=padding))
    #                     if dropout_conv == True:
    #                         model.add(Dropout(dropout_conv_percentage))
    #                 else:
    #                     model.add(Convolution2D(number_kernels[i-1],kernel_size,activation="relu",padding=padding))
    #                     if dropout_conv == True:
    #                         model.add(Dropout(dropout_conv_percentage))
    #                     model.add(Convolution2D(number_kernels[i-1],kernel_size,activation="relu",padding=padding))
    #                     if dropout_conv == True:
    #                         model.add(Dropout(dropout_conv_percentage))
    #     model.add(MaxPooling2D(pool_size=pool_size))

    for j in range(1,number_conv_series+1):
        if j == 1:
            if number_conv_layer == 1:
                model.add(MaxPooling2D(pool_size=pool_size))
            else:
                i = 1
                while i < number_conv_layer:
                    model.add(Convolution2D(number_kernels[j-1],kernel_size,activation="relu",padding=padding))
                    if dropout_conv == True:
                        model.add(Dropout(dropout_conv_percentage))
                    i += 1
                model.add(MaxPooling2D(pool_size=pool_size))
        else:
            if number_conv_layer == 1:
                model.add(Convolution2D(number_kernels[j-1],kernel_size,activation="relu",padding=padding))
                if dropout_conv == True:
                    model.add(Dropout(dropout_conv_percentage))
                model.add(MaxPooling2D(pool_size=pool_size))
            else:
                i = 0
                while i < number_conv_layer:
                    model.add(Convolution2D(number_kernels[j-1],kernel_size,activation="relu",padding=padding))
                    if dropout_conv == True:
                        model.add(Dropout(dropout_conv_percentage))
                    i += 1
                model.add(MaxPooling2D(pool_size=pool_size))
        


   

    model.add(Flatten())

    # Hidden Layers
    for h in range(0,number_hidden_layers):
        if dropout_hidden == True:
            model.add(Dropout(dropout_hidden_percentage))
        model.add(Dense(number_neurons[h]))
        model.add(Activation('relu'))

    # Ouput Layer     
    model.add(Dense(2))
    model.add(Activation('softmax'))

    # compile model and initialize weights
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    ##########################################
    # Parameter speichern & Foldername speichern
    ##########################################

    # Foldername
    if dropout_hidden == False and dropout_conv == False:

        if mc_dropout == True:
            foldername = "CNN_" + str(img_width) + str(img_height) + str(number_conv_layer) + str(number_conv_series) + str(number_kernels) + str(kernel_size) + str(pool_size) + padding + str(number_hidden_layers) + str(number_neurons) + str(batch_size) + str(epochs) + str(augmentation) + str(seed) + "MC"
        else:
            foldername = "CNN_" + str(img_width) + str(img_height) + str(number_conv_layer) + str(number_conv_series) + str(number_kernels) + str(kernel_size) + str(pool_size) + padding + str(number_hidden_layers) + str(number_neurons) + str(batch_size) + str(epochs) + str(augmentation) + str(seed)

        parameter_string = """CNN
        img_width = {},
        img_height = {},
        Number Conv Layer = {},
        Number Conv Series = {},
        Number Kernels = {},
        Kernel Size = {},
        Pool Size = {},
        Padding = {},
        Number Hidden Layers = {},
        Number Neurons = {}
        Dropout Hidden = {},
        Dropout Conv = {},
        Batch Size = {},
        Epochs = {},
        Augmentation = {}
        MC_Dropout = {}
        """.format(img_width, img_height, number_conv_layer, number_conv_series, number_kernels, kernel_size, pool_size, padding, 
                        number_hidden_layers, number_neurons, dropout_hidden, dropout_conv, batch_size, epochs, augmentation, mc_dropout)

    elif dropout_hidden == True and dropout_conv == False:

        if mc_dropout == True:
            foldername = "CNN_" + str(img_width) + str(img_height) + str(number_conv_layer) + str(number_conv_series) + str(number_kernels) + str(kernel_size) + str(pool_size) + padding + str(number_hidden_layers) + str(number_neurons) + str(dropout_hidden) + str(dropout_hidden_percentage) + str(batch_size) + str(epochs) + str(augmentation) + str(seed) + "MC"
        else:
            foldername = "CNN_" + str(img_width) + str(img_height) + str(number_conv_layer) + str(number_conv_series) + str(number_kernels) + str(kernel_size) + str(pool_size) + padding + str(number_hidden_layers) + str(number_neurons) + str(dropout_hidden) + str(dropout_hidden_percentage) + str(batch_size) + str(epochs) + str(augmentation) + str(seed)


        parameter_string = """CNN
        img_width = {},
        img_height = {},
        Number Conv Layer = {},
        Number Conv Series = {},
        Number Kernels = {},
        Kernel Size = {},
        Pool Size = {},
        Padding = {},
        Number Hidden Layers = {},
        Number Neurons = {}
        Dropout Hidden = {},
        Dropout Hidden % = {},
        Dropout Conv = {},
        Batch Size = {},
        Epochs = {},
        Augmentation = {}
        """.format(img_width, img_height, number_conv_layer, number_conv_series, number_kernels, kernel_size, pool_size, padding, 
                        number_hidden_layers, number_neurons, dropout_hidden, dropout_hidden_percentage, dropout_conv, batch_size, epochs, augmentation)

    elif dropout_hidden == False and dropout_conv == True:

        if mc_dropout == True:
            foldername = "CNN_" + str(img_width) + str(img_height) + str(number_conv_layer) + str(number_conv_series) + str(number_kernels) + str(kernel_size) + str(pool_size) + padding + str(number_hidden_layers) + str(number_neurons) + str(dropout_conv) + str(dropout_conv_percentage) + str(batch_size) + str(epochs)+ str(augmentation) + str(seed) + "MC"
        else:
            foldername = "CNN_" + str(img_width) + str(img_height) + str(number_conv_layer) + str(number_conv_series) + str(number_kernels) + str(kernel_size) + str(pool_size) + padding + str(number_hidden_layers) + str(number_neurons) + str(dropout_conv) + str(dropout_conv_percentage) + str(batch_size) + str(epochs)+ str(augmentation) + str(seed)


        parameter_string = """CNN
        img_width = {},
        img_height = {},
        Number Conv Layer = {},
        Number Conv Series = {},
        Number Kernels = {},
        Kernel Size = {},
        Pool Size = {},
        Padding = {},
        Number Hidden Layers = {},
        Number Neurons = {}
        Dropout Hidden = {},
        Dropout Conv = {},
        Dropout Conv % = {}
        Batch Size = {},
        Epochs = {},
        Augmentation = {}
        """.format(img_width, img_height, number_conv_layer, number_conv_series, number_kernels, kernel_size, pool_size, padding, 
                        number_hidden_layers, number_neurons, dropout_hidden, dropout_conv, dropout_conv_percentage, batch_size, epochs, augmentation)

    elif dropout_hidden == True and dropout_conv == True:

        if mc_dropout == True:
            foldername = "CNN_" + str(img_width) + str(img_height) + str(number_conv_layer) + str(number_conv_series) + str(number_kernels) + str(kernel_size) + str(pool_size) + padding + str(number_hidden_layers) + str(number_neurons) + str(dropout_hidden) + str(dropout_hidden_percentage) + str(dropout_conv) + str(dropout_conv_percentage) + str(batch_size) + str(epochs) + str(augmentation) + str(seed) + "MC"
        else:
            foldername = "CNN_" + str(img_width) + str(img_height) + str(number_conv_layer) + str(number_conv_series) + str(number_kernels) + str(kernel_size) + str(pool_size) + padding + str(number_hidden_layers) + str(number_neurons) + str(dropout_hidden) + str(dropout_hidden_percentage)


        parameter_string = """CNN
        img_width = {},
        img_height = {},
        Number Conv Layer = {},
        Number Conv Series = {},
        Number Kernels = {},
        Kernel Size = {},
        Pool Size = {},
        Padding = {},
        Number Hidden Layers = {},
        Number Neurons = {}
        Dropout Hidden = {},
        Dropout Hidden % = {},
        Dropout Conv = {},
        Dropout Conv % = {}
        Batch Size = {},
        Epochs = {},
        Augmentation = {}
        Seed = {}
        """.format(img_width, img_height, number_conv_layer, number_conv_series, number_kernels, kernel_size, pool_size, padding, 
                        number_hidden_layers, number_neurons, dropout_hidden, dropout_hidden_percentage, dropout_conv, dropout_conv_percentage, batch_size, epochs, augmentation, seed)


    save_model_parameters(base_dir="./Results/", folder_name=foldername, parameter_string=parameter_string)

    return model, batch_size, epochs, foldername


###########################################################################
###########################################################################
###########################################################################
###### Main Function
###########################################################################
###########################################################################
###########################################################################


def main():

    df = read_model_parameter_file(file_dir = "models.xlsx")

    base_dir = "./Results/"

    image_dir="./data/brain_tumor_dataset/"
    dir_yes =image_dir+'yes'
    dir_no = image_dir+'no'


    for n in range(0,len(df)):
        
        #####################
        # Model Parameter extrahieren
        #####################
        img_width, img_height, randomForest, vgg16, randomForest_vgg16, batch_size, epochs, number_conv_layer, number_conv_series, number_kernels, kernel_size, pool_size, padding, number_hidden_layers, number_neurons, dropout_hidden, dropout_hidden_percentage, dropout_conv, dropout_conv_percentage, augmentation, seed, mc_dropout = extract_parameters(df, n)


        #####################
        # Daten einlesen
        #####################
        X, y = load_data([dir_yes, dir_no], (img_width, img_height))

        #####################
        # Train, Val, Test-Split
        #####################
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.3, seed = seed)

        #####################
        # Class Weights
        #####################
        y_arr = []
        for i in range(0,len(y_train)):
            y_arr.append(y_train[i][0])

        # class weights berechnen mit compute_class_weight() von sklearn
        class_weights = class_weight.compute_class_weight('balanced',
                                                        np.unique(y_arr),
                                                        y_arr)
        class_weights = dict(enumerate(class_weights))

        #####################
        # Data split info
        #####################
        data_split_info = """ Data Split Informations

        Number of trainings examples = {},
        Number of validation examples = {},
        Number of test examples = {}
        Class weights = {}
        """.format(X_train.shape[0], X_val.shape[0], X_test.shape[0], class_weights)

        #####################
        # One hot encoding
        #####################
        y_train=to_categorical(y_train,2) # one-hot encoding
        y_val=to_categorical(y_val,2) # one-hot encoding
        y_test_2=to_categorical(y_test,2) # one-hot encoding

        if randomForest == False and vgg16 == False and randomForest_vgg16 == False:

            #####################
            # CNN Model
            #####################
            model, batch_size, epochs, foldername = create_cnn_model(img_width = img_width, img_height = img_height, number_conv_layer = number_conv_layer, number_conv_series = number_conv_series, number_kernels = number_kernels, kernel_size = kernel_size, pool_size = pool_size, padding = padding, 
                        number_hidden_layers = number_hidden_layers, number_neurons = number_neurons, dropout_hidden = dropout_hidden, dropout_hidden_percentage = dropout_hidden_percentage, dropout_conv = dropout_conv, dropout_conv_percentage = dropout_conv_percentage, batch_size = batch_size, epochs = epochs, augmentation = augmentation, seed = seed, mc_dropout=mc_dropout)


            print("""
            ################################
            ################################
            CURRENTL RUNNING MODEL : {}
            ################################
            ################################
            """.format(foldername))

            #####################
            # Modell Info speichern
            #####################
            save_model_summary(base_dir=base_dir, folder_name=foldername, model=model)

            #####################
            # Data Split Info speichern 
            #####################
            save_data_split_infos(base_dir="./Results/", folder_name=foldername,split_info_string=data_split_info)

            #####################
            # Train Model
            #####################
            history = train_model(model, X_train, y_train, X_val, y_val, class_weights, batch_size, epochs, augmentation = augmentation)

            #####################
            # Plot & Save Accuracy- und Loss-Curve
            #####################
            plot_save_accuracy_loss_curve(history, foldername)

            #####################
            # Prediction & Scores
            #####################
            pred, pred_prob = cnn_prediction(model, X_test, mc_dropout = mc_dropout)
            scores_res(model, pred, pred_prob,X_test, y_test_2, y_test, 'cnn', foldername, base_dir, foldername)

            #####################
            # Confusion Matrix
            #####################
            plot_save_confusion_matrix(base_dir, foldername, y_test, pred)

            #####################
            # Images - False Prediction
            #####################
            plot_images_wrong_prediction(X_test, y_test, pred, foldername)

        elif randomForest == True and vgg16 == False and randomForest_vgg16 == False:
            
            #####################
            # RF Model
            #####################
            clf, foldername = random_forest(X_train, y_train, n_trees=40, img_width = img_width, img_height=img_height, seed = seed)

            print("""
            ################################
            ################################
            CURRENTL RUNNING MODEL : {}
            ################################
            ################################
            """.format(foldername))

            #####################
            # Data Split Info speichern 
            #####################
            save_data_split_infos(base_dir="./Results/", folder_name=foldername,split_info_string=data_split_info)

            #####################
            # Prediction & Scores
            #####################
            pred, pred_prob = rf_prediction(clf, X_test, img_width = img_width, img_height=img_height)
            scores_res(clf, pred, pred_prob,X_test, y_test_2, y_test, 'RF', foldername, base_dir, foldername)

            #####################
            # Confusion Matrix
            #####################
            plot_save_confusion_matrix(base_dir, foldername, y_test, pred)

            #####################
            # Images - False Prediction
            #####################
            plot_images_wrong_prediction(X_test, y_test, pred, foldername)

        elif randomForest == False and vgg16 == True and randomForest_vgg16 == False:

            #####################
            # VGG16
            #####################
            print(number_neurons)
            model, foldername = transfer_learning_vgg16(number_neurons = number_neurons, img_width = img_width, img_height=img_height, seed = seed)


            print("""
            ################################
            ################################
            CURRENTL RUNNING MODEL : {}
            ################################
            ################################
            """.format(foldername))


            #####################
            # Modell Info speichern
            #####################
            save_model_summary(base_dir=base_dir, folder_name=foldername, model=model)

            #####################
            # Data Split Info speichern 
            #####################
            save_data_split_infos(base_dir="./Results/", folder_name=foldername,split_info_string=data_split_info)

            #####################
            # Train Model
            #####################
            history = train_model(model, X_train, y_train, X_val, y_val, class_weights, batch_size, epochs, augmentation = augmentation)

            #####################
            # Plot & Save Accuracy- und Loss-Curve
            #####################
            plot_save_accuracy_loss_curve(history, foldername)

            #####################
            # Prediction & Scores
            #####################
            pred, pred_prob = cnn_prediction(model, X_test)
            scores_res(model, pred, pred_prob,X_test, y_test_2, y_test, 'cnn', foldername, base_dir, foldername)

            #####################
            # Confusion Matrix
            #####################
            plot_save_confusion_matrix(base_dir, foldername, y_test, pred)

            #####################
            # Images - False Prediction
            #####################
            plot_images_wrong_prediction(X_test, y_test, pred, foldername)


        
        elif randomForest == False and vgg16 == False and randomForest_vgg16 == True:

            #####################
            # RF VGG16 Model
            #####################        
            clf, X_test_vgg_features, foldername = random_forest_vgg16(X_train, y_train, X_test, n_trees=40, img_width = img_width, img_height=img_height, seed = seed)

            print("""
            ################################
            ################################
            CURRENTL RUNNING MODEL : {}
            ################################
            ################################
            """.format(foldername))

            #####################
            # Data Split Info speichern 
            #####################
            save_data_split_infos(base_dir="./Results/", folder_name=foldername,split_info_string=data_split_info)

            #####################
            # Prediction & Scores
            #####################
            pred, pred_prob = rf_prediction_vgg16(clf, X_test_vgg_features)
            scores_res(clf, pred, pred_prob,X_test, y_test_2, y_test, 'RF', foldername, base_dir, foldername)

            #####################
            # Confusion Matrix
            #####################
            plot_save_confusion_matrix(base_dir, foldername, y_test, pred)

            #####################
            # Images - False Prediction
            #####################
            plot_images_wrong_prediction(X_test, y_test, pred, foldername)


