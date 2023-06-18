from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, TimeDistributed, Flatten
import pandas as pd
import cv2
import numpy as np
import logging



logging.basicConfig(level=logging.INFO)

"""
Load the TSV file, read image paths, and labels.
Returns:
    images (numpy.ndarray): Array of resized images.
    labels (pandas.Series): Series of labels.
"""
def load_data():
    try:
        with open('gbn-fonts.tsv', 'r') as file:
            data = pd.read_csv(file, sep='\t')

        image_paths = "dataset/Fraktur/" + data['segment_id'] + ".png"
        labels = data['text_equiv']

        images = []
        for path in image_paths:
            try:
                image = cv2.imread(path, 0)
                if image is None:
                    logging.warning(f"Failed to read image: {path}")
                    continue

                # Perform image dilation (expanding) and erosion
                kernel = np.ones((3, 3), np.uint8)
                dilated_image = cv2.dilate(image, kernel, iterations=1)
                eroded_image = cv2.erode(image, kernel, iterations=1)

                # Resize the images if needed
                dilated_image = cv2.resize(dilated_image, (224, 224))
                eroded_image = cv2.resize(eroded_image, (224, 224))

                images.extend([dilated_image, eroded_image])
            except Exception as e:
                logging.warning(f"Error processing image: {path}, {str(e)}")

        images = np.array(images, dtype="float32")
        return images, labels

    except FileNotFoundError:
        logging.error("TSV file not found.")
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")

    return None, None

def divide_dataset(images, labels, test_size=0.1, random_state=42):
    try:
        train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=test_size, random_state=random_state)
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=test_size, random_state=random_state)

        logging.info("Dataset divided successfully.")
        return train_data, train_labels, val_data, val_labels, test_data, test_labels

    except ValueError as e:
        logging.error(f"ValueError: {str(e)}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

    return None, None, None, None, None, None

# SO uma ideia de arquitetura, nao funciona!!!
def train_neural_network():
    pass
    # model = Sequential()

    # # CNN layers
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(train_data.shape[1], train_data.shape[2], 1)))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Flatten())

    # # LSTM layer
    # model.add(TimeDistributed(Dense(64, activation='relu')))
    # model.add(LSTM(64, return_sequences=False))

    # # Output layer
    # model.add(Dense(num_classes, activation='softmax'))

    # # Compile the model
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # # Print the model summary
    # model.summary()


# Call the function to load the data
images, labels = load_data()

train_data, train_labels, val_data, val_labels, test_data, test_labels = divide_dataset(images, labels)

