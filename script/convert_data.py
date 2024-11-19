import tensorflow as tf
from numpy import savez_compressed
import os

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def convert_to_np(batch_size, img_w, img_h, train_dir_src, val_dir_src, test_dir_src, train_dir_dest, val_dir_dest, test_dir_dest):
    # Creazione delle cartelle di destinazione se non esistono
    create_dir_if_not_exists(train_dir_dest)
    create_dir_if_not_exists(val_dir_dest)
    create_dir_if_not_exists(test_dir_dest)

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir_src,
        image_size=(img_w, img_h),
        batch_size=batch_size,
        label_mode='int',
        shuffle=True
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
         val_dir_src,
         image_size=(img_w, img_h),
         batch_size=batch_size,
         label_mode='int',
         shuffle=True
    )

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir_src,
        image_size=(img_w, img_h),
        batch_size=batch_size,
        label_mode='int',
        shuffle=True
    )

    class_names = train_dataset.class_names
    print("Classi:", class_names)

    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    # Estrae immagini e etichette dal dataset di training
    for image_batch, labels_batch in train_dataset:
        x_train.append(image_batch.numpy())
        y_train.append(labels_batch.numpy())

    # Estrae immagini e etichette dal dataset di validation
    for image_batch, labels_batch in val_dataset:
         x_val.append(image_batch.numpy())
         y_val.append(labels_batch.numpy())

    # Estrae immagini e etichette dal dataset di test
    for image_batch, labels_batch in test_dataset:
        x_test.append(image_batch.numpy())
        y_test.append(labels_batch.numpy())

    # Concatenazione di tutti i batch in un unico array
    x_train = tf.concat(x_train, axis=0)
    y_train = tf.concat(y_train, axis=0)
    x_val = tf.concat(x_val, axis=0)
    y_val = tf.concat(y_val, axis=0)
    x_test = tf.concat(x_test, axis=0)
    y_test = tf.concat(y_test, axis=0)

    # Normalizzazione dei dati nell'intervallo [0, 1]
    x_train_normalized = x_train.numpy().astype('float32') / 255.
    x_val_normalized = x_val.numpy().astype('float32') / 255.
    x_test_normalized = x_test.numpy().astype('float32') / 255.

    # Centra e normalizza ulteriormente nell'intervallo [-1, 1]
    x_train = x_train_normalized * 2. - 1.
    x_val = x_val_normalized * 2. - 1.
    x_test = x_test_normalized * 2. - 1.

    # Salva i dati nei file .npz
    savez_compressed(os.path.join(train_dir_dest, f'x_train_{img_w}.npz'), x_train)
    savez_compressed(os.path.join(train_dir_dest, f'y_train_{img_w}.npz'), y_train)
    savez_compressed(os.path.join(val_dir_dest, f'x_val_{img_w}.npz'), x_val)
    savez_compressed(os.path.join(val_dir_dest, f'y_val_{img_w}.npz'), y_val)
    savez_compressed(os.path.join(test_dir_dest, f'x_test_{img_w}.npz'), x_test)
    savez_compressed(os.path.join(test_dir_dest, f'y_test_{img_w}.npz'), y_test)

    print("Dimensione di x_train:", x_train.shape)  # Dimensione delle immagini di training
    print("Dimensione di y_train:", y_train.shape)  # Dimensione delle etichette di training
    print("Dimensione di x_val:", x_val.shape)      # Dimensione delle immagini di validation
    print("Dimensione di y_val:", y_val.shape)      # Dimensione delle etichette di validation
    print("Dimensione di x_test:", x_test.shape)    # Dimensione delle immagini di test
    print("Dimensione di y_test:", y_test.shape)    # Dimensione delle etichette di test

# Parametri di configurazione
batch_size = 32
img_w = 64
img_h = 64

train_dir_dest = './data/data_64_conv/train_data_converted'
val_dir_dest = './data/data_64_conv/val_data_converted'
test_dir_dest = './data/data_64_conv/test_data_converted'

train_dir_src = './data/train_aug'
val_dir_src = './data/valid'
test_dir_src = './data/test'

convert_to_np(batch_size, img_w, img_h, train_dir_src, val_dir_src, test_dir_src, train_dir_dest, val_dir_dest, test_dir_dest)
