import argparse
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers
from squeezenet import SqueezeNet
import os

SIZE = 227

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', default='data/train')
    parser.add_argument('--test-dir', default='data/test')
    parser.add_argument('--logdir', default='logs')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument(
        '--checkpoint-pattern',
        default='weights-{epoch:d}-{val_acc:.4f}.hdf5'
    )
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--restore')
    args = parser.parse_args()

    # count samples
    train_files = count_files(args.train_dir, '.png')
    print('Found %d train files.' % train_files)
    test_files = count_files(args.test_dir, '.png')
    print('Found %d test files.' % test_files)

    if args.restore:
        model = SqueezeNet(weights=None, classes=args.num_classes)
        model.load_weights(args.restore)
    else:
        model = SqueezeNet(weights='imagenet', classes=args.num_classes)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(lr=args.learning_rate),
        metrics=['accuracy']
    )

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_single
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_single
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(SIZE, SIZE),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        args.test_dir,
        target_size=(SIZE, SIZE),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    checkpoint = ModelCheckpoint(
        args.checkpoint_pattern,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    tensorboard = TensorBoard(
        log_dir=args.logdir,
        histogram_freq=0,
        batch_size=args.batch_size,
        write_graph=True,
        write_grads=False,
        write_images=False,
        embeddings_freq=0,
        embeddings_layer_names=None,
        embeddings_metadata=None
    )
    callbacks = [checkpoint, tensorboard]

    model.fit_generator(
        train_generator,
        steps_per_epoch=(train_files // args.batch_size),
        epochs=args.epochs,
        validation_data=test_generator,
        validation_steps=(test_files // args.batch_size),
        callbacks=callbacks
    )

def count_files(directory, extension):
    count = 0
    for _, _, files in os.walk(directory):
        for f in files:
            if f.endswith(extension):
                count += 1
    return count

def preprocess_single(img):
    return preprocess_input(np.array([img]))[0]

if __name__ == '__main__':
    main()
