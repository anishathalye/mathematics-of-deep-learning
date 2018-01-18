import argparse
import numpy as np
from squeezenet import SqueezeNet
import os
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

SIZE = 227

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--image', nargs='+', required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    args = parser.parse_args()

    model = SqueezeNet(weights=None, classes=args.num_classes)
    model.load_weights(args.checkpoint_path)

    xs = []
    for path in args.image:
        img = image.load_img(path, target_size=(SIZE, SIZE))
        x = image.img_to_array(img)
        xs.append(x)

    xs = np.array(xs)
    xs = preprocess_input(xs)

    probs = model.predict(xs)

    print('')
    for i, path in enumerate(args.image):
        print('%s' % path)
        print('    Prediction: %s' % np.argmax(probs[i]))


if __name__ == '__main__':
    main()
