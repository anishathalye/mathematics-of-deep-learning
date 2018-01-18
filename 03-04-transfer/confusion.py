import argparse
import numpy as np
from squeezenet import SqueezeNet
import os
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import time
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIZE = 227

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--test-dir', default='data/test')
    parser.add_argument('--output-file', default='confusion_matrix.png')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--num-classes', type=int, required=True)
    args = parser.parse_args()

    model = SqueezeNet(weights=None, classes=args.num_classes)
    model.load_weights(args.checkpoint_path)

    data = []
    classes = sorted(os.listdir(args.test_dir))
    if len(classes) != args.num_classes:
        raise ValueError('expecting %d classes, found %d in %s' % (
            args.num_classes,
            len(classes),
            args.test_dir
        ))

    for ic in range(len(classes)):
        directory = os.path.join(args.test_dir, classes[ic])
        for path in os.listdir(directory):
            full_path = os.path.join(directory, path)
            data.append((full_path, ic))

    rng = random.Random(0)
    rng.shuffle(data)
    if args.limit > 0:
        data = data[:args.limit]

    chunked = list(chunks(data, args.batch_size))
    gstart = time.time()
    cmat = np.zeros((len(classes), len(classes)), dtype=np.int)
    last_print = 0
    for i, chunk in enumerate(chunked):
        start = time.time()
        paths, ys = zip(*chunk)
        xs = []
        for path in paths:
            img = image.load_img(path, target_size=(SIZE, SIZE))
            x = image.img_to_array(img)
            xs.append(x)
        xs = np.array(xs)
        xs = preprocess_input(xs)

        probs = model.predict(xs, batch_size=args.batch_size)
        preds = probs.argmax(axis=1)
        for actual, predicted in zip(ys, preds):
            cmat[actual][predicted] += 1

        diff = time.time() - start
        gdiff = time.time() - gstart
        if time.time() - last_print > 1 or i == len(chunked)-1:
            last_print = time.time()
            print('batch %d/%d (in %.3fs, %.1fs elapsed, %.1fs remaining)' % (
                i+1,
                len(chunked),
                time.time() - start,
                gdiff,
                gdiff / (i+1) * (len(chunked)-i-1)
            ))

    print(cmat)
    plot_cmat(cmat, classes, args.output_file)
    print('saved figure to %s' % args.output_file)


def plot_cmat(conf_arr, classes, output_file):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.RdYlGn,
                    interpolation='nearest')

    for x in range(len(classes)):
        for y in range(len(classes)):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.xticks(range(len(classes)), classes)
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('predicted class')
    plt.ylabel('actual class')
    plt.title('confusion matrix')
    plt.savefig(output_file)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


if __name__ == '__main__':
    main()
