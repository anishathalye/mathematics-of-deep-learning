# Transfer learning

We're going to train a [SqueezeNet](https://arxiv.org/abs/1602.07360) to solve
some new k-way classification task using transfer learning.

Starting with a SqueezeNet trained to classify ImageNet images, we're going to
strip the last fully-connected layer and replace it with a new layer with k
neurons. Then, we'll fine-tune the network on our new dataset.

## Collecting data

We want to collect a dataset with images of all our classes (and split them
into a train and test set). Our training code expects data in the following format:

```
data/
|
+- train/
|  |
|  +- foo/
|  |  - filename.png
|  |  - someotherfilename.png
|  |
|  +- bar/
|     - examplebar.png
|     - morebar.png
|
+- test/
   |
   +- foo/
   |  - 1.png
   |  - asdf.png
   |
   +- bar/
      - file.png
      - anotherfile.png
```

Within the `data` directory, you should have a `train` and `test` directory,
and within each of those, have a directory for each of your class labels.
Within each of those, you should have a bunch of `.png` files (the filenames
themselves don't matter).

You can either collect images manually and split them into a train and test
set, or you can record videos (recommended) and use a script we've written to
automatically extract frames and split them into a train and test set.

If you're recording videos, e.g. using your phone or webcam, place each video
(in `.mov` or `.mp4` format) into a `datasource/{label}` folder (the actual
file names of the videos don't matter), and then run the following script:

```bash
python datagen.py
```

The script will automatically split each video clip into a train set and a test
set (with a 90-10 split). If you'd like to make a certain video only be used in
the test set (it's highly recommended to have at least one test-only video for
each class), make the video's filename start with "val". Similarly, if you want
a video clip only used for the test set, make the filename start with "test".

The script will place images in the appropriate format into the `data` folder.

It's suggested that you use a small number of classes, around 2-4. You also
don't need to collect _that_ much data: a couple short videos for each class
should suffice. Also, try to get a diverse set of videos. If you're
classifying, say, facial expressions, try to get videos from a couple different
people.

## Training the network

You can train the network using:

```bash
python train.py --epochs {number of epochs} --num-classes {number of classes}
```

Start off by training for a small number of epochs, e.g. 5. This is the number
of passes over the training dataset before training will stop.

As the network is training, it will print messages like this:

```
Epoch 1/5
15/15 [==============================] - 21s 1s/step - loss: 0.7226 - acc: 0.6193 - val_loss: 0.5433 - val_acc: 0.6250
```

After each epoch, it shows the loss (over the training data), accuracy (over
the training data), validation loss (over the test data), and validation
accuracy (over the test data). The validation accuracy is the metric you care
about -- it shows how well the network generalizes to new data points.

As the network is training, it saves the network weights to files with the name
`weights-{epoch}-{val_acc}.h5py`. You can restore from this file if you want to
resume training from a checkpoint. You will also need the parameters from this
file to evaluate your network.

## Evaluating the network

One way you can evaluate your network is by computing the [confusion
matrix](https://en.wikipedia.org/wiki/Confusion_matrix) over your test set. As
the name indicates, this makes it easy to see if your classifier is confusing
instances of two different classes. You can compute this using the following:

```bash
python confusion.py --checkpoint-path {weights file} --num-classes {number of classes}
```

The script will output a visualization in `confusion_matrix.png`.

## Classifying new data

You can classify a new image with:

```bash
python predict.py --checkpoint-path {weights file} --num-classes {number of classes} --image {path to image}
```

The script will output the prediction label (a zero-indexed number i,
corresponding to the ith class in alphabetical order).

## Understanding what's going on

What's happening conceptually is pretty simple. If you want to understand the
details of how to implement this stuff, read through the source code.

## Exploration

Once you've successfully trained a SqueezeNet to solve a custom classification
task, you can try exploring new things. Some ideas:

* Solve more sophisticated tasks, perhaps with more class labels, and see how
  the difficulty of training the network changes

* Train for a large number of epochs (especially if you have access to a GPU),
  and try training to convergence

* Experiment with different learning rates to see how they affect convergence
  (use the `--learning-rate` argument to `train.py`)
