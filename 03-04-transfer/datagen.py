# generate data from video files
#
# this is supposed to make the train-test split deterministic
# it should be deterministic based on the filename of the video

import argparse
import os
import subprocess
import hashlib
import random

TEST_FRACTION = 0.1
VIDEO_EXTENSIONS = ['.mov', '.mp4']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', default='datasource')
    parser.add_argument('--target-dir', default='data')
    args = parser.parse_args()


    class_dirs = [
        i for i in os.listdir(args.source_dir)
        if os.path.isdir(os.path.join(args.source_dir, i))
    ]
    out_paths = [
        os.path.join(args.target_dir, split, i)
        for split in ['train', 'test'] for i in class_dirs
    ]
    for path in out_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    ic = 0
    for c in class_dirs:
        ic += 1
        path = os.path.join(args.source_dir, c)
        videos = [i for i in os.listdir(path) if any(
            i.endswith(ext) for ext in VIDEO_EXTENSIONS
        )]
        iv = 0
        for video in videos:
            iv += 1
            video_path = os.path.join(path, video)
            output_base = os.path.splitext(video_path)[0]
            output_pattern = output_base + '-%06d.png'
            with open(os.devnull) as devnull:
                subprocess.check_call(
                    [
                        'ffmpeg',
                        '-i',
                        video_path,
                        '-vf',
                        'scale=320:-1',
                        output_pattern
                    ],
                    stdout=devnull,
                    stderr=devnull
                )
            frames = [
                i for i in os.listdir(path)
                if i.startswith(os.path.basename(output_base)) and i.endswith('.png')
            ]
            rng = random.Random(hashlib.md5(video.encode('utf8')).digest())
            rng.shuffle(frames)

            if video.startswith('val'):
                split_idx = len(frames) # everything is validation
            elif video.startswith('train'):
                split_idx = 0 # everything is train
            else:
                split_idx = int(len(frames) * TEST_FRACTION)
            test = frames[:split_idx]
            train = frames[split_idx:]
            move_all(path, test, os.path.join(args.target_dir, 'test', c))
            move_all(path, train, os.path.join(args.target_dir, 'train', c))
            print('[c: %d/%d, v: %d/%d] %s:%s - %d test, %d train' %
                    (ic, len(class_dirs), iv, len(videos), c, video, len(test), len(train)))


def move_all(prefix, files, directory):
    for f in files:
        source = os.path.join(prefix, f)
        dest = os.path.join(directory, f)
        os.rename(source, dest)


if __name__ == '__main__':
    main()
