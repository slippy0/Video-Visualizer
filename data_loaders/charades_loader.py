import torchfile
import numpy as np
import os
import pdb

from data_loader import DataLoader

class CharadesLoader(DataLoader):
    # Static variables
    FRAME_RATE = 24 # in FPS
    SAMPLE_RATE = 10 # one prediction per N frames
    _TIME_PER_SAMPLE = SAMPLE_RATE / float(FRAME_RATE)

    def __init__(self, videos_dir, prediction_dir, class_list):
        """
        videos_dir     - Path to directory of videos.
        prediction_dir - Path to directory of .t7 predictions, one per video.
        """
        print "Init Charades"
        self.videos_dir = videos_dir
        self.prediction_dir = prediction_dir
        with open(class_list) as f:
            self.class_list = [line.strip() for line in f.readlines()]
        self._video_list = None

    def get_videos_dir(self):
        return self.videos_dir

    def video_list(self):
        """
        Returns:
            videos (list): List containing video names as strings.
        """
        # Lazy load video list
        if self._video_list is None:
            #paths = [os.path.join(self.videos_dir, f) for f in os.listdir(self.videos_dir)]
            #self._video_list = [path for path in paths if os.path.isfile(path)]
            self._video_list = [f for f in os.listdir(self.videos_dir) if os.path.isfile(os.path.join(self.videos_dir, f))]

        return self._video_list

    def video_groundtruth(self, video_name):
        """
        Args:
            video_name (str): One video name from the list returned by
                video_list.

        Returns:
            groundtruth (list): Each element is a tuple of the form
                (start_sec, end_sec, label)
        """
        # Get full filepath
        video_name = os.path.splitext(video_name)[0]
        file_path = os.path.join(self.prediction_dir, video_name + ".t7")

        # Load the data
        data = torchfile.load(file_path)
        gt_labels = data[1] # Hardcoding this, oops

        # Shape the data
        num_frames = gt_labels.shape[0]

        groundtruth = []
        for frame in xrange(num_frames):
            for class_idx, class_name in enumerate(self.class_list):
                if gt_labels[frame, class_idx]:
                    groundtruth.append(
                            (frame*CharadesLoader._TIME_PER_SAMPLE,
                            (frame+1)*CharadesLoader._TIME_PER_SAMPLE,
                            class_name))
        return groundtruth

    def video_predictions(self, video_name):
        """
        Args:
            video_name (str): One video name from the list returned by
                video_list.

        Returns:
            predictions (dict): Maps label name to list of floats representing
                confidences. The list spans the length of the video.
        """
        # Get full filepath
        video_name = os.path.splitext(video_name)[0]
        file_path = os.path.join(self.prediction_dir, video_name + ".t7")

        # Load the data
        data = torchfile.load(file_path)
        pred_raw = data[0] # Hardcoding this, oops
        pred_raw = np.maximum(pred_raw, 0) # ReLU
        pred_raw = np.tanh(pred_raw)

        # Shape the data
        num_frames = pred_raw.shape[0]
        num_classes = pred_raw.shape[1]

        # Put into required format
        predictions = {}
        for class_idx, class_name in enumerate(self.class_list):
            # Repeat by SAMPLE_RATE to have one label per frame
            predictions[class_name] = pred_raw[:,class_idx] \
                    .flatten() \
                    .repeat(CharadesLoader.SAMPLE_RATE) \
                    .tolist()

        return predictions
