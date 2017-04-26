from data_loader import DataLoader

class DummyLoader(DataLoader):
    def __init__(seslf, *args, **kwargs):
        print "Init DummyLoader"
        pass

    def video_list(self):
        """
        Returns:
            videos (list): List containing video names as strings.
        """
        pass

    def video_groundtruth(self, video_name):
        """
        Args:
            video_name (str): One video name from the list returned by
                video_list.

        Returns:
            groundtruth (list): Each element is a tuple of the form
                (start_sec, end_sec, label)
        """
        pass

    def video_predictions(self, video_name):
        """
        Args:
            video_name (str): One video name from the list returned by
                video_list.

        Returns:
            predictions (dict): Maps label name to list of floats representing
                confidences. The list spans the length of the video.
        """
        pass