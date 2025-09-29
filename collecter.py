from river import metrics
import numpy as np




class WindowClassificationPerformanceEvaluator():
    """Evaluator for tracking classification performance in a window-wise manner.

    This class is designed to evaluate a classification model's performance in a window-wise
    fashion. It uses a specified metric to measure the performance and maintains a list of scores
    calculated at the end of each window.

    Parameters:
    - metric: metrics.base.MultiClassMetric, optional (default=None)
        The metric used to evaluate the model's predictions. If None, the default metric is
        metrics.Accuracy().
    - window_width: int, optional (default=1000)
        The width of the evaluation window, i.e., the number of samples after which the metric is
        calculated and the window is reset.
    - print_every: int, optional (default=1000)
        The interval at which the current metric value is printed to the console.

    Methods:
    - update(y_pred, y, sample_weight=1.0):
        Update the evaluator with the predicted and true labels for a new sample. The metric is
        updated, and if the window is complete, the metric value is added to the scores list.
    - get():
        Get the list of metric scores calculated at the end of each window.

    Example:
    >>> evaluator = WindowClassificationPerformanceEvaluator(
    ...     metric=metrics.Accuracy(),
    ...     window_width=500,
    ...     print_every=500
    ... )
    >>> for x, y in stream:
    ...     y_pred = model.predict(x)
    ...     evaluator.update(y_pred, y)
    ...
    >>> scores = evaluator.get()
    >>> print(scores)

    Note: This class assumes a multi-class classification scenario and is designed to work with
    metrics that inherit from metrics.base.MultiClassMetric.
    """
    def __init__(self, metric=None, window_width=1000, print_every=1000):
        self.window_width = window_width
        self.metric = metric if metric is not None else metrics.Accuracy()
        self.print_every = print_every
        self.counter = 0
        self.scores_list = []
    
    def __repr__(self):
        """Return the class name along with the current value of the metric."""
        metric_value = np.mean(self.get()) * 100
        return f"{self.__class__.__name__}({self.metric.__class__.__name__}): {metric_value:.2f}%"


    def update(self, y_pred, y, sample_weight=1.0):
        """Update the evaluator with new predictions and true labels.

        Parameters:
        - y_pred: Predicted label for the current sample.
        - y: True label for the current sample.
        - sample_weight: Weight assigned to the current sample (default=1.0).
        """
        self.metric.update(y_pred, y, sample_weight=sample_weight)
        self.counter += 1

        # if self.counter % self.print_every == 0:
        #     print(f"[{self.counter}] - {self.metric}")

        if self.counter % self.window_width == 0:
            self.scores_list.append(self.metric.get())
            self.metric = type(self.metric)()  # resetting using the same metric type

    def get(self):
        """Get the list of metric scores calculated at the end of each window."""
        return self.scores_list