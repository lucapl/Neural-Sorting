import tensorflow as tf
from tensorflow.keras.metrics import Metric, Precision, Recall, AUC,Accuracy
from tensorflow.keras import backend as K


class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + K.epsilon())) # epsilon to prevent zeroes

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


class UtilMetric(Metric):
    def __init__(self,metric, **kwargs):
        super(UtilMetric, self).__init__(name=metric.name, **kwargs)
        self.metric = metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.metric.update_state(y_true, tf.cast(y_pred>0,tf.float32), sample_weight)

    def result(self):
        return self.metric.result()

    def reset_states(self):
        self.metric.reset_states()