import tensorflow as tf
import numpy as np

from hausdorff.hausdorff import cdist, weighted_hausdorff_distance


def test_cdist():
    # GIVEN we have 2 identical matrices A and B with two pairs, (1, 2) and (3, 4)
    data = [[1, 2], [3, 4]]
    A = tf.convert_to_tensor(data, dtype=tf.float32)
    B = tf.convert_to_tensor(data, dtype=tf.float32)
    # WHEN we run compute the pairwise distance
    d = cdist(A, B)
    # THEN we expect the result to be a diagonal matrix with 0s along the right diagonal and sqrt(8) along the left
    assert (np.array(d) == [[0.0, 2.8284270763397217],
                            [2.8284270763397217, 0.0]]).all()


def test_hausdorff_loss_match():
    y_true = np.zeros((5, 5)).astype('float32')
    y_true[1, 1] = 1
    y_pred = np.zeros((5, 5)).astype('float32')
    y_pred[1, 1] = 1
    results = weighted_hausdorff_distance(5, 5, 0)(np.expand_dims(y_true, 0), np.expand_dims(y_pred, 0))
    assert int(results) == 0.0


def test_hausdorff_loss_diff():
    y_true = np.zeros((5, 5)).astype('float32')
    y_true[0, 0] = 1
    y_pred = np.zeros((5, 5)).astype('float32')
    y_pred[4, 4] = 1
    results = weighted_hausdorff_distance(5, 5, 0)(np.expand_dims(y_true, 0), np.expand_dims(y_pred, 0))
    assert int(results) == 5.0
