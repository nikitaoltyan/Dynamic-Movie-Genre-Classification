import tensorflow as tf
import os
from keras.models import load_model
from process_data import *

data_dir = '../data/raw/movie_classification/val'
report_path = '../output'
model_path = '../model/trained_model.h5'


def evaluate():
    print('evaluating...')
    test_generator, data_shape = prepare_test_data()

    print('Loading model...')
    model = load_model(model_path)
    print('Done')

    metrics = [
        tf.metrics.BinaryAccuracy(),
        tf.metrics.Precision(),
        tf.metrics.Recall()
    ]
    model.compile(metrics=metrics)

    _, acc, precision, recall, auc = model.evaluate(test_generator)

    result_lines = [f'Test data len: {len(test_generator)}\n'
                    f'Accuracy: {acc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, AUC: {auc: .3f}']

    if not os.path.exists(report_path):
        os.makedirs(report_path)
    with open(f'{report_path}/report.txt', 'w', encoding='utf-8') as f:
        f.writelines(result_lines)


if __name__  == '__main__':
    evaluate()