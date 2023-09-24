# Nikita Oltyan
import click
import os
from model import *
from process_data import *
# from visualize import *
from model import make_model

@click.command()
@click.option('-b', '--batch_size', type=int, default=32, help='Batch size of data')
@click.option('-e', '--epochs', type=int, default=5, help='Epochs of training')
@click.option('-v', '--val_split', type=float, default=0.1, help='Split of data for validation during training')
@click.option('-vb', '--verbose', type=int, default=0, help='Visualization flag for training')
@click.option('-h', '--history', type=bool, default=False, help='Training history flag for visualizing metrics')
def train(batch_size, epochs, val_split, verbose, history):
    print('training...')
    train_generator, val_generator, data_shape = prepare_train_data(BATCH_SIZE=batch_size, VAL_SPLIT=val_split)

    model = make_model(data_shape)

    print(model.summary())

    # Fitting
    train_history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        verbose=verbose)

    # Define number for correct run save
    if not os.path.exists('../runs'):
        os.makedirs('../runs')
    number_of_runs = len(os.listdir('../runs'))
    if not os.path.exists(f'../runs/run_{number_of_runs}'):
        os.makedirs(f'../runs/run_{number_of_runs}')

    # Save history data and visualize if nesesary
    # plot_training_data(train_history, number_of_runs, history)

    # Save the trained model
    model.save(f'../runs/run_{number_of_runs}/trained_model.h5')


if __name__ == '__main__':
    train()