'''
General methods used by the models classes in models_tf.py
'''

import glob # Unix style pathname pattern expansion
import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


## augmentation pipeline

def augmentor(inputs):
    aug = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            layers.experimental.preprocessing.RandomRotation(0.02)],
            name='Augmentation')
    return aug(inputs)

## other utilities

def leraningRateScheduler(lr_start, current_epoch, max_epochs, power):
    '''
    Implements a polynimial leraning rate decay based on the:
    - lr_start: initial learning rate
    - current_epoch: current epoch
    - max_epochs: max epochs
    - power: polynomial power
    '''

    decay = (1 - (current_epoch / float(max_epochs))) ** power
    return lr_start * decay


def accuracy(y_true, y_pred):
    '''
    Computes the accuracy number of correct classified / total number of samples
    '''
    count = tf.reduce_sum(tf.cast(tf.argmax(y_true, -1) == tf.argmax(y_pred, -1),dtype=tf.int64))
    total = tf.cast(tf.reduce_sum(y_true),dtype=tf.int64)
    return count/total

def f1Score(y_true, y_pred):
    '''
    Computes the f1 score defined as:
    f1-score = 2 * (Precision * Recall) / (Precision + Recall)

    where
    Precision = tp / (tp + fp)
    Recall = tp / (tp + fn)
    '''
    return f1_score(tf.argmax(y_true, -1), tf.argmax(y_pred, -1), average='macro')


def plotModelPerformance(tr_loss, tr_acc, val_loss, val_acc, save_path, display=False):
    '''
    Saves training and validation curves.
    INPUTS
    - tr_loss: training loss history
    - tr_acc: training accuracy history
    - val_loss: validation loss history
    - val_acc: validation accuracy history
    - save_path: path to where to save the model
    '''

    fig, ax1 = plt.subplots(figsize=(15, 10))
    colors = ['blue', 'orange', 'green', 'red','purple','brown','pink','gray','olive','cyan','teal']
    line_style = [':', '-.', '--', '-']
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=15)
    l1 = ax1.plot(tr_loss, colors[0], ls=line_style[0])
    l2 = ax1.plot(val_loss, colors[1], ls=line_style[1])
    plt.legend(['Training loss', 'Validation loss'])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Accuracy', fontsize=15)
    ax2.set_ylim(bottom=0, top=1)
    l3 = ax2.plot(tr_acc, colors[2], ls=line_style[2])
    l4 = ax2.plot(val_acc, colors[3], ls=line_style[3])

    # add legend
    lns = l1+l2+l3+l4
    labs = ['Training loss', 'Validation loss', 'Training accuracy', 'Validation accuracy']
    ax1.legend(lns, labs, loc=7, fontsize=15)

    ax1.set_title('Training loss and accuracy trends', fontsize=20)
    ax1.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(os.path.join(save_path, 'perfomance.pdf'), bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(save_path, 'perfomance.png'), bbox_inches='tight', dpi = 100)
    plt.close()

    if display is True:
        plt.show()
    else:
        plt.close()

def plotModelPerformance_v2(tr_loss, tr_acc, val_loss, val_acc, tr_f1, val_f1, save_path, display=False, best_epoch=None):
    '''
    Saves training and validation curves.
    INPUTS
    - tr_loss: training loss history
    - tr_acc: training accuracy history
    - tr_f1 : training f1-score history
    - val_loss: validation loss history
    - val_acc: validation accuracy history
    - val_f1 : validation f1-score history
    - save_path: path to where to save the model
    '''

    fig, ax1 = plt.subplots(figsize=(15, 10))
    colors = ['blue', 'orange', 'green', 'red','pink','gray','purple','brown','olive','cyan','teal']
    line_style = [':', '-.', '--', '-']
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Loss', fontsize=15)
    l1 = ax1.plot(tr_loss, colors[0], ls=line_style[2])
    l2 = ax1.plot(val_loss, colors[1], ls=line_style[3])
    plt.legend(['Training loss', 'Validation loss'])

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Accuracy and F1-score', fontsize=15)
    ax2.set_ylim(bottom=0, top=1)
    l3 = ax2.plot(tr_acc, colors[2], ls=line_style[2])
    l4 = ax2.plot(val_acc, colors[3], ls=line_style[3])
    l5 = ax2.plot(tr_f1, colors[4], ls=line_style[2])
    l6 = ax2.plot(val_f1, colors[5], ls=line_style[3])
    if best_epoch:
        l7 = ax2.axvline(x=best_epoch, color=colors[6], ls=line_style[0] )

    # add legend
    if best_epoch:
        # lns = l1+l2+l3+l4+l5+l6+l7
        # labs = ['Training loss', 'Validation loss', 'Training accuracy', 'Validation accuracy', 'Training F1-score', 'Validation F1-score', 'Best_model']
        lns = l1+l2+l3+l4+l5+l6
        labs = ['Training loss', 'Validation loss', 'Training accuracy', 'Validation accuracy', 'Training F1-score', 'Validation F1-score']
        ax1.legend(lns, labs, loc=7, fontsize=15)
    else:
        lns = l1+l2+l3+l4+l5+l6
        labs = ['Training loss', 'Validation loss', 'Training accuracy', 'Validation accuracy', 'Training F1-score', 'Validation F1-score']
        ax1.legend(lns, labs, loc=7, fontsize=15)

    ax1.set_title('Training loss, accuracy and F1-score trends', fontsize=20)
    ax1.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if save_path:
        if os.path.isdir(save_path):
            fig.savefig(os.path.join(save_path, 'perfomance.pdf'), bbox_inches='tight', dpi = 100)
            fig.savefig(os.path.join(save_path, 'perfomance.png'), bbox_inches='tight', dpi = 100)
        else:
            print(f'Save path not found. Given {save_path}. Skipping...')

    if display is True:
        plt.show()
    else:
        plt.close()

def plotLearningRate(lr_history, save_path, display=False):
    '''
    Plots and saves a figure of the learning rate history
    '''
    fig = plt.figure(figsize=(20,20))
    plt.plot(lr_history)
    plt.title('Learning rate trend')
    plt.legend(['Learning rate'])
    fig.savefig(os.path.join(save_path, 'learning_rate.pdf'), bbox_inches='tight', dpi = 100)
    fig.savefig(os.path.join(save_path, 'learning_rate.png'), bbox_inches='tight', dpi = 100)

    if display is True:
        plt.show()
    else:
        plt.close()

def tictoc(tic=0, toc=1):
    '''
    Returns a string and a dictionary that contains the number of days, hours, minutes and
    seconds elapsed between tic and toc
    '''
    elapsed = toc-tic
    days, rem = np.divmod(elapsed, 86400)
    hours, rem = np.divmod(rem, 3600)
    minutes, rem = np.divmod(rem, 60)
    seconds, rem = np.divmod(rem, 1)
    milliseconds = rem*1000

    dictionary = {}
    dictionary['days']=days
    dictionary['hours']=hours
    dictionary['minutes']=minutes
    dictionary['seconds']=seconds
    dictionary['milliseconds']=milliseconds

    # form a string in the format d:h:m:s
    return "%2dd:%02dh:%02dm:%02ds:%02dms" % (days, hours, minutes, seconds, milliseconds), dictionary

def printPerformance(self):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(self.train_loss_results, label='Training loss')
    axes[0].plot(self.val_loss_results, label='Validation loss')
    axes[0].legend(loc='upper right')
    # axes[0].set_ylim([0,5])

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(self.train_accuracy_results, label='Training accuracy')
    axes[1].plot(self.val_accuracy_results, label='Validation accuracy')
    axes[1].legend(loc='upper right')
    plt.show()


## TRAINING ROUTINE FOR NOT VAE MODELS
def train(self, training_dataloader,
                validation_dataloader,
                unique_labels,
                loss=('cee'),
                start_learning_rate = 0.001,
                scheduler='polynomial',
                power=0.1,
                max_epochs=200,
                early_stopping=True,
                patience=20,
                warm_up=True,
                warm_up_learning_rate=0.00001,
                warm_up_epochs=15,
                save_model_path=os.getcwd(),
                save_model_architecture_figure=True,
                verbose=1):

    # define parameters useful to store training and validation information
    self.initial_learning_rate = start_learning_rate
    self.scheduler = scheduler
    self.maxEpochs = max_epochs
    self.learning_rate_history = []
    self.loss = loss
    self.num_validation_samples = 0
    self.num_training_samples = 0
    self.unique_labels=unique_labels
    self.save_model_path=save_model_path
    self.best_epoch = 0

    if verbose <= 2 and isinstance(verbose, int):
        self.verbose=verbose
    else:
        print('Invalid verbose parameter. Given {} but expected 0, 1 or 2. Setting to default 1'.format(verbose))

    # save model architecture figure
    if save_model_architecture_figure is True:
        try:
            tf.keras.utils.plot_model(self.model, to_file=os.path.join(self.save_model_path, 'model_architecture.png'), show_shapes=True)
        except:
            print('Cannot save model architecture as figure. Printing instead.')
            self.model.summary()

    if early_stopping:
        self.best_acc = 0.0
        self.best_f1 = 0.0
        n_wait = 0

    # set loss function based in the type of classification
    '''
    Categorica cross entropy and Sparce categorical cross entropy are the same
    apart from the fact that the first takes gt as categorical and the later as
    sparce (argmax) labels.
    '''
    if len(self.unique_labels) == 2:
        # using binary cross entropy as loss function
        classification_loss_object = tf.keras.losses.BinaryCrossentropy(
                                from_logits=True)
    else:
        classification_loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def classificationLoss(model, x, y, training, class_weights=None):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_= model(x, training=training)

        # Since using Categorical cross entropy, make gt categorical
        if class_weights:
            # tf.gather - gets the weigth for each sample
            per_sample_weights = tf.gather(class_weights, y)
            classification_loss = classification_loss_object(
                        y_true=to_categorical(y, num_classes=len(self.unique_labels)),
                        y_pred=y_,
                        sample_weight=per_sample_weights)
        else:
            classification_loss = classification_loss_object(
                        y_true=to_categorical(y, num_classes=len(self.unique_labels)),
                        y_pred=y_)

        return classification_loss

    def grad(model, inputs, targets, class_weights=None):
        with tf.GradientTape() as tape:
            loss_value = classificationLoss(model, inputs, targets, training=True, class_weights=class_weights)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    # Keep results for plotting
    self.train_loss_history = []
    self.train_accuracy_history = []
    self.val_loss_history = []
    self.val_accuracy_history = []
    self.train_f1_history = []
    self.val_f1_history = []
    start = time.time()

    # initialize the variables
    tr_epoch_loss_avg = tf.keras.metrics.Mean()
    tr_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    tr_epoch_f1 = tfa.metrics.F1Score(num_classes=self.num_classes, average='macro')
    val_epoch_loss_avg = tf.keras.metrics.Mean()
    val_epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    val_epoch_f1 = tfa.metrics.F1Score(num_classes=self.num_classes, average='macro')


    # start looping through the epochs
    for epoch in range(self.maxEpochs):
        # reset metrics (keep only values for one epoch at the time)
        tr_epoch_loss_avg.reset_states()
        tr_epoch_accuracy.reset_states()
        tr_epoch_f1.reset_states()
        val_epoch_loss_avg.reset_states()
        val_epoch_accuracy.reset_states()
        val_epoch_f1.reset_states()

        # compute learning rate based on the scheduler
        if self.scheduler == 'linear':
            self.power = 1
        elif self.scheduler == 'polynomial':
            self.power = power
        elif self.scheduler == 'constant':
            self.power = 0
        else:
            raise TypeError('Invalid scheduler. given {} but expected linear, polynomial or constant'.format(self.scheduler))

        # check if gradient warm-up is needed
        if warm_up is True and epoch < warm_up_epochs:
            lr = warm_up_learning_rate
        else:
            if self.scheduler == 'constant':
                lr =  self.initial_learning_rate
            else:
                if warm_up is True:
                    # do not count the number of epochs used for warm_up
                    lr = leraningRateScheduler(self.initial_learning_rate, epoch-warm_up_epochs, self.maxEpochs, power)
                else:
                    lr = leraningRateScheduler(self.initial_learning_rate, epoch, self.maxEpochs, power)

        # save learning rate info
        self.learning_rate_history.append(lr)

        if "LightOCT" in self.model_name:
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.initial_learning_rate , momentum=0.9, nesterov=False, name='SGD')
        else:

            optimizer = tfa.optimizers.RectifiedAdam(learning_rate=lr)
            optimizer = tfa.optimizers.Lookahead(optimizer=optimizer, sync_period=5, slow_step_size=0.5)

        # ####### TRAINING
        step = 0
        for x, y in training_dataloader:
            step += 1

            # save information about training image size
            if epoch == 0 and step == 1:
                self.batch_size = x.shape[0]
                self.input_size = (x.shape[1], x.shape[2])

            train_loss, grads = grad(self.model, x, y, self.class_weights)

            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Track loss and accuracy
            tr_epoch_loss_avg.update_state(train_loss)
            tr_epoch_accuracy.update_state(y, self.model(x, training=True))
            tr_epoch_f1.update_state(to_categorical(y, num_classes=self.num_classes), self.model(x, training=True))

            # print values
            if self.verbose == 2:
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} training -> {:04d}/unknown -> tr_loss:{:.4f}, tr_acc:{:.4f}, tr_f1:{:.4f}\r'
                            .format(epoch+1,
                                step,
                                tr_epoch_loss_avg.result(),
                                tr_epoch_accuracy.result(),
                                tr_epoch_f1.result()),
                            end='')
                else:
                    print('Epoch {:04d} training -> {:04d}/{:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, tr_f1:{:.4f} \r'
                            .format(epoch+1,
                                    step,
                                    self.num_training_samples//self.batch_size,
                                    tr_epoch_loss_avg.result(),
                                    tr_epoch_accuracy.result(),
                                    tr_epoch_f1.result()),
                            end='')

        # finisced all the training batches -> save training loss
        self.train_loss_history.append(tr_epoch_loss_avg.result().numpy().astype(float))
        self.train_accuracy_history.append(tr_epoch_accuracy.result().numpy().astype(float))
        self.train_f1_history.append(tr_epoch_f1.result().numpy().astype(float))

        if epoch == 0:
            self.num_training_samples = self.batch_size*step

        # ########### VALIDATION
        step = 0
        for x, y in validation_dataloader:
            step += 1

            val_loss = classificationLoss(self.model, x, y, training=False, class_weights=self.class_weights)

            # track progress
            val_epoch_loss_avg.update_state(val_loss)
            val_epoch_accuracy.update_state(y, self.model(x, training=False))
            val_epoch_f1.update_state(to_categorical(y, num_classes=self.num_classes), self.model(x, training=False))

            # print values
            if self.verbose == 2:
                if epoch == 0:
                    print('\r', end='')
                    print('Epoch {:04d} validation -> {:04d}/unknown -> val_loss:{:.4f}, val_acc:{:.4f}, val_f1:{:.4f}\r'
                            .format(epoch+1,
                                    step,
                                    val_epoch_loss_avg.result(),
                                    val_epoch_accuracy.result(),
                                    val_epoch_f1.result()),
                                end='')
                else:
                    print('Epoch {:04d} validation -> {:04d}/{:04d} -> val_loss:{:.4f}, val_acc:{:.4f}, val_f1:{:.4f}\r'
                            .format(epoch+1,
                                step,
                                self.num_validation_samples//self.batch_size,
                                val_epoch_loss_avg.result(),
                                val_epoch_accuracy.result(),
                                val_epoch_f1.result()),
                            end='')


        # finisced all the batches in the validation
        self.val_loss_history.append(val_epoch_loss_avg.result().numpy().astype(float))
        self.val_accuracy_history.append(val_epoch_accuracy.result().numpy().astype(float))
        self.val_f1_history.append(val_epoch_f1.result().numpy().astype(float))


        if self.verbose == 1 or self.verbose == 2:
            print('Epoch {:04d} -> tr_loss:{:.4f}, tr_acc:{:.4f}, tr_f1:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}, val_f1:{:.4f}'.format(epoch+1,
                            self.train_loss_history[-1], self.train_accuracy_history[-1], self.train_f1_history[-1],
                            self.val_loss_history[-1], self.val_accuracy_history[-1],  self.val_f1_history[-1]))
        if epoch == 0:
            self.num_validation_samples = self.batch_size*step

        if (epoch+1) % 5 == 0:
            plotModelPerformance_v2(self.train_loss_history,
                                    self.train_accuracy_history,
                                    self.val_loss_history,
                                    self.val_accuracy_history,
                                    self.train_f1_history,
                                    self.val_f1_history,
                                    self.save_model_path,
                                    best_epoch=self.best_epoch,
                                    display=False)

            plotLearningRate(self.learning_rate_history, self.save_model_path, display=False)

        if early_stopping:
            # check if model accurary improved, and update counter if needed
            if self.val_f1_history[-1] >= self.best_f1:
                # save model checkpoint
                if self.verbose == 1 or self.verbose == 2:
                    print(' - Saving model checkpoint in {}'.format(self.save_model_path))
                # save some extra parameters

                stop = time.time()
                self.training_time, _ = tictoc(start, stop)
                self.training_epochs = epoch
                self.best_acc = self.val_accuracy_history[-1]
                self.best_f1 = self.val_f1_history[-1]
                self.best_epoch = epoch

                # save model with weights
                save_model(self, save_weights=True)

                # reset counter
                n_wait = 0
            else:
                if warm_up is True:
                    if epoch > warm_up_epochs:
                        n_wait += 1
                else:
                    n_wait += 1
            # check max waiting is reached
            if n_wait == patience :
                if self.verbose == 1 or self.verbose == 2:
                    print(' -  Early stopping patient reached. Last model saved in {}'.format(self.save_model_path))
                # saving last model as well
                self.model.save(os.path.join(self.save_model_path, 'last_model'+'.tf'))
                self.model.save_weights(os.path.join(self.save_model_path, 'last_model_weights.tf'))

                # save model info (but not weights)
                save_model(self, save_weights=False)

                plotModelPerformance_v2(self.train_loss_history,
                                        self.train_accuracy_history,
                                        self.val_loss_history,
                                        self.val_accuracy_history,
                                        self.train_f1_history,
                                        self.val_f1_history,
                                        self.save_model_path,
                                        best_epoch=self.best_epoch,
                                        display=False)
                break

        # save last model even when running through all the available epochs
        if epoch == self.maxEpochs-1:
            if self.verbose == 1 or self.verbose == 2:
                print(' -  Run through all the epochs. Last model saved in {}'.format(self.save_model_path))

            # saving last model as well
            self.model.save(os.path.join(self.save_model_path, 'last_model'+'.tf'))
            self.model.save_weights(os.path.join(self.save_model_path, 'last_model_weights.tf'))

            # save model info (but not weights)
            save_model(self, save_weights=False)

            plotModelPerformance_v2(self.train_loss_history,
                                    self.train_accuracy_history,
                                    self.val_loss_history,
                                    self.val_accuracy_history,
                                    self.train_f1_history,
                                    self.val_f1_history,
                                    self.save_model_path,
                                    best_epoch=self.best_epoch,
                                    display=False)
            break

## TEST ROUTINE
def test(self, test_dataloader):
    '''

    Function that given a trained model outputs the prediction on the test dataset

    INPUTS

    self : models_tf.model
        Trained model
    test_dataloader : tensorflow.python.data.ops.dataset_ops.PrefetchDataset
            Tensorflow dataloader that outputs a tuple of (image, label)

    OUTPUTS

    test_gt : tensorflow.tensor
        Ground truth values
    test_logits : tensorflow.tensor
        Logits predicted values
    test_time : float
        Time spent for testing

    Testing loop
    1 - for all the batches of data in the dataloader
        - fix labels based on the unique labels specification
        - compute prediction on all the test images
    2 - compute accuracy
    3 - print class accuracy and overall accuracy
    4 - return groud truth, predicted values and test time
    '''
    # initialize variables
    test_logits = tf.zeros([0, len(self.unique_labels)])
    test_gt = tf.zeros([0, len(self.unique_labels)])

    step = 0
    test_start = time.time()
    for x, y in test_dataloader:
        x = x.numpy()
        test_gt = tf.concat([test_gt, to_categorical(y.numpy(), num_classes=len(self.unique_labels))], axis=0)
        if 'VAE' in self.model_name:
            aus_logits, _, _, _, _, _ = self.model(x, training=False)
        else:
            aus_logits= self.model(x, training=False)
        test_logits = tf.concat([test_logits, aus_logits], axis=0)

    test_stop = time.time()
    print(f'Model accuracy: {accuracy(test_gt, test_logits):.02f}')

    # return test predictions
    return test_gt, test_logits, test_stop-test_start

def test_independent(model, configuration, test_dataloader):
    '''

    Function that given a trained model outputs the prediction on the test dataset

    INPUT


    model : tensorflow model
        Trained model
    config : dict
        dictionary containing all the information regarding the model training
    test_dataloader : tensorflow.python.data.ops.dataset_ops.PrefetchDataset
            Tensorflow dataloader that outputs a tuple of (image, label)

    OUTPUT
    test_gt : tensorflow.tensor
        Ground truth values
    test_logits : tensorflow.tensor
        Logits predicted values
    test_time : float
        Time spent for testing

    Testing loop
    1 - for all the batches of data in the dataloader
        - fix labels based on the unique labels specification
        - compute prediction on all the test images
    2 - compute accuracy
    3 - print class accuracy and overall accuracy
    4 - return groud truth, predicted values and test time
    '''
    # initialize variables
    test_logits = tf.zeros([0, len(configuration["unique_labels"])])
    test_gt = tf.zeros([0, len(configuration["unique_labels"])])

    step = 0
    test_start = time.time()
    for x, y in test_dataloader:
        x = x.numpy()
        y = to_categorical(y.numpy(), num_classes=len(configuration["unique_labels"]))
        test_gt = tf.concat([test_gt, y], axis=0)
        if 'VAE' in configuration['model_save_name']:
            aus_logits, _, _, _, _, _ = model(x, training=False)
        else:
            aus_logits= model(x, training=False)
        test_logits = tf.concat([test_logits, aus_logits], axis=0)

    test_stop = time.time()
    print(f'{" "*4}Model accuracy: {accuracy(test_gt, test_logits):.02f}')

    # return test predictions
    return test_gt, test_logits, test_stop-test_start

def save_model(self, save_weights=True):
    '''

    This functionsaves the model along with its weights and additional
    information about its architecture and training performance.

    Parameters
    ---------
    self : models_tf.model
        Model to save
    '''

    # save model and weights
    if save_weights:
       self.model.save(os.path.join(self.save_model_path, 'model'+'.tf'))
       self.model.save_weights(os.path.join(self.save_model_path, 'model_weights.tf'))

    # save extra information
    model_summary = {
        'Model_name' : self.model_name,
        'Num_input_channels': self.number_of_input_channels if hasattr(self, 'number_of_input_channels') else 'NaN',
        'Num_classes' : self.num_classes,
        'Input_size' : self.input_size,
        'Class_weights': self.class_weights.tolist() if not isinstance(self.class_weights, list) else self.class_weights,
        'Custom_model': self.custom_model,
        'Dropout_rate': self.dropout_rate if hasattr(self, 'dropout_rate') else 'NaN',
        'Normalization': self.normalization if hasattr(self, 'normalization') else 'NaN',
        'Model_depth' : int(self.depth),
        'Num_filter_start' : self.num_filter_start if hasattr(self, 'num_filter_start') else 'NaN',
        'Kernel_size' : list(self.kernel_size) if hasattr(self, 'kernel_size') else 'NaN',
        'Num_training_samples' : int(self.num_training_samples),
        'nVALIDATION' :  int(self.num_validation_samples),
        'BATCH_SIZE' : int(self.batch_size),
        'EPOCHS' : int(self.training_epochs),
        'TRAINING_TIME' : self.training_time,
        'BEST_ACC' : self.best_acc.astype(float),
        'BEST_ACC' : self.best_f1.astype(float),
        'LOSS': self.loss,
        'INITIAL_LERANING_RATE':self.initial_learning_rate,
        'SCHEDULER':self.scheduler,
        'POWER':self.power,
        'TRAIN_LOSS_HISTORY':self.train_loss_history,
        'VALIDATION_LOSS_HISTORY':self.val_loss_history,
        'TRAIN_ACC_HISTORY':self.train_accuracy_history,
        'VALIDATION_ACC_HISTORY':self.val_accuracy_history,
        'TRAIN_F1_HISTORY':self.train_f1_history,
        'VALIDATION_F1_HISTORY':self.val_f1_history,
        }
    # for key, value in model_summary.items():
    #     print(f'{key} - {type(value)}')
    with open(os.path.join(self.save_model_path, 'model_summary_json.txt'), 'w') as outfile:
        json.dump(model_summary, outfile)



















