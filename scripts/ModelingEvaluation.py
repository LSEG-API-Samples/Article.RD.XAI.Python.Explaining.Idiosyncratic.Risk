import tensorflow as tf
import os
import shap
import matplotlib.pyplot as plt
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from utilities import xai_logs_path
import datetime


@dataclass(frozen=True)
class DNNExceptionData:
    data: str


class DNNException(Exception):
    def __init__(self, exception_details: DNNExceptionData):
        self.details = exception_details

    def to_string(self):
        return self.details.data


class XAI:
    def __init__(self, model_name, x, y, structure={'core': [{'id': 'input', 'type': 'Dense', 'neurons': 3},
                                                    {'id': 'hidden', 'type': 'Dense',
                                                        'neurons': 10},
                                                    {'id': 'output', 'type': 'Dense', 'neurons': 1}],
                                                    'hp': {'dropout_rate': 0.1, 'learning_rate': 0.01}}):
        self.structure = structure
        self.feature_names = x.columns.values
        self.model_name = model_name
        self.x = x
        self.y = y

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x,
                                                                                y,
                                                                                test_size=0.20,
                                                                                random_state=0)
        self.num_of_features = self.x.shape[1]

        for ix, layer in enumerate(structure['core']):
            if layer['id'] == 'input':
                structure['core'][ix]['neurons'] = self.num_of_features

        self.log_dir = "./logs/" + model_name + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.dnn = DeepNeuralNetwork(structure=self.structure)

    def fit(self, max_epochs=150, produce_tensorboard=True, explainable=True):
        _callbacks = []
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=7, start_from_epoch=70)
        _callbacks.append(stop_early)

        if produce_tensorboard:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=self.log_dir, histogram_freq=1)
            _callbacks.append(tensorboard_callback)

        hp_tuner = kt.Hyperband(self.dnn.model_builder, objective='mean_squared_error', max_epochs=max_epochs, factor=5,
                                directory='./hp/', project_name='kt_hb_' + self.model_name)

        hp_tuner.search(self.x_train, self.y_train, epochs=max_epochs, validation_split=0.1, callbacks=[
                        tf.keras.callbacks.TensorBoard('./hp/tb_logs/')])
        best_hpm = hp_tuner.get_best_hyperparameters(num_trials=1)[0]

        self.dnn.model = hp_tuner.hypermodel.build(best_hpm)
        train_history = self.dnn.model.fit(
            self.x_train, self.y_train, epochs=max_epochs, validation_data=(
                self.x_test, self.y_test),
            callbacks=[stop_early])

        train_loss_epoch = train_history.history['loss']
        best_epoch_num = train_loss_epoch.index(min(train_loss_epoch))
        self.dnn.model = hp_tuner.hypermodel.build(best_hpm)

        self.dnn.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test),
                           epochs=best_epoch_num, callbacks=_callbacks)
        if explainable:
            self.get_explanations(
                self.dnn.model, self.x_train.to_numpy(), self.x_test.to_numpy(), visualisations=True)

    def get_explanations(self, model, background_data, input_data, visualisations=True):

        self.explainer = shap.DeepExplainer(
            model, background_data
        )
        self.shap_values = self.explainer.shap_values(input_data)
        if visualisations == True:
            self.get_visualisations(self.shap_values, self.explainer)

    def get_visualisations(self, shap_values, explainer,
                           decision_plot=True,
                           force_plot=True,
                           waterfall_plot=True,
                           summary_plot=True):
        shap.initjs()
        if summary_plot:
            shap.summary_plot(shap_values[0],
                              feature_names=self.feature_names, plot_type='bar', show=False)
            plt.savefig(
                xai_logs_path + self.model_name + '_summary_plot.png', bbox_inches='tight', dpi=600)
            plt.close()
        if force_plot:
            shap.force_plot(explainer.expected_value[0].numpy(),
                            shap_values[0][0], features=self.feature_names, matplotlib=True, show=False)
            plt.savefig(
                xai_logs_path + self.model_name + '_force_plot.png', bbox_inches='tight', dpi=600)
            plt.close()

        if waterfall_plot:
            shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0].numpy(),
                                                   shap_values[0][0],
                                                   features=self.x_test.iloc[0, :],
                                                   feature_names=self.feature_names, show=False)
            plt.savefig(
                xai_logs_path + self.model_name + '_waterfall_plot.png', bbox_inches='tight', dpi=600)
            plt.close()

        if decision_plot:
            shap.decision_plot(explainer.expected_value[0].numpy(),
                               shap_values[0][0],
                               features=self.x_test.iloc[0, :],
                               feature_names=self.feature_names.tolist(), show=False)
            plt.savefig(
                xai_logs_path + self.model_name + '_decision_plot.png', bbox_inches='tight', dpi=600)
            plt.close()


class DeepNeuralNetwork(tf.keras.Model):
    def __init__(self, structure):
        super(DeepNeuralNetwork, self).__init__(name='DNN')
        self.structure = structure

    def model_builder(self, hp):
        _model = tf.keras.Sequential()

        hp_neurons = hp.Int('units', min_value=5, max_value=20, step=1)
        hp_dropout_rate = hp.Choice('dropout_rate', values=[0.05, 0.1, 0.2])
        hp_learning_rate = hp.Choice(
            'learning_rate', values=[1e-2, 1e-3, 1e-4])

        for ix, layer in enumerate(self.structure['core']):
            if layer['id'] == 'input':
                _model.add(tf.keras.layers.Flatten(
                    input_shape=(layer['neurons'], )))
                exec(
                    f"_model.add(tf.keras.layers.{layer['type']}({layer['neurons']}, trainable=True))")
                _model.add(tf.keras.layers.Dropout(hp_dropout_rate))
            elif layer['id'] == 'hidden':
                exec(
                    f"_model.add(tf.keras.layers.{layer['type']}(hp_neurons, activation='elu', trainable=True))")
                _model.add(tf.keras.layers.Dropout(hp_dropout_rate))
            else:
                exec(
                    f"_model.add(tf.keras.layers.{layer['type']}({layer['neurons']}, trainable=True))")
        _model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate,
                                                                 epsilon=1e-07),
                       loss=tf.keras.losses.MeanSquaredError(),
                       metrics=["mean_squared_error"])

        return _model
