import warnings



# loaded config
lstm_conf = {
    'WINDOW_SIZE' : 15,
    'NUM_OF_FEATURES': 42,
    'LABEL_LENGTH': 4,
    'BATCH_SIZE': 25,
    'DROPOUT': 0.3,
    'OPTIMIZER': 'adam'
}




class Config(object):
    def __init__(self, conf):
        self._config = conf # set it to conf

    def get_property(self, property_name):
        if property_name not in self._config.keys(): # we don't want KeyError
            return ValueError  # just return None if not found
        return self._config[property_name]

    def set_property(self, property_name, value):
        if property_name not in self._config.keys():  # we don't want KeyError
            return ValueError  # just return None if not found

        # Raise a warning in case the user sets a property with a different type
        if type(property_name) != type(value):
            warnings.warn('Setting configuration property from {0}, to {1}'.format(type(property_name), type(value)),
                          RuntimeWarning)


        self._config[property_name] = value


class LstmConfig(Config):

    def __init__(self):
        super(LstmConfig, self).__init__(conf= lstm_conf)


    @property
    def window_size(self):
        return self.get_property('WINDOW_SIZE')

    @window_size.setter
    def window_size(self, value):
        self.set_property('WINDOW_SIZE', value)

    @property
    def num_of_features(self):
        return self.get_property('NUM_OF_FEATURES')

    @num_of_features.setter
    def num_of_features(self, value):
        self.set_property('NUM_OF_FEATURES', value)

    @property
    def label_length(self):
        return self.get_property('LABEL_LENGTH')

    @label_length.setter
    def label_length(self, value):
        self.set_property('LABEL_LENGTH', value)

    @property
    def batch_size(self):
        return self.get_property('BATCH_SIZE')

    @batch_size.setter
    def batch_size(self, value):
        self.set_property('BATCH_SIZE', value)

    @property
    def dropout(self):
        return self.get_property('DROPOUT')

    @dropout.setter
    def dropout(self, value):
        self.set_property('DROPOUT', value)

    @property
    def optimizer(self):
        return self.get_property('OPTIMIZER')

    @optimizer.setter
    def optimizer(self, value):
        self.set_property('OPTIMIZER', value)

lconf = LstmConfig()


for value in [name for name, value in vars(lconf).items() if isinstance(value, property)]:
    print(value)

