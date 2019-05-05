from ml.utils import batch


class Model():

    def __init__(self, model=None, data_params=None, train_params=None,
                 checkpoint_params=None, model_params=None):
        self.model = model
        self.data_params = data_params
        self.train_params = train_params
        self.checkpoint_params = checkpoint_params
        self.model_params = model_params

    def __getitem__(self, item):
        value = exec('self.{}'.format(item))
        return value

    def check_model(self):
        if self.model is None:
            self.model = lambda x: x

    def initialise_model(self):
        self.model = self.model(self.model_params)

    def generate_bacthes(self):
        return batch.batch(**self.data_params)
        

    
