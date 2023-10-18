from pytorch_lightning.core import LightningModule

class Distiller(LightningModule):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.net = student
        self.t_net = teacher

    def forward(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()