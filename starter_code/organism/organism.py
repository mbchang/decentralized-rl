from starter_code.infrastructure.utils import visualize_parameters

class Organism():
    def forward(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def store_path(self):
        raise NotImplementedError

    def clear_buffer(self):
        raise NotImplementedError

    def visualize_parameters(self, pfunc):
        visualize_parameters(self, pfunc)