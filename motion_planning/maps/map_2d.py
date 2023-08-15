import yaml
import numpy as np


class Map:
    pass


class Map2D:
    def __init__(self, config_path: str):
        self.map_path = config_path
        config = yaml.safe_load(open(config_path, "r"))

        if 'map' in config:
            map_config = config['map']
        else:
            raise ValueError
        if 'dimensions' in map_config:
            self.width, self.heigth = map_config['dimensions']
        else:
            raise ValueError

    def sample_pos_uniform(self):
        """
        samples a valid pos from inside the map
        this does not incclude a check for collisions!
        @return:
        """
        x = np.random.uniform(0, self.width)
        y = np.random.uniform(0, self.heigth)
        return x, y
