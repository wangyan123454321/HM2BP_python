import os
import json
from copy import deepcopy

class Config:
    def __init__(self, data):
        assert "LAYERS" in data
        self.data = data
    
    def param(self, name: str):
        assert name != "LAYERS" and name in self.data
        return self.data[name]
    
    def get_layer_by_idx(self, idx: int):
        assert idx >= 0 and idx < self.layer_count()
        return self.data["LAYERS"][idx]
    
    def get_layer_by_name(self, name: str):
        for idx in range(self.layer_count):
            if self.get_layer_by_idx(idx)["NAME"] == name:
                return self.get_layer_by_idx(idx)
        return None
    
    def layer_count(self):
        return len(self.data["LAYERS"])
    
    def hyperparameters(self):
        data_out = deepcopy(self.data)
        del data_out["LAYERS"]
        return data_out
        
    def layers(self):
        data_out = deepcopy(self.data["LAYERS"])
        return data_out

def read(config_name: str) -> Config:
    path = "." + os.sep + "config" + os.sep + config_name + ".json"
    assert os.path.isfile(path)
    res = None
    with open(path, "r") as f:
        res = Config(json.loads(f.read()))
    return res