from os import path
import yaml
import cpgan


with open(path.join(path.dirname(cpgan.__file__),"config.yaml"),'r') as f:
    yaml_f = yaml.safe_load(f)


