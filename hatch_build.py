import os
import scipy
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        build_data['include-dirs'] = [os.path.dirname(scipy.__file__)]

def get_include_dirs():
    return [os.path.dirname(scipy.__file__)]

def build_hook(config):
    config['include-dirs'] = get_include_dirs()
