from configparser import ConfigParser

def get_config():
    config = ConfigParser()
    config.read("../../config.ini") # config.read("../../config.ini") (IQJ)
    return config['claimrank']