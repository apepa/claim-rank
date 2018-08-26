from configparser import ConfigParser

def get_config():
    config = ConfigParser()
    config.read("config.ini")
    return config['claimrank']