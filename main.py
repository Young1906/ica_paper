import configparser

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config.ini")

    print(config["DEFAULT"]["FS"])