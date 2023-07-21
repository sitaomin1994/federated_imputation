from abc import ABC, abstractmethod

import numpy as np
import pymongo
import json
from pathlib import Path
from config import settings, ROOT_DIR
import base64


class Backend(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def save(self, experiment_name, experiment_result):
        pass


class MongoBackend(Backend):

    def __init__(self):
        super().__init__()
        self.client = None
        self.db = None
        self.collection = None
        self.db_name = settings['mongodb_backend']['db']

    def connect(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client[self.db_name]

    def save(self, experiment_type, experiment_result):
        if self.db is None:
            raise Exception("Please connect to MongoDB before saving fed_imp results")

        if experiment_type not in self.db.list_collection_names():
            self.db.create_collection(experiment_type)

        collection = self.db[experiment_type]
        collection.insert_one(experiment_result)

    def find_experiments(self, experiment_type, query):
        if self.db is None:
            raise Exception("Please connect to MongoDB before finding experiments")

        if experiment_type not in self.db.collection_names():
            raise Exception("Experiment type {} not found in database".format(experiment_type))

        collection = self.db[experiment_type]
        return collection.find(query)

    def disconnect(self):
        self.client.close()


class FileBackend:

    def __init__(self):
        self.result_dir_path = settings['experiment_result_dir']
        self.config_dir_path = settings['experiment_config_dir']
        self.raw_result_dir_path = settings['raw_result_dir']

    def save(self, dir_name, file_name, experiment_data):
        # create directory if not exists
        dir_path = ROOT_DIR + "/" + self.result_dir_path + '/' + self.raw_result_dir_path + '/' + dir_name
        Path(dir_path).mkdir(parents=True, exist_ok=True)

        # save images
        if "results" in experiment_data:
            if "plots" in experiment_data["results"]:
                for i in range(len(experiment_data["results"]["plots"])):
                    img_file_name = file_name.replace(".json", '.png')
                    with open(dir_path + '/' + img_file_name, 'wb') as f:
                        f.write(base64.b64decode(experiment_data["results"]["plots"][i]))

        # save data
        if "results" in experiment_data:
            if "data" in experiment_data["results"]:
                data_dir = dir_path + '/' + file_name.split('.')[0]
                Path(data_dir).mkdir(parents=True, exist_ok=True)
                for key, value in experiment_data["results"]["data"].items():
                    if isinstance(value, np.ndarray):
                        data_file_name = "{}.npy".format(key)
                        np.save(data_dir + '/' + data_file_name, value)
                    elif isinstance(value, list):
                        for idx, item in enumerate(value):
                            data_file_name = "{}_{}.npy".format(key, idx)
                            np.save(data_dir + '/' + data_file_name, item)

        # save experiment config and results
        experiment_data["results"]["plots"] = []
        experiment_data["results"]["data"] = None
        with open(dir_path + '/' + file_name, 'w') as f:
            json.dump(experiment_data, f)
