import urllib
import requests
import argparse
import os

class Cfg(object):
    def __init__(self):
        super(Cfg, self).__init__()
        self.data_url = ["https://github.com/DatacollectorVN/Machine-Learning-Tutorial/releases/download/data/titanic.csv"]
        self.data_name = ["titanic.csv"]
    
    def download_data(self, data_name):
        os.makedirs("data", exist_ok = True)

        destination = os.path.join("data", data_name)
        data_url = self.data_url[self.data_name.index(data_name)]
        print ('Start to download, this process take a few minutes')
        urllib.request.urlretrieve(data_url, destination)
        print("Downloaded data - {} to-'{}'".format(data_url, destination))


def main(data_name):
    cfg = Cfg()
    cfg.download_data(data_name = data_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataname', help = 'data name', type = str, dest = "data_name",
                        default = 'data_name')

    args = parser.parse_args()
    main(data_name = args.data_name)