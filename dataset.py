import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
import os
import numpy as np
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="datasets")

mySoccerNetDownloader.downloadDataTask(task="calibration", split=["train","valid","test"])
mySoccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train","valid","test"])
# mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test"])
# mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test"])



