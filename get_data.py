from requests_oauthlib import OAuth1Session
import keyring
import yaml
from yaml.loader import SafeLoader
import json
import time
import pandas as pd


def load_config(filename: str) -> dict:
    with open(filename, "r") as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


config = load_config("data/config.yml")
consumerKey = keyring.get_password("Trademe", "key")
consumerSecret = keyring.get_password("Trademe", "secret")

tradeMe = OAuth1Session(consumerKey, consumerSecret)

make = config["make"]
model = config["model"]
base_url = f"https://api.trademe.co.nz/v1/Search/Motors/Used.json?make={make}&model={model}&rows=500"
returnedPageAll = tradeMe.get(base_url)
dataRawAll = returnedPageAll.content
parsedDataAll = json.loads(dataRawAll)
totalCount = parsedDataAll["TotalCount"]
n_pages = int(totalCount / 500) + 1
for i in range(1, n_pages + 1):
    url = base_url + f"&page={i}"
    returnedPageAll = tradeMe.get(url)
    dataRawAll = returnedPageAll.content
    parsedDataAll = json.loads(dataRawAll)
    eachListingAll = parsedDataAll["List"]
    pandaAll = pd.DataFrame.from_dict(eachListingAll)
    if i == 1:
        pandaAll.to_pickle(f"data/{make}_{model}_data.pkl")
    else:
        pandaAllStorage = pd.read_pickle(f"data/{make}_{model}_data.pkl")
        pandaAllStorage = pandaAllStorage._append(pandaAll, ignore_index=True)
        pandaAllStorage.to_pickle(f"data/{make}_{model}_data.pkl")
    time.sleep(0.5)
    print(f"{i} out of {n_pages}")
print("All pages saved")
