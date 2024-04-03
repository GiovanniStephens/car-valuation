from requests_oauthlib import OAuth1Session
import keyring
import yaml
from yaml.loader import SafeLoader
import json


def load_config(filename: str) -> dict:
    with open(filename, 'r') as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


config = load_config('config.yml')
consumerKey = keyring.get_password('Trademe', 'key')
consumerSecret = keyring.get_password('Trademe', 'secret')

tradeMe = OAuth1Session(consumerKey, consumerSecret)

make = config['make']
model = config['model']
url = f'https://api.trademe.co.nz/v1/Search/Motors/Used.json?make={make}&model={model}&rows=500'
returnedPageAll = tradeMe.get(url)
dataRawAll = returnedPageAll.content
parsedDataAll = json.loads(dataRawAll)

print(parsedDataAll)
