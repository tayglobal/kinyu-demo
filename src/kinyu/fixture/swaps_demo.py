from kinyu.db.base import BaseDB
from kinyu.db.api import kydb
from datetime import date
import json
import os


def create_swaps_fixture(db: BaseDB):
    res = kydb.connect('memory://unittests')
    env = db.new('kinyu.env.Env', '/Env')
    env.marketdata_interface = '/TestMarketDataInterface'
    env.pricing_date = date(2012, 12, 11)
    env.put()
    mi = db.new('kinyu.marketdata.interface.MarketDataInterface',
                '/TestMarketDataInterface')
    mi.put()

    eonia_curve = db.new('kinyu.marketdata.ir.funding.FundingCurve',
                         mi.get_marketdata_path('EUR-EONIA-1B'),
                         index='Eonia')
    eonia_curve.put()

    data_folder = os.getcwd() + '/../data/'
    path = data_folder + 'eonia_marketdata.json'
    with open(path, 'r') as f:
        marketdata = json.load(f)

    db[mi.get_marketdata_path('EUR-EONIA-1B.Data')] = marketdata
