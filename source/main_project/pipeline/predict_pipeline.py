import sys
import pandas as pd
from source.commons import load_object
from source.exception import UserException
from source.logger import logging

class PredicPipeline:
    def __init__(self):
        pass
    
    logging.info('Preprocessing user input and making predictions')
    def predict(self,features):
        model_path = 'elements\model.pkl'
        preprocessor_path = 'elements\preprocessor.pkl'
        # loaeding objects
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        
        data_processed = preprocessor.transform(features)
        prediction = model.predict(data_processed)
        
        return prediction
logging.info('This class is responsible for mapping all the inputs from html to flask')
class UserData:
    def __init__(self,gender,campaignchannel,campaigntype,adspend,clickthroughrate,pagespervisit,
                 timeonsite,emailopens,emailclicks,previouspurchases,loyaltypoints):
        self.gen = gender
        self.chan = campaignchannel
        self.camp = campaigntype
        self.ad = adspend
        self.click = clickthroughrate
        self.page = pagespervisit
        self.site = timeonsite
        self.email_ = emailopens
        self.eclick = emailclicks
        self.purch = previouspurchases
        self.points = loyaltypoints
        
    # let's write a function that returns the user input as a pandas dataframe
    def get_data_as_df(self):
        try:
            user_data = {
                "gender":[self.gen],
                "campaignchannel":[self.chan],
                "campaigntype":[self.camp],
                "adspend":[self.ad],
                "clickthroughrate":[self.click],
                "pagespervisit":[self.page],
                "timeonsite":[self.site],
                "emailopens":[self.email_],
                "emailclicks":[self.eclick],
                "previouspurchases":[self.purch],
                "loyaltypoints":[self.points]
            }
            return pd.DataFrame(user_data)
        except Exception as e:
            raise UserException(e,sys)
        