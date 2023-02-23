from app_model import svc_model
from app_preprocessing import review_process
import pandas as pd

tf, model  = svc_model()
answer = input('Write your Review: ') 
while answer != "" :  
    answer = pd.Series(review_process(answer))
    new = tf.transform(answer)
    if str(model.predict(new)[0]) == '1':
        print('Positive Review')
    else:
        print('Negative Review')
    answer = input('Write your Review or print enter to terminate the app: ')
