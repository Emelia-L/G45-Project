from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.metrics import  mean_squared_log_error
pred =  pd.read_csv('5_2_tesing_result.csv', sep=',',parse_dates=['trip_start_timestamp'])
test  = pd.read_csv('5_2_groundtruth.csv', sep=',',parse_dates=['trip_start_timestamp'])
# fare = np.hstack((np.array(pred['order1_fare']),np.array(pred['order2_fare']),np.array(pred['order3_fare'])))
# truth_fare = np.hstack((np.array(test['order1_fare']),np.array(test['order2_fare']),np.array(test['order3_fare'])))
tips = np.hstack((np.array(pred['order1_tips']),np.array(pred['order2_tips']),np.array(pred['order3_tips'])))
truth_tips = np.hstack((np.array(test['order1_tips']),np.array(test['order2_tips']),np.array(test['order3_tips'])))
# fare_err = mean_squared_log_error(fare, truth_fare)
tips_err = mean_squared_log_error(tips, truth_tips)
# print('mean log square error of fare is: ' + str(fare_err))
print('mean log square error of tips is: ' + str(tips_err))
print('accuracy for best by fare: %s' % accuracy_score(pred['best'], test['best']))

