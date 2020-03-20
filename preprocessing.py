import pandas as pd
source = pd.read_csv('data/data(201502).csv', parse_dates=["pickup_datetime"])
print("---finish import---")
source = source.dropna(how = 'any', axis = 'rows')
source = source[['pickup_datetime','pickup_longitude','pickup_latitude']]
source = source.astype({
  'pickup_longitude':float,
  'pickup_latitude':float
  })
delete_index_list = []
for item in source.iterrows():
  if (abs(item[1]['pickup_longitude']-0))<0.001:
    delete_index_list.append(item[0])
    # source = source.drop(item[0])
  elif (abs(item[1]['pickup_latitude']-0))<0.001:
    # print(item[1])
    delete_index_list.append(item[0])
  # elif (item[1]['pickup_longitude']<-74.252193 or item[1]['pickup_longitude']>-72.986532) and (abs(item[1]['pickup_longitude']-0))>0.001:
  elif (item[1]['pickup_longitude']<-74.025 or item[1]['pickup_longitude']>-73.925) and (abs(item[1]['pickup_longitude']-0))>0.001:
    delete_index_list.append(item[0])
  elif (item[1]['pickup_latitude']<40.7 or item[1]['pickup_latitude']>40.8) and (abs(item[1]['pickup_latitude']-0))>0.001:
    delete_index_list.append(item[0])

source = source.drop(delete_index_list)
source.to_csv('Data/train(201502).csv',index=False)
