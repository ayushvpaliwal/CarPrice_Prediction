import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization

# sample = 28
sample = 24
trainsample = sample - 1
# time = 24
time = 28
testtime = time - 4
size1 = 15
size2 = 15
batchsize = 30
epochs = 800
num_max = 0
# the_time_split = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4]

data = np.zeros((sample,time,size1,size2,1))

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, size1, size2, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

# seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
#                    padding='same', return_sequences=True))
# seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='mean_squared_logarithmic_error', optimizer='adam',metrics='mean_squared_logarithmic_error')
print("______________________________________")
print(seq.summary())
print("______________________________________")

def getData():
  global num_max
  source = pd.read_csv('data/train(201502).csv', parse_dates=["pickup_datetime"])
  print("---finish import---")
  source = source.dropna(how = 'any', axis = 'rows')
  source = source[['pickup_datetime','pickup_longitude','pickup_latitude']]
  source = source.astype({
    'pickup_longitude':float,
    'pickup_latitude':float
    })
  print(source.size)
  lon_max = source['pickup_longitude'].max() + 0.0000000001
  lon_min = source['pickup_longitude'].min()
  lat_max = source['pickup_latitude'].max() + 0.0000000001
  lat_min = source['pickup_latitude'].min()
  print("The data discribe:",lon_max,lon_min,lat_max,lat_min)
  source = source.groupby(source['pickup_datetime'].dt.hour)
  # source = source.groupby(source['pickup_datetime'].dt.date)
  step1 = 0
  step2 = 0
  for times,tmp in source: #for every day (sample)
    # print(type(tmp['pickup_datetime'].dt.hour))
    # tmp = tmp.assign(hour = thehour)
    # tmp = tmp.assign(hour = lambda x:x['pickup_datetime'].dt.hour)
    tmp = tmp.assign(hour = lambda x:x['pickup_datetime'].dt.date)
    tmp = tmp[['hour','pickup_longitude','pickup_latitude']]
    tmp = tmp.groupby(tmp['hour'])
    step2 = 0
    for hour, item in tmp: #for every hour
      item = item.to_numpy()
      for i in item: # for every item in hour
        lon = i[1]
        lat = i[2]
        lon = size1 * (lon - lon_min) / (lon_max - lon_min)
        lat = size2 * (lat - lat_min) / (lat_max - lat_min)
        lon = int(lon)
        lat = int(lat)
        data[step1][step2][lon][lat][0] += 1
      if num_max < data[step1][step2][lon][lat][0] and step1 != sample:
        num_max = data[step1][step2][lon][lat][0]
      step2 += 1
    # print("Step 2 is ",step2)
    step1 += 1
    if step1 == sample+1:
      print("---finish data counting?????---")
      break

  for i in range(sample):
    for j in range(time):
      for k1 in range(size1):
        for k2 in range(size2):
          # print(">>>",i,">>>>>>>",j)
          data[i][j][k1][k2][0] = 1.0 * data[i][j][k1][k2][0] / num_max
    # print("Step 1 is ",step1)
      # return data
  print("Step 1 is",step1)
  print("---finish data---")
  return data

def main():
  data = getData()
  #Train the network - use 26 days(or can be regarded as 27 days train 26 times)
  newdata = data[1:trainsample]
  olddata = data[:trainsample-1]
  seq.fit(olddata,newdata, batch_size=batchsize, epochs=epochs, validation_split=0.05)
  print("---finish training---")

  #Test the network - use the last day
  testdata = data[sample-2]
  # use the first 20 hours to get the last 4
  track = testdata[:testtime,::,::,::]
  # print(">>>>>><><>",track.shape)

  for i in range(time-testtime):
    print("The time",i)
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)
    score = seq.evaluate(track[np.newaxis, ::, ::, ::, ::],testdata[:(testtime+i+1)][np.newaxis, ::, ::, ::, ::], batch_size=batchsize, verbose=1)
    print(score)
  print(track.shape)
  # print(data[sample-1][20][:,:,0].shape)
  # print(data[sample-1][20][:,:,0].tolist())
  # print(track[23][:,:,0].tolist())
  # print("The Norm Value is: ",num_max)
  F = open("model8.out","w")
  F.write("The first result: ")
  for i in range(time-testtime):
    F.write("The time is: "+str(i+testtime)+"\n")
    # F.write(str(data[sample-1][i+testtime][:,:,0].tolist()))
    tmp = testdata[i+testtime][:,:,0].tolist()
    for j1 in range(size1):
      for j2 in range(size2):
        F.write(str(int(num_max*tmp[j1][j2]))+" ")
      F.write('\n')
    F.write('\n')
    # F.write(str(track[i+testtime][:,:,0].tolist()))
    tmp = track[i+testtime][:,:,0].tolist()
    for j1 in range(size1):
      for j2 in range(size2):
        F.write(str(int(num_max*tmp[j1][j2]))+" ")
      F.write('\n')

  testdata = data[sample-1]
  # use the first 20 hours to get the last 4
  track = testdata[:testtime,::,::,::]
  # print(">>>>>><><>",track.shape)

  for i in range(time-testtime):
    print("The time",i)
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)
    score = seq.evaluate(track[np.newaxis, ::, ::, ::, ::],testdata[:(testtime+i+1)][np.newaxis, ::, ::, ::, ::], batch_size=batchsize, verbose=1)
    print(score)
  print(track.shape)

  F.write("The second result: ")
  for i in range(time-testtime):
    F.write("The time is: "+str(i+testtime)+"\n")
    # F.write(str(data[sample-1][i+testtime][:,:,0].tolist()))
    tmp = testdata[i+testtime][:,:,0].tolist()
    for j1 in range(size1):
      for j2 in range(size2):
        F.write(str(int(num_max*tmp[j1][j2]))+" ")
      F.write('\n')
    F.write('\n')
    # F.write(str(track[i+testtime][:,:,0].tolist()))
    tmp = track[i+testtime][:,:,0].tolist()
    for j1 in range(size1):
      for j2 in range(size2):
        F.write(str(int(num_max*tmp[j1][j2]))+" ")
      F.write('\n')

  F.close()
  # and compare the preditions later

def test():
  data = getData()
  cnt = 0
  # print(data[27][19][:,:,0].tolist())
  F = open("model7.out","w")
  np.set_printoptions(precision=3)
  for i in range(sample):
    F.write("Hour is : "+str(i)+'\n')
    for j in range(time):
      F.write("Date is : "+str(j)+'\n')
      tmp = data[i][j][:,:,0].tolist()
      for k1 in range(size1):
        for k2 in range(size2):
          F.write(str(tmp[k1][k2])+" ")
          cnt = cnt + tmp[k1][k2]
        F.write('\n')
  F.close()
  print("The final count is:",cnt)

if __name__ == '__main__':
  # test() 
  main()
