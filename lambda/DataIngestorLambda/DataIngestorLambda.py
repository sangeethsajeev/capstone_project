import greengrasssdk
from pandas import read_csv
import json
import time

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')
data_dir = '/dest/'

def publishToIotCloud(train_dataset):
    for i in range(len(train_dataset)):
        current_row = train_dataset[i:i+1]
        data = current_row.to_dict(orient='records')[0]
        payload = json.dumps(data)
        response = client.publish(
            topic='pollution/data',
            payload=payload
        )
        time.sleep(0.1)

def lambda_handler(event, context):
    dataset = read_csv(data_dir + 'pollution.csv', header=0)
    dataset.dropna(inplace=True)
    publishToIotCloud(dataset)
    return