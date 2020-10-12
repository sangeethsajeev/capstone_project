import boto3
import cfnresponse
from zipfile import ZipFile
import os
import json

s3Client = boto3.client('s3')
iotClient = boto3.client('iot')
tmpDir = "/tmp/"
zipFileName = "certs.zip"
configName = "config.json"
certPemName = "cert.pem"
publicKeyName = "public.key"
privateKeyName = "private.key"

THING_ARN = os.environ.get('THING_ARN')

def createConfig(iotEndpoint):
  config = {
            "coreThing" : {
              "caPath" : "root.ca.pem",
              "certPath" : "cert.pem",
              "keyPath" : "private.key",
              "thingArn" : THING_ARN,
              "iotHost" : "{iotendpoint}",
              "ggHost" : "greengrass-ats.iot.{}.amazonaws.com".format(AWS_REGION),
              "keepAlive" : 600
            },
            "runtime" : {
              "cgroup" : {
                "useSystemd" : "yes"
              }
            },
            "managedRespawn" : False,
            "crypto" : {
              "principals" : {
                "SecretsManager" : {
                  "privateKeyPath" : "file:///greengrass/certs/private.key"
                },
                "IoTCertificate" : {
                  "privateKeyPath" : "file:///greengrass/certs/private.key",
                  "certificatePath" : "file:///greengrass/certs/cert.pem"
                }
              },
              "caPath" : "file:///greengrass/certs/root.ca.pem"
            }
          }
  config['coreThing']['iotHost'] = iotEndpoint
  return config

def saveStringToFile(stringValue, fileName):
  text_file = open(fileName, "w")
  text_file.write(stringValue)
  text_file.close()

def saveCertsAndKeys(response, iotEndpoint):
  config = createConfig(iotEndpoint)
  saveStringToFile(json.dumps(config), tmpDir + configName)
  saveStringToFile(response['certificatePem'], tmpDir + certPemName)
  saveStringToFile(response['keyPair']['PublicKey'], tmpDir + publicKeyName)
  saveStringToFile(response['keyPair']['PrivateKey'], tmpDir + privateKeyName)

  file_paths = [tmpDir+certPemName, tmpDir+publicKeyName, tmpDir+privateKeyName, tmpDir+configName]
  # writing files to a zipfile
  with ZipFile(tmpDir + zipFileName,'w') as zip:
    for file in file_paths:
      zip.write(file)

def deleteCerts(certificateId):
  iotClient.update_certificate(certificateId=certificateId,newStatus='INACTIVE')
  iotClient.delete_certificate(certificateId=certificateId)

def uploadTos3(s3Bucket, objectPrefix):
  response = s3Client.upload_file(tmpDir + zipFileName, s3Bucket, objectPrefix + "/" + zipFileName)

def deleteS3Object(s3bucket, s3objectPrefix):
  s3Client.delete_object(Bucket = s3bucket, Key = s3objectPrefix + "/" + zipFileName)

def lambda_handler(event, context):
  s3bucket = event['ResourceProperties']['S3Bucket']
  s3objectPrefix = event['ResourceProperties']['S3ObjectPrefix']
  if event['RequestType'] == 'Delete':
    try:
      deleteS3Object(s3bucket, s3objectPrefix)
      certificateId = event['PhysicalResourceId']
      if certificateId != '' :
        deleteCerts(certificateId)
        cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
    except Exception as ex:
      print(ex)
      cfnresponse.send(event, context, cfnresponse.FAILED, {})
  else:
    physicalResourceId = ''
    try:
      iotEndpoint = iotClient.describe_endpoint(endpointType="iot:Data-ATS")['endpointAddress']
      response = iotClient.create_keys_and_certificate(setAsActive=True)
      physicalResourceId = response['certificateId']
      saveCertsAndKeys(response, iotEndpoint)
      uploadTos3(s3bucket, s3objectPrefix)
      response_data = {'Arn': response['certificateArn']}
      cfnresponse.send(event, context, cfnresponse.SUCCESS, response_data, physicalResourceId)
    except Exception as ex:
      print(ex)
      cfnresponse.send(event, context, cfnresponse.FAILED, {}, physicalResourceId)
