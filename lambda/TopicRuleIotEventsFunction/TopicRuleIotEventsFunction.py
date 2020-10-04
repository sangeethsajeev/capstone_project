import boto3
import cfnresponse

iotClient = boto3.client('iot')

def createTopicRule(ruleName, topicSql, topicDescription, inputName, roleArn):
  response = iotClient.create_topic_rule(
    ruleName=ruleName,
    topicRulePayload={
      'sql': topicSql,
      'description': topicDescription,
      'actions': [
        {
          'iotEvents': {
            'inputName': inputName,
            'roleArn': roleArn
            }
        },
        ],
      'ruleDisabled': False
    }
  )

def deleteTopicRule(ruleName):
  response = iotClient.delete_topic_rule(ruleName=ruleName)

def lambda_handler(event, context):
  ruleName = event['ResourceProperties']['RuleName']
  topicSql = event['ResourceProperties']['TopicSql']
  topicDescription = event['ResourceProperties']['TopicDescription']
  inputName = event['ResourceProperties']['InputName']
  roleArn = event['ResourceProperties']['RoleArn']
  
  if event['RequestType'] == 'Delete':
    try:
      deleteTopicRule(ruleName)
      cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
    except Exception as ex:
      print(ex)
      cfnresponse.send(event, context, cfnresponse.FAILED, {})
  else:
    physicalResourceId = ruleName
    try:
      createTopicRule(ruleName, topicSql, topicDescription, inputName, roleArn)
      response_data = {}
      cfnresponse.send(event, context, cfnresponse.SUCCESS, {}, physicalResourceId)
    except Exception as ex:
      print(ex)
      cfnresponse.send(event, context, cfnresponse.FAILED, {}, physicalResourceId)