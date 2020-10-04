#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function usage {
    echo ""
    echo "Usage: $(basename "$0") [<units>] [<options>]"
    echo ""
    echo "uploads files from local to aws s3"
    echo "creates CFN for the iot app"
    echo ""
 
}

function fail {
    echo "ERROR: $1"
    exit 1
}

function on_exit {
    EXIT_CODE=$?
    [ $EXIT_CODE -eq 0 ] || echo -e "\nERROR: Encountered an error. See logs for details."
    exit $EXIT_CODE
}
trap on_exit EXIT

# Process script parameters
while test $# -gt 0; do
    case "$1" in
        -h|--help)
            usage; exit 0
            ;;
        --aws-profile)
            shift; DEPLOYMENT_PROFILE=$1; shift
            ;;
        --deployment-region)
            shift; DEPLOYMENT_REGION=$1; shift
            ;;
        --source-bucket) 
            shift; DEPLOYMENT_BUCKET=$1; shift
            ;;
        --deployment-stack-name)
            shift; PARENT_STACK_NAME=$1; shift
            ;;
        --instance-type)
            shift; DEPLOYMENT_INSTANCE_TYPE=$1; shift
            ;;
    esac
done


if [ ! -z "$DEPLOYMENT_PROFILE" ]; then
    AWS_CLI_ARGS="--profile $DEPLOYMENT_PROFILE"
fi

echo "DEPLOYMENT_BUCKET=${DEPLOYMENT_BUCKET}"
echo ""

function package() {
  LAMBDA_FUNCTION_NAME=$1

  echo "Packaging Lambda function $LAMBDA_FUNCTION_NAME"

  cd $DIR/../lambdas/$LAMBDA_FUNCTION_NAME
  PACKAGE_NAME=lambda-$LAMBDA_FUNCTION_NAME.zip
  PACKAGE_PATH="/tmp/packages/$PACKAGE_NAME"

  echo "Creating package $PACKAGE_NAME"

  1>/dev/null zip -9 -r $PACKAGE_PATH * -x \*.pyc \*.md \*.zip \*.log \*__pycache__\* \*.so
  echo ""
}

LAMBDA_FUNCTION_NAMES="TopicRuleIotEventsFunction IotCertificateKeyCreatorFunction InferenceLambda DataIngestorLambda"

rm -rf /tmp/packages
mkdir -p /tmp/packages

for LAMBDA_FUNCTION_NAME in $LAMBDA_FUNCTION_NAMES; do
  package $LAMBDA_FUNCTION_NAME $DEPLOYMENT_BUCKET
done

aws s3 $AWS_CLI_ARGS --region $DEPLOYMENT_REGION cp /tmp/packages s3://$DEPLOYMENT_BUCKET/lambdaCode/ --recursive --sse AES256
rm -rf /tmp/packages

echo "Uploading CloudFormation templates to $DEPLOYMENT_BUCKET/cloudformation"
aws s3 $AWS_CLI_ARGS --region $DEPLOYMENT_REGION cp $DIR/../cloudformation s3://$DEPLOYMENT_BUCKET/cloudformation --recursive --sse AES256

echo "Uploading pollution.csv to <Bucket>/data/"
aws s3 $AWS_CLI_ARGS --region $DEPLOYMENT_REGION cp $DIR/../data/pollution.csv s3://$DEPLOYMENT_BUCKET/data/pollution.csv --sse AES256

# Upload Sagemaker notebook files
echo "Upload Sagemaker notebook files to S3"

aws s3 $AWS_CLI_ARGS \
    --region $DEPLOYMENT_REGION \
    cp $DIR/../sagemaker s3://$DEPLOYMENT_BUCKET/sagemaker/ \
    --recursive \
    --sse AES256 
echo ""
echo "Sagemaker File Upload done"

echo "Creating stacks"
aws cloudformation $AWS_CLI_ARGS \
    "create-stack --on-failure DELETE" \
    --region $DEPLOYMENT_REGION \
    --stack-name "$PARENT_STACK_NAME" \
    --template-url "https://$DEPLOYMENT_BUCKET.s3.$DEPLOYMENT_REGION.amazonaws.com/cloudformation/deploy-app.yaml" \
    --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
    --parameters \
    "ParameterKey=SourceS3Bucket,ParameterValue=$DEPLOYMENT_BUCKET" \
    "ParameterKey=InstanceType,ParameterValue=$DEPLOYMENT_INSTANCE_TYPE" \
    --output text

echo ""

echo "App resource creation initiated, monitor CloudFormation for stacks to finish."