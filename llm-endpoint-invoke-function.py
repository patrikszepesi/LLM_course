import json
import boto3

def lambda_handler(event, context):

    sagemaker_runtime = boto3.client('sagemaker-runtime')

    body = json.loads(event['body'])

    headline = body['query']['headline'] # new vaccines discovered

    #headline = "How I met your Mother voted as best sitcom in Europe"

    endpoint_name = 'your-endpoint-name'

    payload = json.dumps({"inputs": headline})

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName = endpoint_name,
        ContentType = "application/json",
        Body = payload
        )
    result = json.loads(response['Body'].read().decode())

    return {

        'statusCode':200,
        'body': json.dumps(result)
    }
