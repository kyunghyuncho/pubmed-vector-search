try:
  import unzip_requirements
except ImportError:
  pass

import json
import boto3
import pyarrow.parquet as pq
import numpy as np

s3 = boto3.client('s3')
bucket_name = 'pubmed-abstract-vectors'
file_key = 'pubmed_embeddings.parquet'
N_CHUNKS = 100

def get_parquet_total_rows(bucket, key):
    s3_path = f's3://{bucket}/{key}'
    parquet_file = pq.ParquetFile(s3_path)
    return parquet_file.metadata.num_rows

def load_parquet_rows_from_s3(bucket, key, row_start, row_end):
    s3_path = f's3://{bucket}/{key}'
    table = pq.read_table(s3_path, filters=[('index', '>=', row_start), ('index', '<', row_end)])
    return table

def retrieve_chunk(event, context):
    try:
        body = json.loads(event['body'])
        query_vector = np.array(body['vector'])
        row_start = body['row_start']
        row_end = body['row_end']

        # Load specific rows
        table = load_parquet_rows_from_s3(bucket_name, file_key, row_start, row_end)
        vectors = np.vstack(table['vectors'].to_numpy())
        pmids = table['pmid'].to_numpy()
        normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized_query = query_vector / np.linalg.norm(query_vector)

        # Compute cosine similarity
        similarities = np.dot(normalized_vectors, normalized_query)

        # Get the top N results
        top_n_indices = similarities.argsort()[-10:][::-1]
        top_n_pmids = pmids[top_n_indices]
        top_n_similarities = similarities[top_n_indices]

        results = [{'pmid': int(pmid), 'similarity': float(sim)} for pmid, sim in zip(top_n_pmids, top_n_similarities)]
        response = {
            "statusCode": 200,
            "body": json.dumps(results)
        }
    except Exception as e:
        response = {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

    return response

def retrieve(event, context):
    try:
        body = json.loads(event['body'])
        query_vector = body['vector']
        num_chunks = N_CHUNKS  # Number of chunks to divide the data
        total_rows = get_parquet_total_rows(bucket_name, file_key)  # Automatically get total rows

        chunk_size = total_rows // num_chunks
        lambda_client = boto3.client('lambda')
        responses = []

        for i in range(num_chunks):
            row_start = i * chunk_size
            row_end = row_start + chunk_size if i < num_chunks - 1 else total_rows
            payload = {
                'vector': query_vector,
                'row_start': row_start,
                'row_end': row_end
            }
            response = lambda_client.invoke(
                FunctionName='pubmed-abstract-vectors-dev-retrieve_chunk',
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            result = json.loads(response['Payload'].read())
            responses.extend(json.loads(result['body']))

        # Combine and get top N results
        all_results = sorted(responses, key=lambda x: x['similarity'], reverse=True)[:10]
        response = {
            "statusCode": 200,
            "body": json.dumps(all_results)
        }
    except Exception as e:
        response = {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }

    return response