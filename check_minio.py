import os, boto3, json
s3 = boto3.client("s3", 
                  endpoint_url=os.environ["MINIO_ENDPOINT"], 
                  aws_access_key_id=os.environ["MINIO_ACCESS_KEY"], 
                  aws_secret_access_key=os.environ["MINIO_SECRET_KEY"])
bucket = os.environ["MINIO_BUCKET_PROCESSED"]
resp = s3.list_objects_v2(Bucket=bucket, Prefix="chunks/")

count = 0
found_any = False

for obj in resp.get("Contents", []):
    if not obj["Key"].endswith(".json"): continue
    found_any = True
    body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode("utf-8")
    
    for line in body.splitlines():
        if not line.strip(): continue
        data = json.loads(line)
        emb = data.get("embedding")
        if not emb or (isinstance(emb, list) and len(emb) == 0):
            print(f"MISSING EMBEDDING | ID: {data.get('essay_id')} | Index: {data.get('chunk_index')}")
            count += 1
        if count >= 5: break
    if count >= 5: break

if not found_any:
    print("No processed JSON files found in chunks/ yet.")
elif count == 0:
    print("Verification Successful: All checked chunks have embeddings!")
