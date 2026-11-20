# Lambda container for CPU inference: a FastAPI app served by uvicorn. The AWS Lambda Web
# Adapter (an extension, inert outside Lambda) turns Function URL requests into plain HTTP
# calls to it, so the same image also runs anywhere with `docker run -p 8080:8080`.
# Expects ./model/{model.onnx,labels.json} in the build context (CI pulls them from S3;
# locally: python -m slugid.export --out model).
FROM public.ecr.aws/docker/library/python:3.12-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter

RUN pip install --no-cache-dir onnxruntime==1.32.0 numpy==2.5.4 pillow==13.0.0 \
    fastapi==0.165.0 uvicorn==0.54.0 python-multipart==0.0.34 boto3==1.44.8

COPY model/ /opt/model/
WORKDIR /app
COPY src/slugid/__init__.py src/slugid/serve.py src/slugid/api.py src/slugid/index.html slugid/

ENV MODEL_DIR=/opt/model \
    AWS_LWA_PORT=8080 \
    AWS_LWA_READINESS_CHECK_PATH=/health
CMD ["uvicorn", "slugid.api:app", "--host", "0.0.0.0", "--port", "8080"]
