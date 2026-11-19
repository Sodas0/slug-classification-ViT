# Lambda container for CPU inference. Expects ./model/{model.onnx,labels.json} in the build
# context (CI pulls them from S3; locally: python -m slugid.export --out model).
FROM public.ecr.aws/lambda/python:3.12

RUN pip install --no-cache-dir onnxruntime==1.32.0 numpy==2.5.4 pillow==13.0.0

COPY model/ /opt/model/
COPY src/slugid/__init__.py src/slugid/serve.py src/slugid/index.html ${LAMBDA_TASK_ROOT}/slugid/

ENV MODEL_DIR=/opt/model
CMD ["slugid.serve.handler"]
