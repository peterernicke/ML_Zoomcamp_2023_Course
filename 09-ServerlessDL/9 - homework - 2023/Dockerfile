FROM agrigorev/zoomcamp-bees-wasps:v2
#FROM public.ecr.aws/lambda/python:3.10
#COPY bees-wasps-v2.tflite .

#RUN pip install keras-image-helper
#RUN pip install tflite-runtime
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install pillow
                

#COPY clothing-model.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]