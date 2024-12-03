FROM python:3.12.5

#chage working directory
WORKDIR /code

# add requirements file to image
COPY ./requirements.txt /code/requirements.txt

#install pyhton libraries
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# add python code
COPY ./app /code/app

# specify default commands
#CMD ["fastapi", "run", "app/main.py", "--port", "80"]

EXPOSE 8000

CMD ["uvicorn", "app.main:application", "--host", "0.0.0.0", "--port", "8000"]