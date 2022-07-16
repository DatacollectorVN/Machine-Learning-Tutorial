FROM python:3.8
WORKDIR /project
COPY . /project
RUN pip install -r requirements.txt
RUN ["chmod", "+x", "/project/setup_data.sh"]
RUN ["bash", "setup_data.sh"]
RUN ["mkdir", "experiments"]
CMD ["python", "train.py"]
