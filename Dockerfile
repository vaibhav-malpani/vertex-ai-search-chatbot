FROM python:3.11
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY . /app
WORKDIR /app/
EXPOSE 8000
CMD ["chainlit", "run", "chainlit_main.py", "--host", "0.0.0.0", "--port", "8000"]