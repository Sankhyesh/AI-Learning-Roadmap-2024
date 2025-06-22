@echo off
REM Check if a container named spark-notebook already exists
for /f "tokens=*" %%i in ('docker ps -a --format "{{.Names}}"') do (
    if "%%i"=="spark-notebook" (
        echo Container 'spark-notebook' already exists. Stopping and removing it...
        docker stop spark-notebook
        docker rm spark-notebook
    )
)

REM Run the container with GPU support and mount the current directory as /home/jovyan/work
REM Also mount the Kaggle API credentials to /home/jovyan/.kaggle inside the container
docker run --gpus all -p 8888:8888 -p 4040:4040 ^
    -v "%cd%":/home/jovyan/work ^
    -v "C:\Users\sankh\.kaggle":/home/jovyan/.kaggle ^
    --name spark-notebook jupyter/pyspark-notebook

pause
