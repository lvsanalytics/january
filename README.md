# January - A Virtual Slack Assistant

January is Jeff's personal digital therapy friend. She is programmed to talk to him and provide emotional support and wellbeing.

# Installation Notes

## Ollama -  Instructions

Ideally Docker would run as a container, however I keep running into issues where it won't use the GPU.

It is important to bind Ollama to all interfaces so that it is accessible within the container.

1. Create a Service Override
```
sudo mkdir -p /etc/systemd/system/ollama.service.d/
```

- This creates a directory (-p any parent directories if they don't exist)
- This directory is for systemd override files. When systemd runs it will check in here for any additional configuration data. In this case ollama.service

2. Create Override Configuration File


```
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
```

- printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"\n': This creates a string that specifies a new environment variable for the Ollama service:
    - [Service]: This section indicates that the following configuration applies to the service itself.
    - Environment="OLLAMA_HOST=0.0.0.0": This sets the OLLAMA_HOST environment variable to 0.0.0.0. This instructs Ollama to bind to all network interfaces (localhost and the VM’s external IP).
- sudo tee /etc/systemd/system/ollama.service.d/override.conf:
    - tee writes the output from printf to the specified file (in this case, /etc/systemd/system/ollama.service.d/override.conf), while also displaying it on the terminal.
    - The file being written is the override configuration file for the Ollama service, and it will contain the environment variable configuration.

3. Apply Changes

```
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

# Building and Running January

## Create a Slack Application

Go to https://api.slack.com and create a bot application for the Workspace. You need to setup an APP Key and a Bot Key with the necessary permissions. Then the keys will need to be added to an environment file.

## .ENV File
You will need to create a .env file with the following variables

```
STORAGE_PATH = /chroma/chroma
SLACK_BOT_TOKEN=xoxb-XXXX
SLACK_APP_TOKEN=xapp-XXXX
```

## Docker Create Image
The following command will create the image

```
sudo docker build -t january .
```

## Docker Compose File

The following can be added to a Docker compose file

```
  january:
    image: january  # The January service image
    container_name: january
    network_mode: host
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - /home/aibox/storage/chroma_db:/chroma/chroma
    restart: always
```

# Troubleshooting

If you recieve errors accessing Ollama you can try installing a Docker container and using either sh or bash to run networking commands.

```
docker run -itd --rm --network host --name testerman ubuntu
sudo docker exec -it testerman bin/bash
```

When in the ubuntu container

```
apt update && apt upgrade
apt install curl

curl http://localhost:11434
```

You should see that Ollama is accessible. This will also work with http://0.0.0.0:11434 and http://127.0.1.0:11434

```
Ollama is runningroot@host
````
