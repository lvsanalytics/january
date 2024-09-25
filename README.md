# h1 January - A Virtual Slack Assistant

January is Jeff's personal digital therapy friend. She is programmed to talk to him and provide emotional support and well being.

# h1 Installation Notes

# h2 Ollama -  Instructions

Ideally Docker would run as a container, however I keep running into issues where it won't use the GPU.

It is important to open up Ollama on all 


1. Create a Service Override
```
sudo mkdir -p /etc/systemd/system/ollama.service.d/

```

- This creates a directory (-p any parent directories if they don't exist)
- This specific directory is specifically for systemd override files. When systemd runs it will check in here for any additional configuration data, specifically ollama.service

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
