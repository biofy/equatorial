# Earth-2 2025 DLI

Example for starting the environment:

```
docker run --rm -d --shm-size 2gb -v <your-container-mounts> -p 8889:8889 --gpus all nvcr.io/nvidia/physicsnemo/physicsnemo:25.03 bash run.sh
```

You can also prebuild the image using the Dockerfile. To reuse the cache across multiple sessions, set `EARTH2STUDIO_CACHE` to a mounted directory, like in the demo `.env` file.
