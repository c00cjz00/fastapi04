# fastapi01
## Build image
```
docker build -t fastapi .
```

## Start the Docker Container
```
docker run -d --name fastapi -p 9080:80 fastapi

```

## Check it
Now you can go to  http://127.0.0.1:9080/docs (or equivalent, using your Docker host).
