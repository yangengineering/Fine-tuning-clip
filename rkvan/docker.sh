docker run --runtime=nvidia -it --name houjian1 -v /data/huojian/vqa:/vqa -p 5699:91 -p 5709:41 --ipc=host --privileged=true a4a27f35555f