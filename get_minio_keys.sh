#!/bin/bash

ctl=
if [ -z "$1" ]; then
  echo "Usage: $0 <docker|kubectl> [<minio_container_name>]"
  echo "Example: $0 docker doctolib_minio"
  exit 1
else
    if [ "$1" = "docker" ]; then
        ctl="docker exec -it ${2:-doctolib_minio}"
    elif [ "$1" = "kubectl" ]; then
        ctl="kubectl exec -it ${2:-doctolib_minio} --"
    else
        echo "Unsupported command: $1. Use 'docker' or 'podman'."
        exit 1
    fi
fi

echo $ctl mc mb --ignore-existing local/audio
$ctl mc mb --ignore-existing local/audio

echo $ctl mc mb --ignore-existing local/audio
$ctl mc mb --ignore-existing local/audio

echo $ctl mc admin user svcacct list local/ minio-root-user
$ctl mc admin user svcacct list local/ minio-root-user

echo $ctl mc admin user svcacct add local  minio-root-user --name uploaderKey
$ctl mc admin user svcacct add local minio-root-user --name uploaderKey
