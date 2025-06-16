# Helm creation

<https://artifacthub.io/packages/helm/bitnami/minio>

```shell
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

## Helm configuration

Extract the default values

```shell
helm show values oci://registry-1.docker.io/bitnamicharts/minio > values.yaml
```

You should rename the file to **helm-minio-values.yaml**

## Helm installation

Example:

```shell
function prd() { LC_ALL=C tr -dc 'A-Za-z0-9!@#$%^&*()-_=+' < /dev/urandom | head -c $((12 + RANDOM % 7)) ; }
kubectl create secret generic \
    minio-secret \
    --from-literal=MINIO_ROOT_USER="minio-root-user" \
    --from-literal=MINIO_ROOT_PASSWORD="$(prd)"
```

```shell
helm upgrade --install minio bitnami/minio -f helm-minio-values.yaml
```

## Setup keys

Generate the keys then setup the `AWS_SECRET` and `AWS_SECRET` variables in your .env file.

```shell
$ kubectl exec -it doctolibminio-xxxxxx-xxx -- mc admin user svcacct add local minio-root-user --name uploaderKey --description "foobar uploader scripts"

Access Key: EIF7TF2ERAURZT3F48U3
Secret Key: F2CBFZhYM3DZi+0RxEJcVu9RYAIJyfuUl1FV24IZ
Expiration: no-expiry
```
