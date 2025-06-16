# Helm creation

<https://artifacthub.io/packages/helm/bitnami/redis>

```shell
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

## Helm configuration

Extract the default values

```shell
helm show values oci://registry-1.docker.io/bitnamicharts/redis > values.yaml
```

You should rename the file to **helm-redis-values.yaml**

## Helm installation

```shell
helm upgrade --install doctolib_redis bitnami/redis -f helm-redis-values.yaml
kubectl apply -f service-doctolib_redis.yaml
```

## monitoring

<https://min.io/docs/minio/linux/operations/monitoring/collect-minio-metrics-using-prometheus.html?ref=docs-redirect>
