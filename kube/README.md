# How to deploy K8S

> Note: the following variables will be used in this document:

```shell
ENVFILE=".env"
KUBECMD="kubectl --kubeconfig=${HOME}/config.yaml"
```

## secret env file

Create/Update the secret hosting all your environment variables:

```shell
(
  set -a ; source "${ENVFILE}" ; set +a
  envsubst < "${ENVFILE}" > "${ENVFILE}".expanded
  sed -i -E "s/^([^=]+)=['\"](.*)['\"]$/\1=\2/" "${ENVFILE}.expanded"
  ${KUBECMD} create secret generic env-secret --from-env-file="${ENVFILE}.expanded" --dry-run=client -o yaml| ${KUBECMD} apply -f -
  rm -f "${ENVFILE}.expanded"
)
```
