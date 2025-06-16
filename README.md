# in-house-speech-to-text-solution

## Technical choices

* Whisper (openai-whisper) transcript created as it was specifically requested
* Whisperx transcript has also been created instead of whisper to include diarization and better timestamping
* Whisperx has a GPU and CPU support implemented
* Python Boto3 to use an S3 compatible solution (aws s3, Google Storage, Minio)

Additional choices

* Python 3.11 to remains compatible with the current prodution platform
* Uv to manage python dependencies and project
* Ruff as a python Linter

## Main challenges to be addressed

### Time spent on this project (1 day)

* Lot of time was spent finding the *libcudnn_ops_infer.so.8* library for Fedora.
* Diarization was not working out of the box, mismatch between pandas dataframe and dictionnary format.

### What would architecture design be ?

#### Components

* A storage solution (Minio) to host audio samples and text outputs.
* One queuing system (Redis) to manage the FIFO operations.
* One API service to ease storing/managing audio samples and manage the FIFO queuing system.
* A stateless and scalable Whisper service, which will poll regularly the queue for an audio sample available for transcript.

#### Infrastructure

Hosting system

* Docker to host containers
* Docker Compose to build and push images
* Kubernetes to deliver the services

Hosting components

* One CPU VM with storage (Minio) and memory (Redis)
* If possible, one or more GPU - L40S should be enough, depending on the model size - to run the transcripts

### How to build a scalable inference code (python / libs) ?

Adding more stateless Whisper services permits to transcript more and more audio samples, they need to access the queuing system.

### How to containerize your code (docker) and how to serve it scalably ?

To containerize the code:

```shell
docker compose build --push
```

To scale the code

vertical way:

```shell
docker compose up --scale process_queue=<Nb Instance> -d
```

Using Kubernetes:

* Vertical Pod Autoscaling for immediate performances based on CPU.
* Horizontal Pod Autoscaling if raw GPU performances are needed.

### How to build a scalable infrastructure with terraform ? What is the deployment process ?

While I already created clusters using Terraform, I do need more practicing to be very efficient.

To answer the question, I would say that we need to focus on modularity for the deployment files, and variables to change the number of instances.

Something like:

```txt
resource "aws_autoscaling_group" "app" {
  min_size         = var.min_instances
  max_size         = var.max_instances
  desired_capacity = var.desired_instances
  
  target_group_arns = [aws_lb_target_group.app.arn]
  
  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
}
```

PS: do not forget to add a label for the GPU nodes, so K8S can use affinity on it

```txt
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.my_cluster.name
  node_group_name = "gpu-node-group"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = module.vpc.private_subnets
  instance_types  = ["g4dn.xlarge"]

  labels = {
    accelerator = "nvidia"
  }

  scaling_config {
    desired_size = 1
    max_size     = 2
    min_size     = 1
  }

  ami_type       = "AL2_x86_64_GPU" # Important pour drivers NVIDIA
  capacity_type  = "ON_DEMAND"
}
```

Based on healthchecks, alarms and dashboards, we can manage running less or more instances dynamically.

### What infrastructure issues need to be addressed ?

* cost for building unused resources.
* instant scaling requirements.

### How to monitor the performance of the solution ?

Using Prometheus, we can expose the time to render and other metrics

### What are the limitations and possible improvements for your solution ?

* Managing multiple audio languages in the same audio sample can lead to transcription errors.
* This product will work asynchronously only, it is not expected to be used with a stream.
* Add non root user to containers.
* Manage local temp dir.
* Change *process_file* image to use smaller images like alpine (no need to cuda).
* A page to List and Clear redis queue.
* Github workflow is not working (Python crash, probably a memory issue), need to investigate.
* Prometheus metrics should be added to the file processing
