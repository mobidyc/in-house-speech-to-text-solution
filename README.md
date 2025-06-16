# in-house-speech-to-text-solution

## Technical choices

* Whisper (openai-whisper) transcript created as it was specically requested
* Whisperx transcript also created insted of whisper to include diarization and better timestamping
* Whisperx has a GPU and CPU support implemented
* Python Boto3 to use an S3 compatible solution (aws s3, Google Storage, Minio)

Additional choices

* Python 3.11 to remains compatible with the current prodution platform
* Uv to manage python dependencies and project
* Ruff as a python Linter

## Main challenges to be addressed

### What would architecture design be ?

#### Components

* A storage solution (Minio) to host audio samples and text outputs.
* One queuing system (Redis) to manage the FIFO operations.
* One API service to ease storing/managing audio samples and manage the FIFO queuing system.
* A stateless and scalable Whisper service, which will poll regularly in the queue if an audio sample is available to transcript it.

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

To containeriez the code:

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

While I already Created and deployed K8s clusters using Terraform, I do need more practicing to be very efficient.

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

Based on healthchecks, alarms and dashboards, we can manage running less or more instances dynamically.

### What infrastructure issues need to be addressed ?

* cost for building unused resources.
* instant scaling requirements.

### How to monitor the performance of the solution ?

Using Prometheus, we can expose the time to render and other metrics

### What are the limitations and possible improvements for your solution ?

* Managing multiple audio languages in the same audio sample can lead to transcription errors.
* This product will work asynchronously only, it is not expected to be used with a stream.
* This transcript could be translated to a defined language if needed.
* We need a better naming and sharding to store files
* Add non root user to containers
* Manage local temp dir
* Change process_file image to use lighter images like alpine (no need to cuda)
* Clear redis queue if needed
