# OrcaDetector setup

This UC Berkeley Master of Information in Data Science capstone project was developed by
[Spyros Garyfallos](https://github.com/paloukari), [Ram Iyer](https://github.com/ram-iyer), and [Mike Winton](https://github.com/mwinton).

## 0. Provision a cloud GPU machine

### Using AWS

If using AWS, as assumed by these setup instructions, provision an Ubuntu 18.04 `p2.xlarge` instance.  It's got older GPUs (Tesla K80) but is much cheaper.  Make sure to upgrade the storage space (e.g. 500 GB).  Also, make sure to pick a prebuilt Deep Learning AMI during the first step of the provisioning process. The most current version as of writing is `Deep Learning AMI (Ubuntu) Version 23.0 - ami-058f26d848e91a4e8`. This will already have `docker` and `nvidia-docker` pre-installed and will save you a lot of manual installation "fun".

## 1. Set up required environment variables - UPDATE

Set the environment variable for MLFlow in your system's `~/.bashrc` file (or on a Mac, in your `~/.bash_profile` file):

```
TBD
```

After saving the file, type `source ~/.bashrc` to load the new variables into your environment.


## 2. Clone the project repo

If you haven't already, clone the project Git repo to your instance.  Doing so in your home directory is convenient, and this document assumes you have done so.

```
cd ~
git clone https://github.com/paloukari/OrcaDetector
```

## 3. Download data files and model weights

First, if you don't already have the AWS CLI installed, follow the instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html).

### Marine mammal audio samples

We have downloaded the audio files from the [Watkins Marine Mammal Sound Database](https://cis.whoi.edu/science/B/whalesounds/fullCuts.cfm) using the [/orca_detector/getData.py](./orca_detector/getData.py) script, based on a sample provided to us by Watkins.

Expand `data.tar.gz` inside the main directory of the repo; it will create a `/data` folder.  (There are `.gitignore` rules in place to prevent it from accidentally being submitted to GitHub.)

```
cd ~/OrcaDetector
aws s3 cp s3://w251-orca-detector-data/data.tar.gz .
tar -xvf data.tar.gz
```

Alternately you can download [data.zip](https://drive.google.com/file/d/10mGIptby8SEf4yk0m57mVtiXCgRQrgTc/view?usp=sharing) from Google Drive, but due to the size of the archive, it's probably much faster to pull it from the S3 bucket.

### Pretrained weights (for VGGish Keras model)

Download and extract `vggish_weights.tar.gz ` inside the main directory of the repo; it will create a `/vggish_weights` folder containing VGGish model weights that were pretrained on the [AudioSet](https://research.google.com/audioset/index.html) dataset, which contains audio from 2 million human-labeled 10-second YouTube clips.

```
cd ~/OrcaDetector
aws s3 cp s3://w251-orca-detector-data/vggish_weights.tar.gz .
tar -xvf vggish_weights.tar.gz
```

## 4. Create the `orca_dev` Docker image

### GPU OPTION: Build our `orca_dev` base Docker image

In the project repo, `cd` into the `orca_detector` directory:

```
cd ~/OrcaDetector/orca_detector
```

Build the Docker image (this will take a while):

```
sudo docker build -t orca_dev -f Dockerfile.dev .
```

## 5. Launch an `orca_dev` Docker container

Run the `orca_dev` Docker container with the following args.  

> NOTE: update the host volume mappings (i.e. `~/OrcaDetector`) as appropriate for your machine in the following script:

```
sudo docker run \
    --rm \
    --runtime=nvidia \
    --name orca_dev \
    -ti \
    -e JUPYTER_ENABLE_LAB=yes \
    -v ~/OrcaDetector/orca_detector:/src \
    -v ~/OrcaDetector/data:/data \
    -v ~/OrcaDetector/vggish_weights:/weights \
    -v ~/OrcaDetector/results:/results \
    -p 8888:8888 \
    -p 4040:4040 \
    orca_dev
```

You will see it listed as `orca_dev ` when you run `docker ps -a`.  

### Verify Keras can see the GPU

Once inside the container, try running:

```
nvidia-smi
```

### Verify that the `vggish` Keras model builds

Once inside the container, the following script should run to make sure we can instantiate the Keras model and load it's pretrained weights:

```
python3 vggish_model.py
```

If it was successful, you should see a Keras model summary.


### (REMOVE?) Verify that the `vggish` TF smoke test runs

Once inside the container, the following script should run an end-to-end check of the `vggish` pipeline:

```
python3 vggish/vggish_smoke_test.py
```

### (OPTIONAL) Launch Jupyter Lab in the container

After you've started the container as described above, if you want to _also_ open a Jupyter notebook (e.g. for development/debugging), issue this command:

```
docker exec -it orca_dev bash
```

Then in the Docker shell that opens, type:

```
jupyter lab --allow-root --port=8888 --ip=0.0.0.0
```

Then go to your browser and enter:

```
http://127.0.0.1:8888?token=<whatever token got displayed in the logs>
```


## 6. Train the OrcaDetector

### Training

TBD

### Testing

TBD
