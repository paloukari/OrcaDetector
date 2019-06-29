ufw allow 32001/tcp

apt-get install -y awscli
aws configure set aws_access_key_id $1
aws configure set aws_secret_access_key $2

aws s3 cp s3://w251-orca-detector-data/data.tar.gz ./
tar -xvf ./data.tar.gz -C ./
rm ./data.tar.gz

aws s3 cp s3://w251-orca-detector-data/vggish_weights.tar.gz ./
tar -xvf ./vggish_weights.tar.gz -C ./
rm ./vggish_weights.tar.gz

docker build -t orca_dev -f ./orca_detector/Dockerfile.dev ./orca_detector
