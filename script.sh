sudo yum update
#sudo yum install gcc g++ make automake autoconf curl-devel openssl-devel zlib-devel httpd-devel apr-devel apr-util-devel sqlite-devel
#sudo yum install python3 already in AWS AMI
sudo yum install -y python3-pip
sudo yum install -y git
sudo yum install -y nodejs

# setup pwd for new ec2: sudo passwd ec2-user
# !pip install --upgrade git+https://github.com/huggingface/diffusers.git
!pip install diffusers==0.14.0
# !pip install --upgrade git+https://github.com/huggingface/transformers/
!pip install transformers==4.27.1
!pip install accelerate==0.12.0
!pip install scipy
!pip install ftfy
!pip install translators
!pip install gradio==3.16.0
!git clone https://github.com/Work-with-Nook/nook-sd-midjourney-v4.git
%cd nook-sd-midjourney-v4

#@markdown ### ⬅️ Run this cell
#@markdown ---
#@markdown ### Install **xformers**?
#@markdown This will take an additional ~3.5 mins.<br>But images will generate 25-40% faster.
install_xformers = True #@param {type:"boolean"}

if install_xformers:
  import os
  from subprocess import getoutput

  os.system("pip install --extra-index-url https://download.pytorch.org/whl/cu113 torch torchvision==0.13.1+cu113")
  os.system("pip install triton==2.0.0.dev20220701")
  gpu_info = getoutput('nvidia-smi')
  if ('T4' in gpu_info):
    %pip install -q https://github.com/daswer123/stable-diffusion-colab/raw/main/xformers%20prebuild/T4/python39/xformers-0.0.14.dev0-cp39-cp39-linux_x86_64.whl

  elif ('P100' in gpu_info):
    %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/P100/xformers-0.0.13.dev0-py3-none-any.whl

  elif ('V100' in gpu_info):
    %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/V100/xformers-0.0.13.dev0-py3-none-any.whl

  elif ('A100' in gpu_info):
    %pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/xformers-0.0.13.dev0-py3-none-any.whl
