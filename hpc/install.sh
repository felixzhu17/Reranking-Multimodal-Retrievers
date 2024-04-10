module purge
module load slurm
module load rhel8/default-amp
module load anaconda/3.2019-10
conda create --prefix=/home/fz288/rds/hpc-work/PreFLMR/VQA python=3.8
conda activate /home/fz288/rds/hpc-work/PreFLMR/VQA
export PATH=/home/fz288/rds/hpc-work/PreFLMR/VQA/bin:$PATH
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.38.2
conda install -c pytorch faiss-gpu -y
pip install setuptools==59.5.0
pip install wandb pytorch-lightning==2.0.4 jsonnet easydict pandas scipy opencv-python fuzzywuzzy scikit-image matplotlib timm scikit-learn sentencepiece tensorboard datasets
pip install ujson evaluate GPUtil easydict peft==0.4.0
pip install bitarray spacy ujson gitpython
pip install ninja
pip install absl-py
pip install openai
pip install sacrebleu
pip install diffusers==0.20.1
pip install einops transformers_stream_generator tiktoken
cd third_party/ColBERT
pip install -e .