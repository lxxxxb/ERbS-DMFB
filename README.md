# Efficient Routing-based Synthesis for Digital Microfluidic Biochips via Reinforcement Learning
This is the code for our work "Efficient Routing-based Synthesis for Digital Microfluidic Biochips via Reinforcement Learning". Biao Liu, Chen Jiang, Qi Xu, Hailong Yao, Tsung-Yi Ho and Bo Yuan.
## Environment
To build the environment, you can follow the code below,

```conda env create -f environment.yml```
## Usage
To train the agent, you can run the code,

```CUDA_VISIBLE_DEVICES=0 python -u multiTrain.py dmfb --task=1 -m=10 --net=crnn --oc=6```

To evaluate on the real-world assays, you can run the code,

```python process_assay.py dmfb -w=set_width -l=set_length --load_model_name2=your_saved_model_name --model_dir=your/model/path --oc=6 --evaluate_task=5```
## Chip Visualization
![Markdown Logo](https://github.com/lxxxxb/ERbS-DMFB/blob/main/chip_visualize.png)
