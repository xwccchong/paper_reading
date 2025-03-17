# OpenVLA

## related work
- Visually-Conditioned Language Models

    VLM大力发展的基础：model architectures that bridge features from pretrained vision encoders [8, 9, 25] with pretrained language models [10, 23, 34–36], directly building on advances in both computer vision and natural language modelling to create powerful multimodal models.

    新型的VLM：have converged on a simpler “patch-as-token” approach, in which patch features from pretrained visual transformers are treated as tokens, and are then projected into the input space of a language model

- Generalist Robot Policies

    模型训练给予大量多样的机器人数据进行训练，但是 Prior works like Octo typically compose pretrained components such as language embeddings or visual encoders with additional model components initialized from scratch；
    而OpenVLA则采用端到端的方式，直接冻结视觉，语言模块，直接对VLMs进行微调

- Vision-Language-Action Models

    直接对VLMs进行微调的好处：
    1. it performs alignment of pretrained vision and language components on a large, Internet-scale vision-language dataset
    2. the use of a generic architecture, not custom-made for robot control, allows us to leverage the scalable infrastructure underlying modern VLM training [75–77] and scale to training billion-parameter policies with minimal code modifications
    3. it provides a direct pathway for robotics to benefit from the rapid improvements in VLMs（废话

## model
### Vision-Language Models
一般由三个部份组成：
1. visual encoder that maps image inputs to a number of “image patch embeddings”
2. projector that takes the output embeddings of the visual encoder and maps them into the input space of a language model
3. a large language model (LLM) backbone

OpenVLA中使用*Prismatic-7B VLM*作为backbone，对应的三个部份分别为：
1. 600M-parameter visual encoder
2. a small 2-layer MLP projector
3. a 7B-parameter Llama 2 language model backbone

uses a two-part visual encoder, consisting of pretrained SigLIP [79] and DinoV2 [25] models.
the addition of DinoV2 features has been shown to be helpful for improved spatial reasoning [44], which can be particularly helpful for robot control.
*对于机器人来说，空间推理能力是关键，所以3D方向应该是一个必然的热点* <font color=red>OpenVLA在视觉编码部分增强空间理解能力；对于pointVLA来说，则是在微调的部分使用3d数据，直接传入的就是3d的视觉信息</font>

SigLIP, DinoV2, and Llama 2 do not release details about their training data, which likely consists of
trillions of tokens of Internet-sourced image-text, image-only, and text-only data respectively. The
Prismatic VLM is fine-tuned on top of these components using the LLaVA 1.5 data mixture
openVLA在此基础上进一步对模型使用多模态的数据集微调模块。

### OpenVLA Training Procedure
为了适配VLM，将连续的机器人动作映射为离散的tokens，能够被语言模型的tokenizer使用。
- 具体操作：

    discretize each dimension of the robot actions separately into one of 256 bins, For each action dimension, we set the bin width to uniformly divide the interval between the 1st and 99th quantile of the actions in the training data.（对机器人动作的每一个维度单独进行离散化；使用分位数来离散区间范围，避免过于边缘的数值造成影响）

    离散之后获取N维的[0,255]的离散tokens，由于Llama的tokenizer在微调的时候只能保留100个指定的tokens，所以考虑对llama语料库中最不常用的256个tokens替换为**action的tokens**。

### Training Data
Open X-Embodiment dataset [1] (OpenX) as a base to curate our training dataset
后续处理：
1. a coherent input and output space across all training datasets（*restrict our training dataset to contain only manipulation datasets with at least one 3rd person camera and use single-arm end-effector control*）
2. a balanced mix of embodiments, tasks, and scenes in the final training mixture.（*we leverage the data mixture weights of Octo [5] for all datasets that pass the first round of filtering. Octo heuristically down-weights or removes less diverse datasets and up-weights datasets with larger task and scene diversity*）

对于DROID数据集：openvla在训练的时候没有使用（octo使用）。可能由于该数据集多样性程度过高，在当前模式下难以更好地学习，与其他数据集质量不同（高）可能破坏分布。  加了这个数据集训练会导致训练不稳定。

### OpenVLA Design Decisions
- VLM Backbone

    The fine-tuned Prismatic VLM policy achieved further improvements, outperforming the LLaVA policy by roughly 10% in absolute success rate across both simple single-object tasks and multi-object, language grounding tasks. We attribute this performance delta to improved spatial reasoning capabilities afforded by the fused SigLIP-DinoV2 backbones；Prismatic also provides a modular and easy-to-use codebase

- Image Resolution（224 × 224px）

    We compared VLAs with 224 × 224px and 384 × 384px inputs, but found no performance difference in our evaluations, while the latter takes 3x longer to train.（但是在部分vla中，增大分辨率会提升性能）

- Fine-Tuning Vision Encoder

    However, we found fine-tuning the vision encoder during VLA training to be crucial for good VLA performance， pretrained vision backbone may not capture sufficient fine-grained spatial details about important parts of the scene to enable precise robotic control（预训练好的模块可能无法捕捉细粒度的信息）

- Training Epochs

    过去的VLA一般最多训练1-2个epoch，但是openvla发现当准确率超过95%时，效果进一步提升，所以训练27个epoch

- Learning Rate （2e-5）

    We did not find learning rate warmup to provide benefits.

### Infrastructure for Training and Inference
trained on a cluster of 64 A100 GPUs for 14 days, or a total of 21,500 A100-hours, using a batch size of 2048

During inference, OpenVLA requires 15GB of GPU memory when loaded in bfloat16 precision (i.e., without quantization) and runs at approximately 6Hz on one NVIDIA RTX 4090 GPU (without compilation, speculative decoding, or other inference speed-up tricks)  *可以进行量化，不会影响性能* 

## experiment
####  eval
- visual (unseen backgrounds, distractor objects, colors/appearances of objects)
- motion (unseen object positions/orientations)
- physical (unseen object sizes/shapes)
- semantic (unseen target objects, instructions, and concepts from the Internet)

### Direct Evaluations on Multiple Robot Platforms
we evaluated each method in 170 rollouts (17 tasks with 10 trials each) for BridgeData V2 experiments and 60 rollouts (12 tasks with 5 trials each) for Google robot experiments

### Data-Efficient Adaptation to New Robot Setups
full fine-tuning of all model parameters, using small datasets with 10–150 demonstrations of a target task；Franka-Tabletop, a stationary, table-mounted Franka Emika Panda 7-DoF robot arm;
and Franka-DROID, the Franka robot arm setup from the recently released DROID dataset;
The setups use 5Hz and 15 Hz non-blocking controllers

- **We also compare to Diffusion Policy (matched), a version of Diffusion Policy that matches the input and output specifications of OpenVLA** (在这里，DP代码被修改，可以接受语言指令)

- **OpenVLA (scratch),directly fine-tune the underlying base Prismatic VLM on the target robot setup – rather than fine-tuning the OpenX-pretrained OpenVLA model – to assess the benefit of large-scale robot pretraining.**（我的理解是：直接拿微调的数据集去训练原先pre-train的模型；既不是微调，也不是完整预训练一遍模型）

*For narrower but highly dexterous tasks, Diffusion Policy still shows smoother and more precise trajectories*
incorporating *action chunking and temporal smoothing*, as implemented in Diffusion Policy, may help OpenVLA attain the same level of dexterity and may be a promising direction for future work

关于DP的设置：predicting a chunk of T future actions and executing the first X actions in open-loop fashion before predicting the next chunk (for 15Hz control, we set T = 16,X = 8 like in the DROID prior work [11]; for 5Hz control, we reduce the chunk sizes to T = 8,X = 3)
It is also the only method in Section 5.2 that predicts **absolute Cartesian coordinates** to control the robot; all other methods use *relative position control*. **Diffusion Policy (matched) uses a single image as input, has no proprioceptive information and no observation history, and predicts a single relative position control action without action chunking**

### Parameter-Efficient Fine-Tuning
The full fine-tuning runs of OpenVLA in the previous section used 8 A100 GPUs for 5-15 hours per task (depending on the dataset size) to achieve high performance.

微调类型：
1. full fine- tuning updates all weights during fine-tuning, as described in Section 5.2（完整版）
2. last layer only fine-tunes only the last layer of OpenVLA’s transformer backbone and the token embedding matrix
3. frozen vision freezes the vision encoder but fine-tunes all other weights
4. sandwich fine-tuning unfreezes the vision encoder, token embedding matrix, and last layer
5. LoRA uses the popular low-rank adaptation technique

We find that the LoRA rank has negligible effect on policy performance and thus recommend using a default rank of $r$ = 32. With LoRA, we can fine-tune OpenVLA on a new task within 10-15 hours on a single A100 GPU – an 8x reduction in compute compared to full fine-tuning.

### Memory-Efficient Inference via Quantization
4-bit inference achieves higher throughput, since reduced GPU memory transfer compensates for the quantization overhead

不同的位数会影响运行的频率；4-bit quantized models can run at 3Hz on the A5000；8-bit quantization: on the A5000 GPU we use for our evaluations, we can only run the model at 1.2Hz

## limitation
1. it currently only supports single-image observations. In reality, real-world robot setups are heterogeneous, with a wide range of possible sensory inputs
2. improving the inference throughput of OpenVLA is critical to enable VLA control for high-frequency control setups such as ALOHA [90], which runs at 50Hz
3. there is room for further performance improvements; it does not yet offer very high reliability on the tested tasks, typically achieving <90% success rate.

- Does co-training on robot action prediction data and Internet-scale vision-language data substantially improve VLA performance? （一起训练效果大概率不如分开训练 go-1）
- What effect does the size of the base VLM have on VLA performance？
- What visual features are best-suited for VLA models？（3D？反正不是2D）