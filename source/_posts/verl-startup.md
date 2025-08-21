---
title: verl 入门：在 RTX 4090 上配置环境
categories:
- 科研
tags:
- verl
- RLHF
index_img: /images/backgrounds/hybridflow.webp
banner_img: /images/backgrounds/hybridflow.webp
date: 2025-07-03 18:00:00
---

verl 是字节跳动 Seed 团队研发的大模型 RLHF 框架，其单-多控制器混合编程模型为算法研究人员提供了更为易用的接口。

<!-- more -->

## 前言

近期主包有幸参与 verl 框架的维护工作，希望后续能在实验室服务器上对 multi-turn rollout 相关功能进行调试和二次开发。verl 目前对各种硬件的兼容和适配尚未成熟，我在 RTX 4090 这种非计算卡上进行环境配置时踩了不少坑。这里记录一下环境配置的流程和踩过的坑，希望能帮到大家。

截止发文，verl 的最新版本为 v0.4.1。

{% note primary %}
**更新**
发文后不久，我的开发环境版本从 v0.4.1 切换至 v0.5.0（因为其包含自己提交的补丁），整体配置流程并无较大差异，同样需求的朋友们推荐使用 v0.5.0 版本。
{% endnote %}

## 基础环境安装

首先需要进行基础 python 环境的配置，安装最新版本 miniconda 即可。verl 要求 python 3.10，在创建虚拟环境时需要对版本号进行指定，即 `conda create -n veRL python=3.10`。

创建环境后，在克隆下来的verl仓库根目录运行 `pip install -e .[sglang]` 安装 verl。`.[sglang]` 表示安装包含 sglang 的 verl，`-e` 表示以本地作为依赖代码源，这样直接在仓库内修改代码可直观体现在运行过程中。

如果像我一样后续需要对 SGLang 进行修改，可以安装本地 SGLang：先将 SGLang 的 git 仓库克隆到本地，随后执行 `pip uninstall sglang` 卸载默认依赖，随后前往 SGLang 本地文件夹，切换至 `v0.4.6.post5` tag 并执行 `pip install -e "python[all]"` 进行本地安装。

## bugfix: `undefined symbol` @`flash_attn_2_cuda.so`

安装后运行verl，初始化阶段出现报错 `ImportError: /.../flash_attn_2_cuda.so: undefined symbol: _ZN3c105ErrorC2ENS_14Source`。调试发现flash-attn出现问题，直接import即可复现报错。

```log
(verl) zhaotianyun@aiseon:~/verl$ python3
Python 3.10.18 (main, Jun  5 2025, 13:14:17) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import flash_attn
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/flash_attn/__init__.py", line 3, in <module>
    from flash_attn.flash_attn_interface import (
  File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/flash_attn/flash_attn_interface.py", line 15, in <module>
    import flash_attn_2_cuda as flash_attn_gpu
ImportError: /home/zhaotianyun/miniconda3/envs/veRL/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

最终定位发现 [flash-attn 2.7.4 预编译二进制包存在 ABI 配置错误](https://github.com/Dao-AILab/flash-attention/issues/1717)，默认安装的 `abiTRUE` 版本无法被 python 解释器正常解析。为解决这一问题，需要卸载默认安装的 flash-attn，重新安装 `abiFALSE` 版本。

```bash
pip uninstall flash-attn
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

重装完毕后，flash-attn 可以被正常导入，verl 可以正常启动并执行初始化。

## bugfix: OOM on first step

{% note success %}
此 bug 已在 v0.5.0 版本中被修复。
{% endnote %}

在推理第一个 step 时遇到了 [OOM 报错](https://github.com/volcengine/verl/issues/2189)：

```log
[torch_memory_saver.cpp] CUresult error  result=2 file=csrc/torch_memory_saver.cpp func=cu_mem_create line=104
```

经调试，发现 resharding 阶段加载内存前程序未能及时清空缓存。通过在适当位置释放缓存的方式可以有效的解决这一问题。补丁已提交 [PR](https://github.com/volcengine/verl/pull/2253) 并被合入。

## bugfix: `peer access is not supported between these two devices`

在推理第一个 step 时遇到了 [SGLang 报错](https://github.com/volcengine/verl/issues/1874)。

```log
(TaskRunner pid=610186) Training from scratch
(TaskRunner pid=610186) test_gen_batch meta info: {'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}
(WorkerDict pid=611035) [2025-07-01 13:15:04 TP1] Cache flushed successfully!
(WorkerDict pid=611035) [2025-07-01 13:15:04 TP0] Cache flushed successfully!
(WorkerDict pid=616095) [2025-07-01 13:15:04 TP0] Cache flushed successfully!
(WorkerDict pid=616093) [2025-07-01 13:15:04 TP1] Cache flushed successfully!
(WorkerDict pid=616093) [2025-07-01 13:15:04 TP0] Cache flushed successfully!
(WorkerDict pid=616097) [2025-07-01 13:15:04 TP1] Cache flushed successfully!
(WorkerDict pid=616095) [2025-07-01 13:15:04 TP1] Cache flushed successfully!
(WorkerDict pid=616097) [2025-07-01 13:15:04 TP0] Cache flushed successfully!
(WorkerDict pid=611035) [2025-07-01 13:15:06 TP1] Scheduler hit an exception: Traceback (most recent call last):
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py", line 2311, in run_scheduler_process
(WorkerDict pid=611035)     scheduler.event_loop_overlap()
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
(WorkerDict pid=611035)     return func(*args, **kwargs)
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py", line 662, in event_loop_overlap
(WorkerDict pid=611035)     self.process_input_requests(recv_reqs)
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py", line 889, in process_input_requests
(WorkerDict pid=611035)     output = self._request_dispatcher(recv_req)
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/utils.py", line 471, in __call__
(WorkerDict pid=611035)     return fn(obj)
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/managers/scheduler.py", line 2035, in update_weights_from_tensor
(WorkerDict pid=611035)     success, message = self.tp_worker.update_weights_from_tensor(recv_req)
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/managers/tp_worker_overlap_thread.py", line 254, in update_weights_from_tensor
(WorkerDict pid=611035)     success, message = self.worker.update_weights_from_tensor(recv_req)
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/managers/tp_worker.py", line 255, in update_weights_from_tensor
(WorkerDict pid=611035)     success, message = self.model_runner.update_weights_from_tensor(
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py", line 742, in update_weights_from_tensor
(WorkerDict pid=611035)     named_tensors = [
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py", line 743, in <listcomp>
(WorkerDict pid=611035)     (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank))
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py", line 1296, in _unwrap_tensor
(WorkerDict pid=611035)     tensor = tensor.get(tp_rank)
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py", line 1308, in get
(WorkerDict pid=611035)     return MultiprocessingSerializer.deserialize(self.values[rank])
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/utils.py", line 1672, in deserialize
(WorkerDict pid=611035)     return ForkingPickler.loads(data)
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/sglang/srt/patch_torch.py", line 51, in _rebuild_cuda_tensor_modified
(WorkerDict pid=611035)     return reductions._rebuild_cuda_tensor_original(*args)
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/torch/multiprocessing/reductions.py", line 181, in rebuild_cuda_tensor
(WorkerDict pid=611035)     storage = storage_cls._new_shared_cuda(
(WorkerDict pid=611035)   File "/home/zhaotianyun/miniconda3/envs/verl/lib/python3.10/site-packages/torch/storage.py", line 1452, in _new_shared_cuda
(WorkerDict pid=611035)     return torch.UntypedStorage._new_shared_cuda(*args, **kwargs)
(WorkerDict pid=611035) RuntimeError: CUDA error: peer access is not supported between these two devices
(WorkerDict pid=611035) CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
(WorkerDict pid=611035) For debugging consider passing CUDA_LAUNCH_BLOCKING=1
(WorkerDict pid=611035) Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

使用 `torch.multiprocessing` 在多张非计算卡（如 4090）上进行分布式训练，当子进程的 `CUDA_VISIBLE_DEVICES` 与主进程不一致时，待传输张量中记录的 `cuda_device` 在不同进程里指代不同的物理 GPU，而父子进程获取这一向量时都使用 `cuda_device` 直接获取，最终导致程序尝试建立一条并不存在的通道，出现报错。社区曾尝试用 UUID 替换设备编号进行修复，但补丁未能在子进程上正常生效，导致了同样的问题。

我编写了一个临时解决方案，通过修改 SGLang 的部分代码进行修复。方案同步发表在 [issue](https://github.com/volcengine/verl/issues/1874) 中。

```python
# sglang/srt/utils.py
class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        # 添加下面两行代码
        from sglang.srt.patch_torch import monkey_patch_torch_reductions
        monkey_patch_torch_reductions()

        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        ...
```

修复后即可正常运行 multi-turn rollout 示例。


## 模型选择

对于 8x4090 48G 的服务器，3B模型会出现OOM的情况，需要修改配置文件使用1.5B模型，更改后的配置文件如下：

{% note danger %}
本配置文件适用于 v0.5.0。其仅修改了 `actor_rollout_ref.model.path` 与 `trainer.experiment_name`，如使用老版本仅需在对应配置文件中修改这两个参数即可。
{% endnote %}

```sh
# verl/examples/sglang_multiturn/run_qwen2.5-1.5b_gsm8k_multiturn.sh
# run on 8x4090 48G
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='gsm8k_async_rl' \
    trainer.experiment_name='qwen2.5-1.5b_function_rm-gsm8k-sgl-multi-w-tool-verify-n16' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml" \
    trainer.total_epochs=15 \
    actor_rollout_ref.rollout.update_weights_bucket_megabytes=512 $@
```

