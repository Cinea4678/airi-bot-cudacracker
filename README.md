# Airi bot cudacracker

用于（田麻小溪创作的）爱莉机器人的“心愿瓶”功能的心愿瓶ID挖掘。

![图片样例](https://s.c.accr.cc/picgo/1746773405-272041.png)

## 性能

在 NVIDIA GeForce RTX 4090 上，每秒破解次数为约100亿次；单条ID的破解时间的数学期望为7小时。

实测Case：

- `saki0509`: 爆破结果 `saki_241717220108532`，尝试 241.7 万亿次。
- `Leo/need`: 爆破结果 `saki_495996201724237`，尝试 495.9 万亿次。

## 使用方法

### 前置：计算目标md5前缀

提示：目标串的长度应该为8，且符合Base64的规范。

参考Python代码（来自田麻小溪）：

``` python
import base64

# 解码函数：将 base64 编码的字符串解码为字节，然后转换为十六进制字符串（不带前缀 0x）
decr = lambda x: "".join([hex(i)[2:] for i in list(base64.b64decode(x.encode()))])

# 编码函数：将十六进制字符串转为字节后再 base64 编码
encr = lambda ff: base64.b64encode(
    bytes([int(ff[i:i+2], 16) for i in range(0, len(ff), 2)])
).decode()

# "下一值加密"函数：对 base64 字符串解码成十六进制后+1，然后再 base64 编码返回
nxcr = lambda x: encr(hex(int(decr(x), 16) + 1)[2:])
```

用例：

``` python
>>> decr('saki0509')
'b1a922d39d3d'
>>> decr('Leo/need')
'2dea3f9de79d'
```

### 环境准备

即将运行本项目的机器需要如下环境：

- CUDA：与机器的 GPU 匹配，愈新愈善。一般来说，如果从 Autodl 等平台租借 GPU 的话，默认的环境里已经包括 CUDA，不需自己安装。
- Rust：愈新愈善，可以从 [rustup.rs](https://rustup.rs) 下载和安装。

此外，您还需要提前查询所用GPU的 Compute Capability。可以在英伟达官方网站查询：[developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)。

### 克隆项目、修改参数、运行

克隆本项目：

``` sh
git clone git@github.com:Cinea4678/airi-bot-cudacracker.git
```

本项目的CUDA代码基于 4090 环境写就，在其他显卡上运行时，可能需要调整代码内如下部分的参数：

1. `build.rs`中，nvcc的arch参数。需要和GPU的Compute Capability匹配。

例如，在 GeForce RTX 3070 上运行时，查表可知其Compute Capability为8.6，因此需要修改arch参数为`arch=compute_86,code=sm_86`。

2. `src/gpu_code/md5.cu`中的`BATCH_SIZE`，根据GPU的显存大小和计算能力，可以上下调整并测试实际运行速度。

运行：

``` sh
cargo run -r -- <MD5前缀> [其他参数]
```

必填参数：

- MD5前缀：需要碰撞的目标MD5前缀，可以使用上文所述的方法计算。

可选参数：

- `-p`/测试前缀：所构造的测试字符串的前缀，默认为`saki_`。例如，默认生成的字符串是`saki_100` `saki_999`等。
- `-s`/起点：开始进行碰撞的数字，可以用于断点恢复。

### 样例

可以使用如下参数进行碰撞尝试：

``` sh
cargo run -r -- 681215efcb71 -p saki_
```

示例输出：

```
Target: ef151268 000071cb; prefix: saki_
找到匹配: saki_100
```

提示：寻找匹配的数学期望是251万亿次。
