import torch
import sys
import time

def stress_gpu(size=20000):
    """
    一个简单的函数，通过连续的矩阵乘法来给GPU施加高负载。

    :param size: 用于矩阵乘法的张量大小 (size x size)。
                 这个值需要根据你GPU的显存（VRAM）大小来调整。
    """
    
    # 1. 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误：CUDA (NVIDIA GPU) 'cuda' 设备不可用。")
        print("请检查你的PyTorch安装和NVIDIA驱动程序。")
        return

    device = torch.device("cuda")
    print(f"成功找到GPU：{torch.cuda.get_device_name(0)}")
    print(f"开始GPU压力测试... (张量大小: {size}x{size})")
    print("--------------------------------------------------")
    print("请在另一个终端使用 'nvidia-smi -l 1' 来监控GPU状态。")
    print("按 Ctrl+C 来停止脚本。")
    print("--------------------------------------------------")

    try:
        # 2. 在GPU上创建初始的大张量
        # 一个 (size, size) 的 float32 张量大约需要 size*size*4 字节的显存
        # 示例: 20000x20000x4 ≈ 1.6GB。我们创建两个，所以大约需要 3.2GB。
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        start_time = time.time()
        loop_count = 0

        # 3. 进入无限计算循环
        while True:
            # 执行核心计算：矩阵乘法
            # 我们不断地将 'a' 和 'b' 相乘，并将结果重新赋值给 'a'
            # 这能保持计算的连续性
            a = torch.matmul(a, b)
            
            loop_count += 1
            
            # 增加一个简单的吞吐量打印，每100次迭代打印一次
            if loop_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"已执行 {loop_count} 次迭代。平均速度: {loop_count / elapsed:.2f} 迭代/秒", end='\r')


    except KeyboardInterrupt:
        print("\n\n[用户中断]")
        print("压力测试已停止。")

    except RuntimeError as e:
        print(f"\n\n[运行时错误]")
        if "out of memory" in str(e):
            print("错误：CUDA 显存不足 (Out of Memory)！")
            print(f"当前的 'size' ({size}) 对你的GPU来说太大了。")
            print("请尝试减小 'size' 的值 (例如: 15000 或 10000)。")
        else:
            print(f"发生错误: {e}")
            
    except Exception as e:
        print(f"\n\n发生未知错误: {e}")

    finally:
        # 4. 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("GPU资源已释放。脚本退出。")


if __name__ == "__main__":
    # --- 在这里调整大小 ---
    # 根据你的GPU显存(VRAM)调整这个值
    # - 8GB VRAM: 尝试 15000 到 20000
    # - 24GB VRAM: 可以尝试 30000 到 40000
    # - 4GB VRAM: 尝试 10000
    # 如果遇到 "out of memory" 错误，请减小这个值。
    tensor_size = 20000

    # (可选) 允许从命令行传入size
    # 用法: python stress_test.py 15000
    if len(sys.argv) > 1:
        try:
            tensor_size = int(sys.argv[1])
        except ValueError:
            print(f"无效的size参数 '{sys.argv[1]}'. 使用默认值: {tensor_size}")

    stress_gpu(tensor_size)