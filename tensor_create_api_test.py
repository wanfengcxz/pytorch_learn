import torch
import numpy as np


def init_mode(data: np.ndarray):
    """
        torch.Tensor是类构造函数。在构造一个张量时使用全局默认值。torch.get_default_dtype()可以查看dtype的全局默认值是torch.float32
        torch.tensor/torch.as_tensor/torch.from_numpy是工厂函数。工厂函数则根据输入推断数据类型。
    """
    
    Tensor = torch.Tensor(data)
    tensor = torch.tensor(data)
    from_numpy = torch.from_numpy(data)
    as_tensor = torch.as_tensor(data)
    
    print('输出的结果：')
    print(Tensor)
    print(tensor)
    print(from_numpy)
    print(as_tensor)

    print('输出的类型：')
    print(Tensor.dtype)
    print(tensor.dtype)
    print(from_numpy.dtype)
    print(as_tensor.dtype)
    
    """
        输出的结果：
        tensor([1., 2., 3.])
        tensor([1, 2, 3], dtype=torch.int32)
        tensor([1, 2, 3], dtype=torch.int32)
        tensor([1, 2, 3], dtype=torch.int32)
        输出的类型：
        torch.float32
        torch.int32
        torch.int32
        torch.int32
    """


def storage_mode():
    """
        Tensor 和 tensor是深拷贝 在内存中创建一个额外的数据副本 不共享内存 所以不受数组改变的影响。
        from_numpy和as_tensor是浅拷贝 在内存中共享数据 他们不同之处就是在于对内存的共享。
        torch.as_tensor()函数可以接受任何像Python数据结构这样的数组。
    """
    
    Tensor = torch.Tensor(data)
    tensor = torch.tensor(data)
    from_numpy = torch.from_numpy(data)
    as_tensor = torch.as_tensor(data)
    
    print('改变前：')
    print(Tensor)
    print(tensor)
    print(from_numpy)
    print(as_tensor)
    data[0] = 0
    data[1] = 0
    data[2] = 0
    print('改变后：')
    print(Tensor)
    print(tensor)
    print(from_numpy)
    print(as_tensor)

    """
        改变前：
        tensor([1., 2., 3.])
        tensor([1, 2, 3], dtype=torch.int32)
        tensor([1, 2, 3], dtype=torch.int32)
        tensor([1, 2, 3], dtype=torch.int32)
        改变后：
        tensor([1., 2., 3.])
        tensor([1, 2, 3], dtype=torch.int32)
        tensor([0, 0, 0], dtype=torch.int32)
        tensor([0, 0, 0], dtype=torch.int32)
    """

if __name__ == "__main__":
    
    data = np.random.rand(100, 100)
    
    # ref https://zhuanlan.zhihu.com/p/345648168
    init_mode()
    storage_mode()
    
