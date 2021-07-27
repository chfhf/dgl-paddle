# from dgl.backend.numpy import tensor as np_tensor
import dgl.backend.numpy as np_tensor
import dgl.backend.paddorch as torch_tensor
import numpy as np
import paddorch as torch

max_error = 1e-6

def test_as_scalar():
    data=np.random.random(1)
    torch_data=torch.Tensor(data)
    np_out= np_tensor.as_scalar(data)
    torch_out=torch_tensor.as_scalar(torch_data)
    assert np.max(np.abs(np_out- torch_out))<=max_error, "as_scalar fail"

def test_sum():
    data=np.random.random([3,4])
    torch_data=torch.Tensor(data)
    np_out=np_tensor.sum(data,dim=1)
    torch_out=torch_tensor.sum(torch_data,dim=1).detach().numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "sum fail"


def test_reduce_sum():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.reduce_sum(data)
    torch_out = torch_tensor.reduce_sum(torch_data).numpy()
    assert np.abs(np_out - torch_out)<=max_error, "reduce sum fail"


def test_mean():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.mean(data,1)
    torch_out = torch_tensor.mean(torch_data,1).numpy()

    assert np.max(np.abs(np_out - torch_out))<=max_error, "mean fail"


def test_max():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.max(data,1)
    torch_out = torch_tensor.max(torch_data,1).detach().numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "max fail"

def test_reduce_max():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.reduce_max(data)
    torch_out = torch_tensor.reduce_max(torch_data).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "reduce_max fail"

def test_min():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.min(data,1)
    torch_out = torch_tensor.min(torch_data,1).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "min fail"

def test_reduce_min():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.reduce_min(data)
    torch_out = torch_tensor.reduce_min(torch_data).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "reduce_min fail"

def test_argsort():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.argsort(data,1,1)
    torch_out = torch_tensor.argsort(torch_data,1,True).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "argsort fail"

def test_topk():
    data = np.random.random([10, 8])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.topk(data,5,1,True)
    torch_out = torch_tensor.topk(torch_data,5,1,True).numpy()
    assert np.max(np.abs(np_out - torch_out))<=1e-6, "topk fail"

def test_argtopk():
    data = np.random.random([10, 8])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.topk(data,5,1,True)
    torch_out = torch_tensor.topk(torch_data,5,1,True).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "argtopk fail"

def test_exp():
    data = np.random.random([3, 4]).astype("float32")
    torch_data = torch.Tensor(data)
    np_out = np_tensor.exp(data)
    torch_out = torch_tensor.exp(torch_data).numpy()
    assert np.max(np.abs(np_out - torch_out)), "exp fail"

def test_softmax():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.softmax(data)
    torch_out = torch_tensor.softmax(torch_data).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "softmax fail"

def test_cat():
    data = np.random.random([3, 4])
    data2 = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    torch_data2 = torch.Tensor(data2)
    np_out = np_tensor.cat([data,data2],1)
    torch_out = torch_tensor.cat([torch_data,torch_data2],1).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "cat fail"

def test_split():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.split(data,2,1)
    torch_out = torch_tensor.split(torch_data,2,1)
    assert np.max(np.abs(np_out[0] - torch_out[0].numpy()))<=max_error, "split fail"
    assert np.max(np.abs(np_out[1] - torch_out[1].numpy())) <= max_error, "split fail"


def test_repeat():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.repeat(data,2,1)
    torch_out = torch_tensor.repeat(torch_data,2,1).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "repeat fail"

def test_gather_row():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.gather_row(data,1)
    torch_out = torch_tensor.gather_row(torch_data,torch.tensor([1])).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "gather_row fail"

def test_slice_axis():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.slice_axis(data,1,0,2)
    torch_out = torch_tensor.slice_axis(torch_data,1,0,2).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "slice_axis fail"

def test_take():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.take(data,[1,2],1)
    torch_out = torch_tensor.take(torch_data,torch.tensor([1,2]),1).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "take fail"

def test_scatter_row():
    data = np.random.random([3, 4])
    value = np.random.random(4)
    torch_data = torch.Tensor(data)
    torch_val = torch.Tensor([value])
    np_out = np_tensor.scatter_row(data,1,value)
    torch_out = torch_tensor.scatter_row(torch_data,torch.tensor(1),torch_val).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "scatter_row fail"

def test_scatter_row_inplace():
    data = np.random.random([3, 4])
    value = np.random.random(4)
    torch_data = torch.Tensor(data)
    torch_val = torch.Tensor([value])
    np_tensor.scatter_row_inplace(data,1,value)
    torch_tensor.scatter_row_inplace(torch_data,torch.tensor(1),torch_val)
    assert np.max(np.abs(data - torch_data.numpy()))<=max_error, "scatter_row_inplace fail"

def test_squeeze():
    data = np.random.random([3, 4,1])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.squeeze(data,2)
    torch_out = torch_tensor.squeeze(torch_data,2).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "squeeze fail"

# def test_unsqueeze():
#     data = np.random.random([3, 4,1])
#     torch_data = torch.Tensor(data)
#     np_out = np_tensor.unsqueeze(data,2)
#     torch_out = torch_tensor.unsqueeze(torch_data,2).numpy()
#     assert np.max(np.abs(np_out - torch_out))<=max_error, "unsqueeze fail"
def test_reshape():
    data = np.random.random([3, 4,1])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.reshape(data,[4,3])
    torch_out = torch_tensor.reshape(torch_data,[4,3]).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "reshape fail"

def test_zeros():

    np_out = np_tensor.zeros([4,3],"float32")
    torch_out = torch_tensor.zeros([4,3],torch.float,"cpu").numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "zeros fail"

def test_ones():

    np_out = np_tensor.ones([4,3],"float32")
    torch_out = torch_tensor.ones([4,3],torch.float,"cpu").numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "ones fail"

def test_unique():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.unique(data)
    torch_out = torch_tensor.unique(torch_data).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "unique fail"

def test_full_1d():

    np_out = np_tensor.full_1d(4,2)
    torch_out = torch.full((4,),2).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "full_1d fail"

def test_nonzero_1d():
    """
    This function is error
    Returns
    -------

    """
    data = np.random.random([3,4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.nonzero_1d(data)
    torch_out = torch_tensor.nonzero_1d(torch_data).numpy()
    assert np.max(np.abs(np_out - torch_out))<=max_error, "nonzero_1d fail"

def test_sort_1d():
    data = np.random.random([13])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.sort_1d(data)
    torch_out = torch_tensor.sort_1d(torch_data)

    assert np.max(np.abs(torch_out[0].numpy() - np_out[0]))<=max_error, "sort_1d fail"

def test_rand_shuffle():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.rand_shuffle(data)
    torch_out = torch_tensor.rand_shuffle(torch_data).numpy()
    assert  torch_out.shape==np_out.shape, "rand_shuffle fail"

def test_zerocopy_to_numpy():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.zerocopy_to_numpy(data)
    torch_out = torch_tensor.zerocopy_to_numpy(torch_data)
    assert np.max(np.abs(np_out - torch_out))<=max_error, "zerocopy_to_numpy fail"

def test_zerocopy_from_numpy():
    data = np.random.random([3, 4])
    torch_data = torch.Tensor(data)
    np_out = np_tensor.zerocopy_to_numpy(data)
    torch_out = torch_tensor.zerocopy_to_numpy(torch_data)
    assert np.max(np.abs(np_out - torch_out))<=max_error, "zerocopy_from_numpy fail"