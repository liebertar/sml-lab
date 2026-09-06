import mlx.core as mx


def test_mlx_matmul_runs_on_gpu():
    left = mx.random.normal((256, 256))
    right = mx.random.normal((256, 256))
    product = left @ right
    mx.eval(product)
    assert product.shape == (256, 256)
    assert mx.default_device() == mx.gpu
