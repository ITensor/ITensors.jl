using CUDA: CuArray
using NDTensors: NDTensors

NDTensors.default_svd_alg(::Type{<:CuArray}, a) = "qr_algorithm"
