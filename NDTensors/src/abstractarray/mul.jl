function mul!!(CM::AbstractArray, AM::AbstractArray, BM::AbstractArray, α, β)
    CM = mul!(expose(CM), expose(AM), expose(BM), α, β)
    return CM
end

## TODO There is an issue in CUDA.jl
## When all are transpose CUDA.mul! isn't being
## Called correctly in `NDTensorsCUDAExt`
function mul!!(CM::Transpose, AM::Transpose, BM::Transpose, α, β)
    CM = mul!!(parent(CM), parent(BM), parent(AM), α, β)
    return CM
end
