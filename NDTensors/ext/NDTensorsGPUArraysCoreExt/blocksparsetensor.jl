using GPUArraysCore: @allowscalar, AbstractGPUArray
using NDTensors: NDTensors, BlockSparseTensor, dense, diag, diaglength, map_diag!
using NDTensors.Expose: Exposed, unexpose

## TODO to circumvent issues with blocksparse and scalar indexing
## convert blocksparse GPU tensors to dense tensors and call diag
## copying will probably have some impact on timing but this code
## currently isn't used in the main code, just in tests.
function NDTensors.diag(ETensor::Exposed{<:AbstractGPUArray, <:BlockSparseTensor})
    return diag(dense(unexpose(ETensor)))
end

## TODO scalar indexing is slow here
function NDTensors.map_diag!(
        f::Function,
        exposed_t_destination::Exposed{<:AbstractGPUArray, <:BlockSparseTensor},
        exposed_t_source::Exposed{<:AbstractGPUArray, <:BlockSparseTensor},
    )
    t_destination = unexpose(exposed_t_destination)
    t_source = unexpose(exposed_t_source)
    @allowscalar for i in 1:diaglength(t_destination)
        NDTensors.setdiagindex!(t_destination, f(NDTensors.getdiagindex(t_source, i)), i)
    end
    return t_destination
end
