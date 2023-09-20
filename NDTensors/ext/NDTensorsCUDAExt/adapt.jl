## Here we need an NDTensorCuArrayAdaptor because the CuArrayAdaptor provided by CUDA
## converts 64 bit numbers to 32 bit.  We cannot write `adapt(CuVector, x)` because this
## Will not allow us to properly utilize the buffer preference without changing the value of
## default_buffertype. Also `adapt(CuVector{<:Any, <:Any, Buffertype})` fails to work properly
struct NDTensorCuArrayAdaptor{B} end
## TODO make this work for unified. This works but overwrites CUDA's adapt_storage.
function cu(xs; unified::Bool=false)
  return fmap(
    x -> adapt(NDTensorCuArrayAdaptor{unified ? Mem.UnifiedBuffer : Mem.DeviceBuffer}(), x),
    xs,
  )
end

buffertype(::NDTensorCuArrayAdaptor{B}) where {B} = B

function Adapt.adapt_storage(adaptor::NDTensorCuArrayAdaptor, xs::AbstractArray)
  ElT = eltype(xs)
  BufT = buffertype(adaptor)
  return isbits(xs) ? xs : CuArray{ElT,1,BufT}(xs)
end

function Adapt.adapt_storage(
  adaptor::NDTensorCuArrayAdaptor, xs::NDTensors.UnallocatedZeros
)
  arraytype_specified_1 = set_unspecified_parameters(
    CuArray, Position(1), get_parameter(xs, Position(1))
  )
  arraytype_specified_2 = set_unspecified_parameters(
    arraytype_specified_1, Position(2), get_parameter(xs, Position(2))
  )

  elt = get_parameter(arraytype_specified_2, Position(1))
  N = get_parameter(arraytype_specified_2, Position(2))
  return NDTensors.UnallocatedZeros{
    elt,N,CUDA.CuArray{elt,N,default_parameter(CuArray, Position(3))}
  }(
    size(xs)
  )
end
