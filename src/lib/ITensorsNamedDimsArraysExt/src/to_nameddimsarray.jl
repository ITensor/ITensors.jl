using ..NDTensors: data, inds
using ITensors: ITensor

# TODO: Delete this, it is a hack to decide
# if an Index is blocked.
function is_blocked_ind(i)
  return try
    blockdim(i, 1)
    true
  catch
    false
  end
end

# TODO: Delete once `TensorStorage` is removed.
function to_axes(inds::Tuple)
  if any(is_blocked_ind, inds)
    return BlockArrays.blockedrange.(map(i -> [blockdim(i, b) for b in 1:nblocks(i)], inds))
  else
    return Base.OneTo.(dim.(inds))
  end
end

using ..NDTensors: DenseTensor
# TODO: Delete once `Dense` is removed.
function to_nameddimsarray(x::DenseTensor)
  return named(reshape(data(x), size(x)), name.(inds(x)))
end

using ..NDTensors: DiagTensor
using ..NDTensors.DiagonalArrays: DiagonalArray
# TODO: Delete once `Diag` is removed.
function to_nameddimsarray(x::DiagTensor)
  return named(DiagonalArray(data(x), size(x)), name.(inds(x)))
end

using ..NDTensors: BlockSparseTensor
using ..NDTensors.BlockSparseArrays: BlockSparseArray
# TODO: Delete once `BlockSparse` is removed.
function to_nameddimsarray(x::BlockSparseTensor)
  blockinds = map(i -> [blockdim(i, b) for b in 1:nblocks(i)], inds(x))
  blocktype = set_ndims(datatype(x), ndims(x))
  # TODO: Make a simpler constructor:
  # BlockSparseArray(blocktype, blockinds)
  arraystorage = BlockSparseArray{eltype(x),ndims(x),blocktype}(blockinds)
  for b in nzblocks(x)
    arraystorage[BlockArrays.Block(Tuple(b)...)] = x[b]
  end
  return named(arraystorage, name.(inds(x)))
end

## TODO: Add this back, define `CombinerArrays` library in NDTensors!
## using ..NDTensors: CombinerTensor, CombinerArray, storage
## # TODO: Delete when we directly use `CombinerArray` as storage.
## function to_nameddimsarray(t::CombinerTensor)
##   return named(CombinerArray(storage(t), to_axes(inds(t))), name.(inds(t)))
## end

to_nameddimsarray(t::ITensor) = ITensor(to_nameddimsarray(t.tensor))
