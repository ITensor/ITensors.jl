
# Diag can have either Vector storage, in which case
# it is a general Diag tensor, or scalar storage,
# in which case the diagonal has a uniform value
struct Diag{ElT,VecT} <: TensorStorage{ElT}
  data::VecT
  function Diag{ElT,VecT}(data) where {ElT,VecT<:AbstractVector{ElT}}
    return new{ElT,VecT}(data)
  end
  function Diag{ElT,ElT}(data) where {ElT}
    return new{ElT,ElT}(data)
  end
end

datatype(::Type{<:Diag{<:Any,DataT}}) where {DataT} = DataT

setdata(D::Diag, ndata) = Diag(ndata)
setdata(storagetype::Type{<:Diag}, data) = Diag(data)

function set_eltype(storagetype::Type{<:Diag}, eltype::Type)
  return set_datatype(storagetype, set_eltype(datatype(storagetype), eltype))
end

function set_datatype(storagetype::Type{<:Diag}, datatype::Type{<:AbstractVector})
  return Diag{eltype(datatype),datatype}
end

Diag(data::VecT) where {VecT<:AbstractVector{ElT}} where {ElT} = Diag{ElT,VecT}(data)

Diag(data::ElT) where {ElT<:Number} = Diag{ElT,ElT}(data)

function Diag{ElR}(data::AbstractVector{ElT}) where {ElR<:Number,ElT<:Number}
  return ElT == ElR ? Diag(data) : Diag(ElR.(data))
end

Diag(::Type{ElT}, n::Integer) where {ElT<:Number} = Diag(zeros(ElT, n))

Diag(x::ElT, n::Integer) where {ElT<:Number} = Diag(fill(x, n))

copy(D::Diag) = Diag(copy(data(D)))

const NonuniformDiag{ElT,VecT} = Diag{ElT,VecT} where {VecT<:AbstractVector}

const UniformDiag{ElT,VecT} = Diag{ElT,VecT} where {VecT<:Number}

function set_eltype(storagetype::Type{<:UniformDiag}, eltype::Type)
  return Diag{eltype,eltype}
end

function set_datatype(storagetype::Type{<:UniformDiag}, datatype::Type)
  return Diag{datatype,datatype}
end

# Special printing for uniform Diag
function show(io::IO, mime::MIME"text/plain", diag::UniformDiag)
  println(io, typeof(diag))
  println(io, "Diag storage with uniform diagonal value:")
  println(io, diag[1])
  return nothing
end

getindex(D::UniformDiag, i::Int) = data(D)

function setindex!(D::UniformDiag, val, i::Int)
  return error("Cannot set elements of a uniform Diag storage")
end

Base.real(::Type{Diag{ElT,Vector{ElT}}}) where {ElT} = Diag{real(ElT),Vector{real(ElT)}}
Base.real(::Type{Diag{ElT,ElT}}) where {ElT} = Diag{real(ElT),real(ElT)}

complex(::Type{Diag{ElT,Vector{ElT}}}) where {ElT} = Diag{complex(ElT),Vector{complex(ElT)}}
complex(::Type{Diag{ElT,ElT}}) where {ElT} = Diag{complex(ElT),complex(ElT)}

# Deal with uniform Diag conversion
convert(::Type{<:Diag{ElT,VecT}}, D::Diag) where {ElT,VecT} = Diag(convert(VecT, data(D)))

# TODO: make this work for other storage besides Vector
zeros(::Type{<:NonuniformDiag{ElT}}, dim::Int64) where {ElT} = Diag(zeros(ElT, dim))
zeros(::Type{<:UniformDiag{ElT}}, dim::Int64) where {ElT} = Diag(zero(ElT))

#
# Type promotions involving Diag
# Useful for knowing how conversions should work when adding and contracting
#

function promote_rule(
  ::Type{<:UniformDiag{ElT1}}, ::Type{<:UniformDiag{ElT2}}
) where {ElT1,ElT2}
  ElR = promote_type(ElT1, ElT2)
  return Diag{ElR,ElR}
end

function promote_rule(
  ::Type{<:NonuniformDiag{ElT1,VecT1}}, ::Type{<:NonuniformDiag{ElT2,VecT2}}
) where {ElT1,VecT1<:AbstractVector,ElT2,VecT2<:AbstractVector}
  ElR = promote_type(ElT1, ElT2)
  VecR = promote_type(VecT1, VecT2)
  return Diag{ElR,VecR}
end

# This is an internal definition, is there a more general way?
#promote_type(::Type{Vector{ElT1}},
#                  ::Type{ElT2}) where {ElT1<:Number,
#                                       ElT2<:Number} = Vector{promote_type(ElT1,ElT2)}
#
#promote_type(::Type{ElT1},
#                  ::Type{Vector{ElT2}}) where {ElT1<:Number,
#                                               ElT2<:Number} = promote_type(Vector{ElT2},ElT1)

# TODO: how do we make this work more generally for T2<:AbstractVector{S2}?
# Make a similartype(AbstractVector{S2},T1) -> AbstractVector{T1} function?
function promote_rule(
  ::Type{<:UniformDiag{ElT1,VecT1}}, ::Type{<:NonuniformDiag{ElT2,Vector{ElT2}}}
) where {ElT1,VecT1<:Number,ElT2}
  ElR = promote_type(ElT1, ElT2)
  VecR = Vector{ElR}
  return Diag{ElR,VecR}
end

function promote_rule(
  ::Type{DenseT1}, ::Type{<:NonuniformDiag{ElT2,VecT2}}
) where {DenseT1<:Dense,ElT2,VecT2<:AbstractVector}
  return promote_type(DenseT1, Dense{ElT2,VecT2})
end

function promote_rule(
  ::Type{DenseT1}, ::Type{<:UniformDiag{ElT2,VecT2}}
) where {DenseT1<:Dense,ElT2,VecT2<:Number}
  return promote_type(DenseT1, ElT2)
end

# Convert a Diag storage type to the closest Dense storage type
dense(::Type{<:NonuniformDiag{ElT,VecT}}) where {ElT,VecT} = Dense{ElT,VecT}
dense(::Type{<:UniformDiag{ElT}}) where {ElT} = Dense{ElT,Vector{ElT}}

const DiagTensor{ElT,N,StoreT,IndsT} = Tensor{ElT,N,StoreT,IndsT} where {StoreT<:Diag}
const NonuniformDiagTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:NonuniformDiag}
const UniformDiagTensor{ElT,N,StoreT,IndsT} =
  Tensor{ElT,N,StoreT,IndsT} where {StoreT<:UniformDiag}

IndexStyle(::Type{<:DiagTensor}) = IndexCartesian()

# TODO: this needs to be better (promote element type, check order compatibility,
# etc.
function convert(::Type{<:DenseTensor{ElT,N}}, T::DiagTensor{ElT,N}) where {ElT<:Number,N}
  return dense(T)
end

convert(::Type{Diagonal}, D::DiagTensor{<:Number,2}) = Diagonal(data(D))

# These are rules for determining the output of a pairwise contraction of NDTensors
# (given the indices of the output tensors)
function contraction_output_type(
  tensortype1::Type{<:DiagTensor}, tensortype2::Type{<:DenseTensor}, indsR
)
  return similartype(promote_type(tensortype1, tensortype2), indsR)
end
function contraction_output_type(
  tensortype1::Type{<:DenseTensor}, tensortype2::Type{<:DiagTensor}, indsR
)
  return contraction_output_type(tensortype2, tensortype1, indsR)
end

# This performs the logic that DiagTensor*DiagTensor -> DiagTensor if it is not an outer
# product but -> DenseTensor if it is
# TODO: if the tensors are both order 2 (or less), or if there is an Index replacement,
# then they remain diagonal. Should we limit DiagTensor*DiagTensor to cases that
# result in a DiagTensor, for efficiency and type stability? What about a general
# SparseTensor result?
function contraction_output_type(
  tensortype1::Type{<:DiagTensor}, tensortype2::Type{<:DiagTensor}, indsR
)
  if length(indsR) == ndims(tensortype1) + ndims(tensortype2)
    # Turn into is_outer(inds1,inds2,indsR) function?
    # How does type inference work with arithmatic of compile time values?
    return similartype(dense(promote_type(tensortype1, tensortype2)), indsR)
  end
  return similartype(promote_type(tensortype1, tensortype2), indsR)
end

# The output must be initialized as zero since it is sparse, cannot be undefined
function contraction_output(T1::DiagTensor, T2::Tensor, indsR)
  return zero_contraction_output(T1, T2, indsR)
end
contraction_output(T1::Tensor, T2::DiagTensor, indsR) = contraction_output(T2, T1, indsR)

function contraction_output(T1::DiagTensor, T2::DiagTensor, indsR)
  return zero_contraction_output(T1, T2, indsR)
end

function Array{ElT,N}(T::DiagTensor{ElT,N}) where {ElT,N}
  return array(T)
end

function Array(T::DiagTensor{ElT,N}) where {ElT,N}
  return Array{ElT,N}(T)
end

function zeros(tensortype::Type{<:DiagTensor}, inds)
  return tensor(zeros(storagetype(tensortype), mindim(inds)), inds)
end

function zeros(tensortype::Type{<:DiagTensor}, inds::Tuple{})
  return tensor(zeros(storagetype(tensortype), mindim(inds)), inds)
end

function diag(tensor::DiagTensor)
  tensor_diag = NDTensors.similar(dense(typeof(tensor)), (diaglength(tensor),))
  # TODO: Define `eachdiagindex`.
  for j in 1:diaglength(tensor)
    tensor_diag[j] = getdiagindex(tensor, j)
  end
  return tensor_diag
end

"""
getdiagindex(T::DiagTensor,i::Int)

Get the ith value along the diagonal of the tensor.
"""
getdiagindex(T::DiagTensor{<:Number}, ind::Int) = storage(T)[ind]

"""
setdiagindex!(T::DiagTensor,i::Int)

Set the ith value along the diagonal of the tensor.
"""
setdiagindex!(T::DiagTensor{<:Number}, val, ind::Int) = (storage(T)[ind] = val)

"""
setdiag(T::UniformDiagTensor,val)

Set the entire diagonal of a uniform DiagTensor.
"""
setdiag(T::UniformDiagTensor, val) = tensor(Diag(val), inds(T))

@propagate_inbounds function getindex(
  T::DiagTensor{ElT,N}, inds::Vararg{Int,N}
) where {ElT,N}
  if all(==(inds[1]), inds)
    return getdiagindex(T, inds[1])
  else
    return zero(eltype(ElT))
  end
end
@propagate_inbounds getindex(T::DiagTensor{<:Number,1}, ind::Int) = storage(T)[ind]
@propagate_inbounds getindex(T::DiagTensor{<:Number,0}) = storage(T)[1]

# Set diagonal elements
# Throw error for off-diagonal
@propagate_inbounds function setindex!(
  T::DiagTensor{<:Number,N}, val, inds::Vararg{Int,N}
) where {N}
  all(==(inds[1]), inds) || error("Cannot set off-diagonal element of Diag storage")
  setdiagindex!(T, val, inds[1])
  return T
end
@propagate_inbounds function setindex!(T::DiagTensor{<:Number,1}, val, ind::Int)
  return (storage(T)[ind] = val)
end
@propagate_inbounds setindex!(T::DiagTensor{<:Number,0}, val) = (storage(T)[1] = val)

function setindex!(T::UniformDiagTensor{<:Number,N}, val, inds::Vararg{Int,N}) where {N}
  return error("Cannot set elements of a uniform Diag storage")
end

# TODO: make a fill!! that works for uniform and non-uniform
#fill!(T::DiagTensor,v) = fill!(storage(T),v)

function dense(::Type{<:Tensor{ElT,N,StoreT,IndsT}}) where {ElT,N,StoreT<:Diag,IndsT}
  return Tensor{ElT,N,dense(StoreT),IndsT}
end

# convert to Dense
function dense(T::TensorT) where {TensorT<:DiagTensor}
  R = zeros(dense(TensorT), inds(T))
  for i in 1:diaglength(T)
    setdiagindex!(R, getdiagindex(T, i), i)
  end
  return R
end

denseblocks(T::DiagTensor) = dense(T)

function outer!(
  R::DenseTensor{<:Number,NR}, T1::DiagTensor{<:Number,N1}, T2::DiagTensor{<:Number,N2}
) where {NR,N1,N2}
  for i1 in 1:diaglength(T1), i2 in 1:diaglength(T2)
    indsR = CartesianIndex{NR}(ntuple(r -> r ≤ N1 ? i1 : i2, Val(NR)))
    R[indsR] = getdiagindex(T1, i1) * getdiagindex(T2, i2)
  end
  return R
end

# TODO: write an optimized version of this?
function outer!(R::DenseTensor{ElR}, T1::DenseTensor, T2::DiagTensor) where {ElR}
  R .= zero(ElR)
  outer!(R, T1, dense(T2))
  return R
end

function outer!(R::DenseTensor{ElR}, T1::DiagTensor, T2::DenseTensor) where {ElR}
  R .= zero(ElR)
  outer!(R, dense(T1), T2)
  return R
end

# Right an in-place version
function outer(T1::DiagTensor{ElT1,N1}, T2::DiagTensor{ElT2,N2}) where {ElT1,ElT2,N1,N2}
  indsR = unioninds(inds(T1), inds(T2))
  R = tensor(Dense(zeros(promote_type(ElT1, ElT2), dim(indsR))), indsR)
  outer!(R, T1, T2)
  return R
end

function permutedims!(
  R::DiagTensor{<:Number,N},
  T::DiagTensor{<:Number,N},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {N}
  # TODO: check that inds(R)==permute(inds(T),perm)?
  for i in 1:diaglength(R)
    @inbounds setdiagindex!(R, f(getdiagindex(R, i), getdiagindex(T, i)), i)
  end
  return R
end

function permutedims(
  T::DiagTensor{<:Number,N}, perm::NTuple{N,Int}, f::Function=identity
) where {N}
  R = NDTensors.similar(T, permute(inds(T), perm))
  g(r, t) = f(t)
  permutedims!(R, T, perm, g)
  return R
end

function permutedims(
  T::UniformDiagTensor{<:Number,N}, perm::NTuple{N,Int}, f::Function=identity
) where {N}
  R = tensor(Diag(f(getdiagindex(T, 1))), permute(inds(T), perm))
  return R
end

# Version that may overwrite in-place or may return the result
function permutedims!!(
  R::NonuniformDiagTensor{<:Number,N},
  T::NonuniformDiagTensor{<:Number,N},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {N}
  R = convert(promote_type(typeof(R), typeof(T)), R)
  permutedims!(R, T, perm, f)
  return R
end

function permutedims!!(
  R::UniformDiagTensor{ElR,N},
  T::UniformDiagTensor{ElT,N},
  perm::NTuple{N,Int},
  f::Function=(r, t) -> t,
) where {ElR,ElT,N}
  R = convert(promote_type(typeof(R), typeof(T)), R)
  R = tensor(Diag(f(getdiagindex(R, 1), getdiagindex(T, 1))), inds(R))
  return R
end

function permutedims!(
  R::DenseTensor{ElR,N}, T::DiagTensor{ElT,N}, perm::NTuple{N,Int}, f::Function=(r, t) -> t
) where {ElR,ElT,N}
  for i in 1:diaglength(T)
    @inbounds setdiagindex!(R, f(getdiagindex(R, i), getdiagindex(T, i)), i)
  end
  return R
end

function permutedims!!(
  R::DenseTensor{ElR,N}, T::DiagTensor{ElT,N}, perm::NTuple{N,Int}, f::Function=(r, t) -> t
) where {ElR,ElT,N}
  RR = convert(promote_type(typeof(R), typeof(T)), R)
  permutedims!(RR, T, perm, f)
  return RR
end

# TODO: make a single implementation since this is
# the same as the version with the input types
# swapped.
function permutedims!!(
  R::DiagTensor{ElR,N}, T::DenseTensor{ElT,N}, perm::NTuple{N,Int}, f::Function=(r, t) -> t
) where {ElR,ElT,N}
  RR = convert(promote_type(typeof(R), typeof(T)), R)
  permutedims!(RR, T, perm, f)
  return RR
end

function _contract!!(
  R::UniformDiagTensor{ElR,NR},
  labelsR,
  T1::UniformDiagTensor{<:Number,N1},
  labelsT1,
  T2::UniformDiagTensor{<:Number,N2},
  labelsT2,
) where {ElR,NR,N1,N2}
  if NR == 0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    R = setdiag(R, diaglength(T1) * getdiagindex(T1, 1) * getdiagindex(T2, 1))
  else
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    R = setdiag(R, getdiagindex(T1, 1) * getdiagindex(T2, 1))
  end
  return R
end

function contract!(
  R::DiagTensor{ElR,NR},
  labelsR,
  T1::DiagTensor{<:Number,N1},
  labelsT1,
  T2::DiagTensor{<:Number,N2},
  labelsT2,
) where {ElR,NR,N1,N2}
  if NR == 0  # If all indices of A and B are contracted
    # all indices are summed over, just add the product of the diagonal
    # elements of A and B
    Rdiag = zero(ElR)
    for i in 1:diaglength(T1)
      Rdiag += getdiagindex(T1, i) * getdiagindex(T2, i)
    end
    setdiagindex!(R, Rdiag, 1)
  else
    min_dim = min(diaglength(T1), diaglength(T2))
    # not all indices are summed over, set the diagonals of the result
    # to the product of the diagonals of A and B
    for i in 1:min_dim
      setdiagindex!(R, getdiagindex(T1, i) * getdiagindex(T2, i), i)
    end
  end
  return R
end

function contract!(
  C::DenseTensor{ElC,NC},
  Clabels,
  A::DiagTensor{ElA,NA},
  Alabels,
  B::DenseTensor{ElB,NB},
  Blabels,
  α::Number=one(ElC),
  β::Number=zero(ElC);
  convert_to_dense::Bool=true,
) where {ElA,NA,ElB,NB,ElC,NC}
  #@timeit_debug timer "diag-dense contract!" begin 
  if all(i -> i < 0, Blabels)
    # If all of B is contracted
    # TODO: can also check NC+NB==NA
    min_dim = min(minimum(dims(A)), minimum(dims(B)))
    if length(Clabels) == 0
      # all indices are summed over, just add the product of the diagonal
      # elements of A and B
      # Assumes C starts set to 0
      c₁ = zero(ElC)
      for i in 1:min_dim
        c₁ += getdiagindex(A, i) * getdiagindex(B, i)
      end
      setdiagindex!(C, α * c₁ + β * getdiagindex(C, 1), 1)
    else
      # not all indices are summed over, set the diagonals of the result
      # to the product of the diagonals of A and B
      # TODO: should we make this return a Diag storage?
      for i in 1:min_dim
        setdiagindex!(
          C, α * getdiagindex(A, i) * getdiagindex(B, i) + β * getdiagindex(C, i), i
        )
      end
    end
  else
    # Most general contraction
    if convert_to_dense
      contract!(C, Clabels, dense(A), Alabels, B, Blabels, α, β)
    else
      if !isone(α) || !iszero(β)
        error(
          "`contract!(::DenseTensor, ::DiagTensor, ::DenseTensor, α, β; convert_to_dense = false)` with `α ≠ 1` or `β ≠ 0` is not currently supported. You can call it with `convert_to_dense = true` instead.",
        )
      end
      astarts = zeros(Int, length(Alabels))
      bstart = 0
      cstart = 0
      b_cstride = 0
      nbu = 0
      for ib in 1:length(Blabels)
        ia = findfirst(==(Blabels[ib]), Alabels)
        if !isnothing(ia)
          b_cstride += stride(B, ib)
          bstart += astarts[ia] * stride(B, ib)
        else
          nbu += 1
        end
      end

      c_cstride = 0
      for ic in 1:length(Clabels)
        ia = findfirst(==(Clabels[ic]), Alabels)
        if !isnothing(ia)
          c_cstride += stride(C, ic)
          cstart += astarts[ia] * stride(C, ic)
        end
      end

      # strides of the uncontracted dimensions of
      # B
      bustride = zeros(Int, nbu)
      custride = zeros(Int, nbu)
      # size of the uncontracted dimensions of
      # B, to be used in CartesianIndices
      busize = zeros(Int, nbu)
      n = 1
      for ib in 1:length(Blabels)
        if Blabels[ib] > 0
          bustride[n] = stride(B, ib)
          busize[n] = size(B, ib)
          ic = findfirst(==(Blabels[ib]), Clabels)
          custride[n] = stride(C, ic)
          n += 1
        end
      end

      boffset_orig = 1 - sum(strides(B))
      coffset_orig = 1 - sum(strides(C))
      cartesian_inds = CartesianIndices(Tuple(busize))
      for inds in cartesian_inds
        boffset = boffset_orig
        coffset = coffset_orig
        for i in 1:nbu
          ii = inds[i]
          boffset += ii * bustride[i]
          coffset += ii * custride[i]
        end
        c = zero(ElC)
        for j in 1:diaglength(A)
          # With α == 0 && β == 1
          C[cstart + j * c_cstride + coffset] +=
            getdiagindex(A, j) * B[bstart + j * b_cstride + boffset]
          # XXX: not sure if this is correct
          #C[cstart+j*c_cstride+coffset] += α * getdiagindex(A, j)* B[bstart+j*b_cstride+boffset] + β * C[cstart+j*c_cstride+coffset]
        end
      end
    end
  end
  #end # @timeit
end

function contract!(
  C::DenseTensor,
  Clabels,
  A::DenseTensor,
  Alabels,
  B::DiagTensor,
  Blabels,
  α::Number=one(eltype(C)),
  β::Number=zero(eltype(C)),
)
  return contract!(C, Clabels, B, Blabels, A, Alabels, α, β)
end

function show(io::IO, mime::MIME"text/plain", T::DiagTensor)
  summary(io, T)
  print_tensor(io, T)
  return nothing
end

function HDF5.write(
  parent::Union{HDF5.File,HDF5.Group}, name::String, D::Store
) where {Store<:Diag}
  g = create_group(parent, name)
  attributes(g)["type"] = "Diag{$(eltype(Store)),$(datatype(Store))}"
  attributes(g)["version"] = 1
  if eltype(D) != Nothing
    write(g, "data", D.data)
  end
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{Store}
) where {Store<:Diag}
  g = open_group(parent, name)
  ElT = eltype(Store)
  VecT = datatype(Store)
  typestr = "Diag{$ElT,$VecT}"
  if read(attributes(g)["type"]) != typestr
    error("HDF5 group or file does not contain $typestr data")
  end
  if ElT == Nothing
    return Dense{Nothing}()
  end
  # Attribute __complex__ is attached to the "data" dataset
  # by the h5 library used by C++ version of ITensor:
  if haskey(attributes(g["data"]), "__complex__")
    M = read(g, "data")
    nelt = size(M, 1) * size(M, 2)
    data = Vector(reinterpret(ComplexF64, reshape(M, nelt)))
  else
    data = read(g, "data")
  end
  return Diag{ElT,VecT}(data)
end
