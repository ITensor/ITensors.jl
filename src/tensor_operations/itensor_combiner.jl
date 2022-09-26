function combiner(is::Indices; kwargs...)
  tags = get(kwargs, :tags, "CMB,Link")
  new_ind = Index(prod(dims(is)), tags)
  new_is = (new_ind, is...)
  return itensor(Combiner(), new_is)
end

combiner(is...; kwargs...) = combiner(indices(is...); kwargs...)
combiner(i::Index; kwargs...) = combiner((i,); kwargs...)

# Special case when no indices are combined (useful for generic code)
function combiner(; kwargs...)
  return itensor(Combiner(), ())
end

"""
    combinedind(C::ITensor)

Given a combiner ITensor, return the Index which is
the "combined" index that is made out of merging
the other indices given to the combiner when it is made

For more information, see the `combiner` function.
"""
function combinedind(T::ITensor)
  if storage(T) isa Combiner && order(T) > 0
    return inds(T)[1]
  end
  return nothing
end

# TODO: add iscombiner(::Tensor) to NDTensors
iscombiner(T::ITensor)::Bool = (storage(T) isa Combiner)

@doc """
    combiner(inds::Indices; kwargs...)

Make a combiner ITensor which combines the indices (of type Index)
into a single, new Index whose size is the product of the indices
given. For example, given indices `i1,i2,i3` the combiner will have
these three indices plus an additional one whose dimension is the
product of the dimensions of `i1,i2,i3`.

Internally, a combiner ITensor uses a special storage type which
means it does not hold actual tensor elements but just information
about how to combine the indices into a single Index. Taking a product
of a regular ITensor with a combiner uses special fast algorithms to
combine the indices.

To obtain the new, combined Index that the combiner makes out of
the indices it is given, use the `combinedind` function.

To undo or reverse the combining process, uncombining the Index back
into the original ones, contract the tensor having the combined Index
with the conjugate or `dag` of the combiner. (If the combiner is an ITensor
`C`, multiply by `dag(C)`.)

### Example
```
# Combine indices i and k into a new Index ci
T = randomITensor(i,j,k)
C = combiner(i,k)
CT = C * T
ci = combinedind(C)

# Uncombine ci back into i and k
TT = dag(C) * CT

# TT will be the same as T
@show norm(TT - T) ≈ 0.0
```

              i  j  k
              |  |  |
     T   =    =======

              ci  i  k
              |   |  |
     C   =    ========

              ci  j
              |   |
     C * T =  =====

""" combiner
