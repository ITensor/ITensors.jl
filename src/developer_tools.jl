
"""
inspectQNITensor is a developer-level debugging tool 
to look at internals or properties of QNITensors
"""
function inspectQNITensor(T::ITensor, is::QNIndexSet)
  #@show T.store.blockoffsets
  #@show T.store.data
  println("Block fluxes:")
  for b in nzblocks(T)
    @show flux(T, b)
  end
end
inspectQNITensor(T::ITensor, is::IndexSet) = nothing
inspectQNITensor(T::ITensor) = inspectQNITensor(T, inds(T))

"""
    pause()

Pauses execution until a key (other than 'q') is pressed.
Entering 'q' exits the program. The `pause()` function
is useful for inspecting output of programs at certain
points while giving the option to continue.
"""
function pause()
  print(stdout, "(Paused) ")
  c = read(stdin, 1)
  c == UInt8[0x71] && exit(0)
  return nothing
end


#######################################################################
#
# Developer ITensor functions
#

"""
    array(T::ITensor)

Given an ITensor `T`, returns
an Array with a copy of the ITensor's elements,
or a view in the case the the ITensor's storage is Dense.

The ordering of the elements in the Array, in
terms of which Index is treated as the row versus
column, depends on the internal layout of the ITensor.

!!! warning
    This method is intended for developer use
    only and not recommended for use in ITensor applications
    unless you know what you are doing (for example
    you are certain of the memory ordering of the ITensor
    because you permuted the indices into a certain order).

See also [`matrix`](@ref), [`vector`](@ref).
"""
array(T::ITensor) = array(tensor(T))

"""
    array(T::ITensor, inds...)

Convert an ITensor `T` to an Array.

The ordering of the elements in the Array are specified
by the input indices `inds`. This tries to avoid copying
of possible (i.e. may return a view of the original
data), for example if the ITensor's storage is Dense
and the indices are already in the specified ordering
so that no permutation is required.

!!! warning
    Note that in the future we may return specialized
    AbstractArray types for certain storage types,
    for example a `LinearAlgebra.Diagonal` type for
    an ITensor with `Diag` storage. The specific storage
    type shouldn't be relied upon.

See also [`matrix`](@ref), [`vector`](@ref).
"""
array(T::ITensor, inds...) = array(permute(T, inds...; allow_alias=true))

"""
    matrix(T::ITensor)

Given an ITensor `T` with two indices, returns
a Matrix with a copy of the ITensor's elements,
or a view in the case the ITensor's storage is Dense.

The ordering of the elements in the Matrix, in
terms of which Index is treated as the row versus
column, depends on the internal layout of the ITensor.

!!! warning
    This method is intended for developer use
    only and not recommended for use in ITensor applications
    unless you know what you are doing (for example
    you are certain of the memory ordering of the ITensor
    because you permuted the indices into a certain order).

See also [`array`](@ref), [`vector`](@ref).
"""
function matrix(T::ITensor)
  ndims(T) != 2 && throw(DimensionMismatch())
  return array(tensor(T))
end

"""
    matrix(T::ITensor, inds...)

Convert an ITensor `T` to a Matrix.

The ordering of the elements in the Matrix are specified
by the input indices `inds`. This tries to avoid copying
of possible (i.e. may return a view of the original
data), for example if the ITensor's storage is Dense
and the indices are already in the specified ordering
so that no permutation is required.

!!! warning
    Note that in the future we may return specialized
    AbstractArray types for certain storage types,
    for example a `LinearAlgebra.Diagonal` type for
    an ITensor with `Diag` storage. The specific storage
    type shouldn't be relied upon.

See also [`array`](@ref), [`vector`](@ref).
"""
matrix(T::ITensor, inds...) = matrix(permute(T, inds...; allow_alias=true))

"""
    vector(T::ITensor)

Given an ITensor `T` with one index, returns
a Vector with a copy of the ITensor's elements,
or a view in the case the ITensor's storage is Dense.

See also [`array`](@ref), [`matrix`](@ref).
"""
function vector(T::ITensor)
  ndims(T) != 1 && throw(DimensionMismatch())
  return array(tensor(T))
end

"""
    vector(T::ITensor, inds...)

Convert an ITensor `T` to an Vector.

The ordering of the elements in the Array are specified
by the input indices `inds`. This tries to avoid copying
of possible (i.e. may return a view of the original
data), for example if the ITensor's storage is Dense
and the indices are already in the specified ordering
so that no permutation is required.

!!! warning
    Note that in the future we may return specialized
    AbstractArray types for certain storage types,
    for example a `LinearAlgebra.Diagonal` type for
    an ITensor with `Diag` storage. The specific storage
    type shouldn't be relied upon.

See also [`array`](@ref), [`matrix`](@ref).
"""
vector(T::ITensor, inds...) = vector(permute(T, inds...; allow_alias=true))
