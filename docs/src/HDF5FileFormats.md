# HDF5 File Formats

This page lists the formats for the HDF5 representations of 
various types in the `ITensors` module. 

HDF5 is a portable file format which has a directory structure similar 
to a file system. In addition to containing "groups" (= directories)
and "datasets" (= files), groups can have "attributes"
appended to them, which are similar to 'tags' or 'keywords'. 
Unless otherwise specified, integers are 64 bit and are signed
(H5T\_STD\_I64LE) unless explicitly stated. (For example, the "id"
field of the `Index` type is stored as an unsigned 64 bit integer
(H5T\_STD\_U64LE).)

Each type in ITensor which is writeable to HDF5 is written
to its own group, with the name of the group either specified
by the user or specified to some default value when it is 
a subgroup of another ITensor type (for example, the `Index`
type saves its `TagSet` in a subgroup named "tags").

Each group corresponding to an ITensors type always carries
the following attributes:
* "type" --- a string such as `Index` or `TagSet` specifying the information 
  necessary to determine the type of the object saved to the HDF5 group
* "version" --- an integer specifying the file format version used to
  store the data. This version is in general different from the release
  version of ITensors.jl. The purpose of the version number is to aid 
  in maintaining backwards compatibility, while allowing the format
  to be occasionally changed.

The C++ version of ITensor uses exactly the same file formats listed below,
for the purpose of interoperability with the Julia version of ITensor,
even though conventions such as the "type" field values are Julia-centric.


## [TagSet](@id tagset_hdf5)

HDF5 file format for the `ITensors.TagSet` type.

Attributes:
* "version" = 1
* "type" = "TagSet"

Datasets and Subgroups:
* "tags" [string] = a comma separated string of the tags in the `TagSet`


## [QN](@id qn_hdf5)

HDF5 file format for the `ITensors.QN` type.

Attributes:
* "version" = 1
* "type" = "QN"

Datasets and Subgroups:
* "names" [group] = array of strings (length 4) of names of quantum numbers
* "vals" [group] = array of integers (length 4) of quantum number values 
* "mods" [group] = array of integers (length 4) of moduli of quantum numbers


## [QNBlocks](@id qnblocks_hdf5)

HDF5 file format for the `ITensors.QNBlocks` type.
(Note: `QNBlocks` is equivalent to `Vector{Pair{QN, Int64}}`.)

Attributes:
* "version" = 1
* "type" = "QNBlocks"

Datasets and Subgroups:
* "length" [integer] = the number of blocks (length of Vector)
* "dims" [group] = array of (integer) dimensions of each block
* "QN[n]" [group] = these groups "QN[1]", "QN[2]", etc.
  correspond to the [QN](@ref qn_hdf5) of each block


## [Index](@id index_hdf5)

HDF5 file format for the `ITensors.Index` type.

Attributes:
* "version" = 1
* "type" = "Index"
* "space_type" = "Int" if the Index is a regular, dense Index or "QNBlocks" if the Index 
  is a QNIndex (carries QN subspace information)

Datasets and Subgroups:
* "id" [unsigned integer] = id number of the Index
* "dim" [integer] = dimension of the Index
* "dir" [integer] = arrow direction of the Index, +1 for `ITensors.Out` and -1 for `ITensors.In`
* "plev" [integer] = prime level of the Index
* "tags" [group] = the [TagSet](@ref tagset_hdf5) of the Index

Optional Datasets and Subgroups:
* "space" [group] = if the `"space_type"` attribute is "QNBlocks", this group
  is present and represents a [QNBlocks](@ref qnblocks_hdf5) object



## [IndexSet](@id indexset_hdf5)

HDF5 file format for types in the Union type `ITensors.Indices`
which includes `IndexSet` and tuples of Index objects.

Attributes:
* "version" = 1
* "type" = "IndexSet"

Datasets and Subgroups:
* "length" [integer] = number of indices
* "index_n" [group] = for n=1 to n=length each of these groups contains an Index


## [ITensor](@id itensor_hdf5)

HDF5 file format for the `ITensors.ITensor` type.

Attributes:
* "version" = 1
* "type" = "ITensor"

Datasets and Subgroups:
* "inds" [group] = indices of the ITensor
* "storage" [group] = storage of the ITensor
  (note that some earlier versions of ITensors.jl may call this group "store")


## [NDTensors.Dense](@id dense_hdf5)

HDF5 file format for objects which are subtypes of `ITensors.NDTensors.Dense`.

Attributes:
* "version" = 1
* "type" = "Dense{Float64}" or "Dense{ComplexF64}"

Datasets and Subgroups:
* "data" = array of either real or complex values (in the same dataset format used
  by the HDF5.jl library for storing `Vector{Float64}` or `Vector{ComplexF64}`)


## [NDTensors.BlockSparse](@id blocksparse_hdf5)

HDF5 file format for objects which are subtypes of `ITensors.NDTensors.BlockSparse`.

Attributes:
* "version" = 1
* "type" = "BlockSparse{Float64}" or "BlockSparse{ComplexF64}"

Datasets and Subgroups:
* "ndims" [integer] = number of dimensions (order) of the tensor
* "offsets" = block offset data flattened into an array of integers
* "data" = array of either real or complex values (in the same dataset format used
  by the HDF5.jl library for storing `Vector{Float64}` or `Vector{ComplexF64}`)


## [MPS](@id mps_hdf5)

HDF5 file format for `ITensors.MPS`

Attributes:
* "version" = 1
* "type" = "MPS"

Datasets and Subgroups:
* "length" [integer] = number of tensors of the MPS
* "rlim" [integer] = right orthogonality limit
* "llim" [integer] = left orthogonality limit
* "MPS[n]" [group,ITensor] = each of these groups, where n=1,...,length, stores the nth ITensor of the MPS


## [MPO](@id mpo_hdf5)

HDF5 file format for `ITensors.MPO`

Attributes:
* "version" = 1
* "type" = "MPO"

Datasets and Subgroups:
* "length" [integer] = number of tensors of the MPO
* "rlim" [integer] = right orthogonality limit
* "llim" [integer] = left orthogonality limit
* "MPO[n]" [group,ITensor] = each of these groups, where n=1,...,length, stores the nth ITensor of the MPO



