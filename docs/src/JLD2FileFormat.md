# JLD2 File Format

This page documents the on-disk format used when reading and writing ITensors core
types and NDTensors storage types through the [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)
backend, provided by the `ITensorsJLD2Ext` and `NDTensorsJLD2Ext` package extensions.

## Round-tripping an ITensor

JLD2 is enabled automatically by loading JLD2 alongside ITensors:

```julia
using ITensors, JLD2
i = Index(3, "i")
j = Index(4, "j")
A = random_itensor(i, j)

jldsave("myfile.jld2"; A)
A_loaded = load("myfile.jld2", "A")
@assert A_loaded ≈ A
```

The same call works for `QN`, `TagSet`, `Index`, `QNIndex`, and ITensors with `Dense`,
`BlockSparse`, `Diag`, `DiagBlockSparse`, and `EmptyStorage` storage.

## The on-disk schema

The format is defined by a set of plain Julia structs in the main packages — one per
core type — and the JLD2 extension is a thin layer that maps each user-facing type
through `JLD2.writeas` / `JLD2.wconvert` / `JLD2.rconvert` to the corresponding schema
struct. Keeping the structs in the main packages means the type names recorded inside
the JLD2 file do not encode an extension module namespace, which keeps the file readable
even when no extension is loaded.

### Core ITensors types

Defined in `ITensors`:

  - [`ITensors.SerializedQNVal`](@ref)
  - [`ITensors.SerializedQN`](@ref)
  - [`ITensors.SerializedTagSet`](@ref)
  - [`ITensors.SerializedQNSpace`](@ref)
  - [`ITensors.SerializedIndex`](@ref)
  - [`ITensors.SerializedITensor`](@ref)

```@docs
ITensors.SerializedQNVal
ITensors.SerializedQN
ITensors.SerializedTagSet
ITensors.SerializedQNSpace
ITensors.SerializedIndex
ITensors.SerializedITensor
```

### NDTensors storage types

Defined in `NDTensors`:

  - [`NDTensors.SerializedDense`](@ref)
  - [`NDTensors.SerializedBlockSparse`](@ref)
  - [`NDTensors.SerializedDiag`](@ref)
  - [`NDTensors.SerializedUniformDiag`](@ref)
  - [`NDTensors.SerializedDiagBlockSparse`](@ref)
  - [`NDTensors.SerializedUniformDiagBlockSparse`](@ref)
  - [`NDTensors.SerializedEmptyStorage`](@ref)

```@docs
NDTensors.SerializedDense
NDTensors.SerializedBlockSparse
NDTensors.SerializedDiag
NDTensors.SerializedUniformDiag
NDTensors.SerializedDiagBlockSparse
NDTensors.SerializedUniformDiagBlockSparse
NDTensors.SerializedEmptyStorage
```

## Integer-width conventions

Every integer field in the schema has an explicit, fixed width so that the file does
not depend on the host word size and so that a reader knows exactly which HDF5 datatype
to expect:

  - `version :: UInt32` — schema version, matching `Base.VersionNumber`'s field width.
  - `Int64` for any logically-signed quantity that could be negative — `QN` charge values
    and modulus sentinel, index dimensions, prime level.
  - `Int8` for the `Arrow` direction on `Index`, with values `In = -1`, `Neither = 0`,
    `Out = +1`.
  - `UInt64` for the `Index` identifier (`id`).

## Block-sparse coordinates

`SerializedBlockSparse`, `SerializedDiagBlockSparse`, and `SerializedUniformDiagBlockSparse`
store block positions as a `Matrix{Int64}` named `block_indices` shaped `(ndims, num_blocks)`,
each column a block position. This is the COO convention used by Apache Arrow, PyData Sparse,
and PyTorch's sparse format. The tensor rank is implicit in `size(block_indices, 1)` and is
preserved by HDF5 even when `num_blocks == 0`, so a separate rank field is not stored.

`block_offsets` is a `Vector{Int64}` of length `num_blocks`, giving the offset into the
flat `data` buffer for each block.

## Versioning

Every schema struct carries a `version::UInt32` field. The current version is `1` for
every struct.

When a struct's layout needs to change in a backwards-incompatible way, the version is
incremented and the rconvert path branches on `s.version` to migrate older files. New
schema versions should be additive whenever possible (extra fields with defaults) so
that older readers can ignore unknown fields.

## Future direction: other backends

The `Serialized*` structs are intentionally backend-agnostic — they describe the on-disk
layout, not the JLD2-specific encoding mechanics. A future native HDF5 backend (using
[HDF5.jl](https://juliaio.github.io/HDF5.jl/stable/) directly) or a cross-language
reader (h5py, C++) can target the same schema by walking the struct fields and emitting
or parsing the obvious HDF5 datatype for each field (primitives, strings, 1-D and 2-D
integer arrays, named groups for nested structs). Today JLD2 is the only backend
shipped, and a separate, legacy HDF5 format documented on the
[HDF5 File Formats](@ref) page predates this schema; the two are not interchangeable.
