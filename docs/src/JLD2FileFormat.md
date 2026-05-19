# JLD2 File Format

This page documents the on-disk format used when reading and writing ITensors core
types and NDTensors storage types through the [JLD2.jl](https://github.com/JuliaIO/JLD2.jl)
backend, provided by the `ITensorsJLD2Ext` and `NDTensorsJLD2Ext` package extensions.

## Why a schema layer

JLD2's default behaviour is to serialize an object using its in-memory struct layout —
each field is written under its current Julia field name and type. That works for one-off
storage, but it ties the file format to the in-memory type definitions: renaming a field,
splitting a type, or moving a struct between modules invalidates older files.

The JLD2 extension shipped here interposes a stable on-disk schema between in-memory types
and the file. Each user-facing type (`ITensor`, `Index`, `QN`, `TagSet`, and the NDTensors
storage types) is written through a `Serialized*` struct that owns the on-disk layout.
Every schema struct carries an explicit `version::UInt32`, so when an in-memory type's
representation changes the loader can migrate older files instead of failing to
reconstruct them. The schema structs are kept deliberately simple — plain fixed-width
integers, strings, and arrays — so the file is portable across host word sizes and easy
for future readers to consume.

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
core type — and the JLD2 extension is a thin layer that bridges JLD2's `writeas`
mechanism to the backend-agnostic `serialized_type` mapping defined in the main package.
Keeping the structs and the conversion logic in the main packages means the type names
recorded inside the JLD2 file do not encode an extension module namespace, and other
serialization backends can reuse the same machinery without depending on JLD2.

### Backend-agnostic layer

  - [`ITensors.serialized_type`](@ref) and [`NDTensors.serialized_type`](@ref): type-level
    map from an in-memory type to its [`SerializedX`](@ref) schema struct.
  - `Base.convert` overloads (defined alongside `serialized_type`) do the value-level
    transform between the in-memory and `Serialized*` representations in both directions.

JLD2's default `wconvert` / `rconvert` already delegate to `Base.convert`, so the
extension itself only needs to provide the `JLD2.writeas` declarations bridging the
two namespaces:

```julia
JLD2.writeas(::Type{T}) where {T <: NDTensors.TensorStorage} = NDTensors.serialized_type(T)
JLD2.writeas(::Type{T}) where {T <: ITensors.Index}          = ITensors.serialized_type(T)
# ...
```

### Core ITensors types

Defined in `ITensors`:

  - [`ITensors.SerializedQNVal`](@ref)
  - [`ITensors.SerializedQN`](@ref)
  - [`ITensors.SerializedTagSet`](@ref)
  - [`ITensors.SerializedQNSpace`](@ref)
  - [`ITensors.SerializedIndex`](@ref)
  - [`ITensors.SerializedITensor`](@ref)

```@docs
ITensors.serialized_type
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
NDTensors.serialized_type
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
store block positions as a `Matrix{Int64}` named `block_indices` shaped
`(num_blocks, ndims)`. Each row is one block's position tuple; column `j` holds that
block's position along axis `j`. The tensor rank is implicit in `size(block_indices, 2)`
and is preserved by HDF5 even when `num_blocks == 0`, so a separate rank field is not
stored.

`block_offsets` is a `Vector{Int64}` of length `num_blocks`, giving the offset into the
flat `data` buffer for each block.

## Versioning

Every schema struct carries a `version::UInt32` field. Versions for each struct start at
`1` and are bumped independently as that struct's layout evolves.

When a struct's layout needs to change in a backwards-incompatible way, the version is
incremented and the `Base.convert(::Type{InMemoryT}, ::SerializedT)` method branches on
`s.version` to migrate older files. New schema versions should be additive whenever
possible (extra fields with defaults) so older readers can ignore unknown fields.

## Future direction: other backends

The `Serialized*` structs are intentionally backend-agnostic — they describe the on-disk
layout, not the JLD2-specific encoding mechanics. In the future, we may use them as a
reference schema for cross-language reading or writing of ITensor types (for example
from h5py or C++). Additionally, we may use them to aid with interoperability between
ITensor objects saved with JLD2 and the separate ITensor [HDF5 File Formats](@ref).
