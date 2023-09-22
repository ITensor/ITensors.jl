# SmallVectors

## Introduction

A module that defines small (mutable and immutable) vectors with a maximum length. Externally they have a dynamic/runtime length, but internally they are backed by a statically sized vector. This makes it so that operations can be performed faster because they can remain on the stack, but it provides some more convenience compared to StaticArrays.jl where the length is encoded in the type.

## Examples

For example:
```julia
using NDTensors.SmallVectors

mv = MSmallVector{10}([1, 2, 3]) # Mutable vector with length 3, maximum length 10
push!(mv, 4)
mv[2] = 12
sort!(mv; rev=true)

v = SmallVector{10}([1, 2, 3]) # Immutable vector with length 3, maximum length 10
v = SmallVectors.push(v, 4)
v = SmallVectors.setindex(v, 12, 2)
v = SmallVectors.sort(v; rev=true)
```
This also has the advantage that you can efficiently store collections of `SmallVector`/`MSmallVector` that have different runtime lengths, as long as they have the same maximum length.

## List of functionality

`SmallVector` and `MSmallVector` are subtypes of `AbstractVector` and therefore can be used in `Base` `AbstractVector` functions, though `SmallVector` will fail for mutating functions like `setindex!` because it is immutable.

`MSmallVector` has specialized implementations of `Base` functions that involve resizing such as:
- `resize!`
- `push!`
- `pushfirst!`
- `pop!`
- `popfirst!`
- `append!`
- `prepend!`
- `insert!`
- `deleteat!`
which are guaranteed to not realocate memory, and instead just use the memory buffer that already exists, unlike Base's `Vector` which may have to reallocate memory depending on the operation. However, they will error if they involve operations that resize beyond the maximum length of the `MSmallVector`, which you can access with `SmallVectors.maxlength(v)`.

In addition, `SmallVector` and `MSmallVector` implement basic non-mutating operations such as:
- `SmallVectors.setindex`
, non-mutating resizing operations:
- `SmallVector.resize`
- `SmallVector.push`
- `SmallVector.pushfirst`
- `SmallVector.pop`
- `SmallVector.popfirst`
- `SmallVector.append`
- `SmallVector.prepend`
- `SmallVector.insert`
- `SmallVector.deleteat`
which output a new vector. In addition, it implements:
- `SmallVectors.circshift`
- `sort` (overloaded from `Base`).

Finally, it provides some new helpful functions that are not in `Base`:
- `SmallVectors.insertsorted[!]`
- `SmallVectors.insertsortedunique[!]`
- `SmallVectors.mergesorted[!]`
- `SmallVectors.mergesortedunique[!]`

## TODO

Add specialized overloads for:
- `splice[!]`
- `union[!]` (`∪`)
- `intersect[!]` (`∩`)
- `setdiff[!]`
- `symdiff[!]`
- `unique[!]`

Please let us know if there are other operations that would warrant specialized implmentations for `AbstractSmallVector`.
