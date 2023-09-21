# SmallVectors

A module that defines small (mutable and immutable) vectors with a maximum length. Externally the have a dynamic (or in the case of immuatable vectors, runtime) length, but internally they are backed by a statically sized vector. This makes it so that operations can be performed faster because they can remain on the stack, but it provides some more convenience compared to StaticArrays.jl where the length is encoded in the type.

For example:
```julia
using NDTensors.SmallVectors
v = SmallVector{10}([1, 2, 3]) # Immutable vector with length 3, maximum length 10
v = push(v, 4)
v = setindex(v, 4, 4)
v = sort(v; rev=true)

mv = MSmallVector{10}([1, 2, 3]) # Mutable vector with length 3, maximum length 10
push!(mv, 4)
mv[2] = 12
sort!(mv; rev=true)
```
This also has the advantage that you can efficiently store collections of `SmallVector`/`MSmallVector` that have different runtime lengths, as long as they have the same maximum length.
