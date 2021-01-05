For QN ITensors (ITensors with block sparse structure), ITensors.jl
provides two "modes" for contracting them. The default mode is treating
them as block sparse multi-dimensional arrays and contracting them
"as is".

In the second mode, the ITensors are reshaped into block
sparse matrices before contracting. Making use of the QN information
in the tensors, the second mode can lead to fewer blocks being contracted,
which can lead to a speedup depending on the number of blocks, the sizes
of the blocks, the order of the ITensors being contracted, etc.

Use the commands `ITensors.enable_combine_contract()` and `ITensors.disable_combine_contract()` to change between the different contraction modes.

Run the example using parity symmetry as follows:
```julia
julia> include("main.jl")

julia> main(Nmax = 8)
#################################################
# order = 2
#################################################

Contract:
  1.609 μs (51 allocations: 4.83 KiB)

Combine then contract:
  45.714 μs (888 allocations: 95.20 KiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 4
#################################################

Contract:
  4.285 μs (60 allocations: 9.66 KiB)

Combine then contract:
  109.065 μs (1631 allocations: 187.13 KiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 6
#################################################

Contract:
  40.041 μs (83 allocations: 55.81 KiB)

Combine then contract:
  288.204 μs (3889 allocations: 466.31 KiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 8
#################################################

Contract:
  302.242 μs (99 allocations: 467.00 KiB)

Combine then contract:
  1.039 ms (13404 allocations: 1.54 MiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 10
#################################################

Contract:
  3.580 ms (115 allocations: 3.21 MiB)

Combine then contract:
  5.208 ms (76632 allocations: 7.13 MiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 12
#################################################

Contract:
  45.379 ms (131 allocations: 21.20 MiB)

Combine then contract:
  31.002 ms (369134 allocations: 37.03 MiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 14
#################################################

Contract:
  670.578 ms (150 allocations: 184.92 MiB)

Combine then contract:
  144.652 ms (1576569 allocations: 165.32 MiB)

C_contract ≈ C_combine_contract = true

#################################################
# order = 16
#################################################

Contract:
  13.581 s (12583081 allocations: 4.12 GiB)

Combine then contract:
  1.211 s (14995656 allocations: 1.22 GiB)

C_contract ≈ C_combine_contract = true

```

