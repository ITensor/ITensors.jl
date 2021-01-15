TBLIS is a permutation-free tensor contraction library 
(https://github.com/devinamatthews/tblis). By default, ITensors.jl performs 
tensor contractions by first permuting the tensors 
so the contraction can be mapped to a matrix multiplication, which is then 
performed by BLAS. Instead, TBLIS avoids BLAS entirely, using a generalization of 
BLAS-like general matrix multiplication (GEMM) algorithms directly to tensor
contractions. See also other implementations of GEMM-like tensor contractions 
like GETT (https://github.com/HPAC/tccg).

To use TBLIS as a contraction backend in ITensors.jl, first install TBLIS.jl: https://github.com/mtfishman/TBLIS.jl

Then, set the number of TBLIS threads by setting the environment variable `TBLIS_NUM_THREADS`, for example by starting Julia with:
```
$ TBLIS_NUM_THREADS=4 julia 
```
Then, to tell ITensors.jl that you want to using TBLIS, type the command:
```julia
julia> using TBLIS
```
You can disable the use of TBLIS with the command:
```julia
julia> ITensors.disable_tblis()
```
and enable it again with the command:
```julia
julia> ITensors.enable_tblis()
```
and you can set the number of threads with:
```julia
julia> TBLIS.set_num_threads(4)
```

You can try out the example scripts like:
```julia
julia> include("contract.jl")

julia> include("1d_heisenberg.jl")
```

Note that in general, we have not found that TBLIS accelerates DMRG calculations.
We believe it is because many of the contractions involved in DMRG can be turned
directly into matrix multiplications without the need for permuting the data first,
so TBLIS doesn't gain an advantage since there is no permutation to avoid.

