function _contract(A::Tensor, B::Tensor)
  labelsA, labelsB = compute_contraction_labels(inds(A), inds(B))
  return contract(A, labelsA, B, labelsB)
  # TODO: Alternative to try (`noncommoninds` is too slow right now)
  #return _contract!!(EmptyTensor(Float64, _Tuple(noncommoninds(inds(A), inds(B)))), A, B)
end

function _contract(A::ITensor, B::ITensor)::ITensor
  C = itensor(_contract(tensor(A), tensor(B)))
  warnTensorOrder = get_warn_order()
  if !isnothing(warnTensorOrder) > 0 && order(C) >= warnTensorOrder
    println("Contraction resulted in ITensor with $(order(C)) indices, which is greater
            than or equal to the ITensor order warning threshold $warnTensorOrder.
            You can modify the threshold with macros like `@set_warn_order N`,
            `@reset_warn_order`, and `@disable_warn_order` or functions like
            `ITensors.set_warn_order(N::Int)`, `ITensors.reset_warn_order()`, and
            `ITensors.disable_warn_order()`.")
    # This prints a vector, not formatted well
    #show(stdout, MIME"text/plain"(), stacktrace())
    Base.show_backtrace(stdout, backtrace())
    println()
  end
  return C
end

_contract(T::ITensor, ::Nothing) = T

function can_combine_contract(A::ITensor, B::ITensor)::Bool
  return hasqns(A) &&
         hasqns(B) &&
         !iscombiner(A) &&
         !iscombiner(B) &&
         !isdiag(A) &&
         !isdiag(B)
end

function combine_contract(A::ITensor, B::ITensor)::ITensor
  # Combine first before contracting
  C::ITensor = if can_combine_contract(A, B)
    uniqueindsA = uniqueinds(A, B)
    uniqueindsB = uniqueinds(B, A)
    commonindsAB = commoninds(A, B)
    combinerA = isempty(uniqueindsA) ? nothing : combiner(uniqueindsA)
    combinerB = isempty(uniqueindsB) ? nothing : combiner(uniqueindsB)
    combinerAB = isempty(commonindsAB) ? nothing : combiner(commonindsAB)
    AC = _contract(_contract(A, combinerA), combinerAB)
    BC = _contract(_contract(B, combinerB), dag(combinerAB))
    CC = _contract(AC, BC)
    _contract(_contract(CC, dag(combinerA)), dag(combinerB))
  else
    _contract(A, B)
  end
  return C
end

"""
    A::ITensor * B::ITensor
    contract(A::ITensor, B::ITensor)

Contract ITensors A and B to obtain a new ITensor. This
contraction `*` operator finds all matching indices common
to A and B and sums over them, such that the result will
have only the unique indices of A and B. To prevent
indices from matching, their prime level or tags can be
modified such that they no longer compare equal - for more
information see the documentation on Index objects.

# Examples

```julia
i = Index(2,"index_i"); j = Index(4,"index_j"); k = Index(3,"index_k")

A = randomITensor(i,j)
B = randomITensor(j,k)
C = A * B # contract over Index j

A = randomITensor(i,i')
B = randomITensor(i,i'')
C = A * B # contract over Index i

A = randomITensor(i)
B = randomITensor(j)
C = A * B # outer product of A and B, no contraction

A = randomITensor(i,j,k)
B = randomITensor(k,i,j)
C = A * B # inner product of A and B, all indices contracted
```
"""
function (A::ITensor * B::ITensor)::ITensor
  return contract(A, B)
end

function contract(A::ITensor, B::ITensor)::ITensor
  NA::Int = ndims(A)
  NB::Int = ndims(B)
  if NA == 0 && NB == 0
    return (iscombiner(A) || iscombiner(B)) ? _contract(A, B) : ITensor(A[] * B[])
  elseif NA == 0
    return iscombiner(A) ? _contract(A, B) : A[] * B
  elseif NB == 0
    return iscombiner(B) ? _contract(B, A) : B[] * A
  else
    C = using_combine_contract() ? combine_contract(A, B) : _contract(A, B)
    return C
  end
end

function optimal_contraction_sequence(A::Union{Vector{<:ITensor},Tuple{Vararg{ITensor}}})
  if length(A) == 1
    return optimal_contraction_sequence(A[1])
  elseif length(A) == 2
    return optimal_contraction_sequence(A[1], A[2])
  elseif length(A) == 3
    return optimal_contraction_sequence(A[1], A[2], A[3])
  else
    return _optimal_contraction_sequence(A)
  end
end

optimal_contraction_sequence(A::ITensor) = Any[1]
optimal_contraction_sequence(A1::ITensor, A2::ITensor) = Any[1, 2]
function optimal_contraction_sequence(A1::ITensor, A2::ITensor, A3::ITensor)
  return optimal_contraction_sequence(inds(A1), inds(A2), inds(A3))
end
optimal_contraction_sequence(As::ITensor...) = _optimal_contraction_sequence(As)

_optimal_contraction_sequence(As::Tuple{<:ITensor}) = Any[1]
_optimal_contraction_sequence(As::Tuple{<:ITensor,<:ITensor}) = Any[1, 2]
function _optimal_contraction_sequence(As::Tuple{<:ITensor,<:ITensor,<:ITensor})
  return optimal_contraction_sequence(inds(As[1]), inds(As[2]), inds(As[3]))
end
function _optimal_contraction_sequence(As::Tuple{Vararg{ITensor}})
  return __optimal_contraction_sequence(As)
end

_optimal_contraction_sequence(As::Vector{<:ITensor}) = __optimal_contraction_sequence(As)

function __optimal_contraction_sequence(As)
  indsAs = [inds(A) for A in As]
  return optimal_contraction_sequence(indsAs)
end

function default_sequence()
  return using_contraction_sequence_optimization() ? "automatic" : "left_associative"
end

function contraction_cost(As::Union{Vector{<:ITensor},Tuple{Vararg{ITensor}}}; kwargs...)
  indsAs = [inds(A) for A in As]
  return contraction_cost(indsAs; kwargs...)
end

# TODO: provide `contractl`/`contractr`/`*ˡ`/`*ʳ` as shorthands for left associative and right associative contractions.
"""
    *(As::ITensor...; sequence = default_sequence(), kwargs...)
    *(As::Vector{<: ITensor}; sequence = default_sequence(), kwargs...)
    contract(As::ITensor...; sequence = default_sequence(), kwargs...)

Contract the set of ITensors according to the contraction sequence.

The default sequence is "automatic" if `ITensors.using_contraction_sequence_optimization()`
is true, otherwise it is "left_associative" (the ITensors are contracted from left to right).

You can change the default with `ITensors.enable_contraction_sequence_optimization()` and
`ITensors.disable_contraction_sequence_optimization()`.

For a custom sequence, the sequence should be provided as a binary tree where the leaves are
integers `n` specifying the ITensor `As[n]` and branches are accessed
by indexing with `1` or `2`, i.e. `sequence = Any[Any[1, 3], Any[2, 4]]`.
"""
function contract(tn::AbstractVector; kwargs...)
  return if all(x -> x isa ITensor, tn)
    contract(convert(Vector{ITensor}, tn); kwargs...)
  else
    deepcontract(tn; kwargs...)
  end
end

# Contract a tensor network such as:
# [A, B, [[C, D], [E, [F, G]]]]
deepcontract(t::ITensor, ts::ITensor...) = *(t, ts...)
function deepcontract(tn::AbstractVector)
  return deepcontract(deepcontract.(tn)...)
end

function contract(
  As::Union{Vector{ITensor},Tuple{Vararg{ITensor}}}; sequence=default_sequence(), kwargs...
)::ITensor
  if sequence == "left_associative"
    return foldl((A, B) -> contract(A, B; kwargs...), As)
  elseif sequence == "right_associative"
    return foldr((A, B) -> contract(A, B; kwargs...), As)
  elseif sequence == "automatic"
    return _contract(As, optimal_contraction_sequence(As); kwargs...)
  else
    return _contract(As, sequence; kwargs...)
  end
end

contract(As::ITensor...; kwargs...)::ITensor = contract(As; kwargs...)

_contract(As, sequence::Int) = As[sequence]

# Given a contraction sequence, contract the tensors recursively according
# to that sequence.
function _contract(As, sequence::AbstractVector; kwargs...)::ITensor
  return contract(_contract.((As,), sequence)...; kwargs...)
end

*(As::ITensor...; kwargs...)::ITensor = contract(As...; kwargs...)

function contract!(C::ITensor, A::ITensor, B::ITensor, α::Number, β::Number=0)::ITensor
  labelsCAB = compute_contraction_labels(inds(C), inds(A), inds(B))
  labelsC, labelsA, labelsB = labelsCAB
  CT = NDTensors.contract!!(
    tensor(C), _Tuple(labelsC), tensor(A), _Tuple(labelsA), tensor(B), _Tuple(labelsB), α, β
  )
  setstorage!(C, storage(CT))
  setinds!(C, inds(C))
  return C
end

function _contract!!(C::Tensor, A::Tensor, B::Tensor)
  labelsCAB = compute_contraction_labels(inds(C), inds(A), inds(B))
  labelsC, labelsA, labelsB = labelsCAB
  CT = NDTensors.contract!!(C, labelsC, A, labelsA, B, labelsB)
  return CT
end

# This is necessary for now since not all types implement contract!!
# with non-trivial α and β
function contract!(C::ITensor, A::ITensor, B::ITensor)::ITensor
  return settensor!(C, _contract!!(tensor(C), tensor(A), tensor(B)))
end

"""
    hadamard_product!(C::ITensor, A::ITensor, B::ITensor)
    hadamard_product(A::ITensor, B::ITensor)
    ⊙(A::ITensor, B::ITensor)

Elementwise product of 2 ITensors with the same indices.

Alternative syntax `⊙` can be typed in the REPL with `\\odot <tab>`.
"""
function hadamard_product!(R::ITensor, T1::ITensor, T2::ITensor)
  if !hassameinds(T1, T2)
    error("ITensors must have some indices to perform Hadamard product")
  end
  # Permute the indices to the same order
  #if inds(A) ≠ inds(B)
  #  B = permute(B, inds(A))
  #end
  #tensor(C) .= tensor(A) .* tensor(B)
  map!((t1, t2) -> *(t1, t2), R, T1, T2)
  return R
end

# TODO: instead of copy, use promote(A, B)
function hadamard_product(A::ITensor, B::ITensor)
  Ac = copy(A)
  return hadamard_product!(Ac, Ac, B)
end

⊙(A::ITensor, B::ITensor) = hadamard_product(A, B)

function directsum_projectors!(D1::Tensor, D2::Tensor)
  d1 = size(D1, 1)
  for ii in 1:d1
    D1[ii, ii] = one(eltype(D1))
  end
  d2 = size(D2, 1)
  for jj in 1:d2
    D2[jj, d1 + jj] = one(eltype(D1))
  end
  return D1, D2
end

# Helper tensors for performing a partial direct sum
function directsum_projectors(
  elt1::Type{<:Number}, elt2::Type{<:Number}, i::Index, j::Index, ij::Index
)
  D1 = ITensor(elt1, QN(), dag(i), ij)
  D2 = ITensor(elt2, QN(), dag(j), ij)
  directsum_projectors!(tensor(D1), tensor(D2))
  return D1, D2
end

function check_directsum_inds(A::ITensor, I, B::ITensor, J)
  a = uniqueinds(A, I)
  b = uniqueinds(B, J)
  if !hassameinds(a, b)
    error("""In directsum, attemptying to direct sum ITensors A and B with indices:

          $(inds(A))

          and

          $(inds(B))

          over the indices

          $(I)

          and

          $(J)

          The indices not being direct summed must match, however they are

          $a

          and

          $b
          """)
  end
end

function _directsum(
  IJ::Nothing, A::ITensor, I, B::ITensor, J; tags=default_directsum_tags(A => I)
)
  N = length(I)
  (N != length(J)) &&
    error("In directsum(::ITensor, ::ITensor, ...), must sum equal number of indices")
  check_directsum_inds(A, I, B, J)
  # Fix the Index direction for QN indices
  # TODO: Define `getfirstind`?
  I = map(In -> getfirst(==(In), inds(A)), I)
  J = map(Jn -> getfirst(==(Jn), inds(B)), J)
  IJ = Vector{Base.promote_eltype(I, J)}(undef, N)
  for n in 1:N
    IJ[n] = directsum(I[n], J[n]; tags=tags[n])
  end
  return _directsum(IJ, A, I, B, J)
end

function _directsum(IJ, A::ITensor, I, B::ITensor, J; tags=nothing)
  N = length(I)
  (N != length(J)) &&
    error("In directsum(::ITensor, ::ITensor, ...), must sum equal number of indices")
  check_directsum_inds(A, I, B, J)
  # Fix the Index direction for QN indices
  # TODO: Define `getfirstind`?
  I = map(In -> getfirst(==(In), inds(A)), I)
  J = map(Jn -> getfirst(==(Jn), inds(B)), J)
  for n in 1:N
    # TODO: Pass the entire `datatype` instead of just the `eltype`.
    D1, D2 = directsum_projectors(eltype(A), eltype(B), I[n], J[n], IJ[n])
    A *= adapt(datatype(A), D1)
    B *= adapt(datatype(B), D2)
  end
  C = A + B
  return C => IJ
end

to_inds(i::Index) = (i,)
to_inds(i::Indices) = i
to_inds(::Nothing) = nothing

function __directsum(
  ij, A::ITensor, i::Index, B::ITensor, j::Index; tags=default_directsum_tags(A => i)
)
  C, (ij,) = _directsum(to_inds(ij), A, to_inds(i), B, to_inds(j); tags=[tags])
  return C => ij
end

function _directsum(ij::Nothing, A::ITensor, i::Index, B::ITensor, j::Index; kwargs...)
  return __directsum(ij, A, i, B, j; kwargs...)
end

function _directsum(ij::Index, A::ITensor, i::Index, B::ITensor, j::Index; kwargs...)
  return __directsum(ij, A, i, B, j; kwargs...)
end

function default_directsum_tags(A_and_I::Pair{ITensor})
  return ["sum$i" for i in 1:length(last(A_and_I))]
end

function default_directsum_tags(A_and_I::Pair{ITensor,<:Index})
  return "sum"
end

"""
    directsum(A::Pair{ITensor}, B::Pair{ITensor}, ...; tags)

    directsum(output_inds, A::Pair{ITensor}, B::Pair{ITensor}, ...; tags)

Given a list of pairs of ITensors and indices, perform a partial
direct sum of the tensors over the specified indices. Indices that are
not specified to be summed must match between the tensors.

If all indices are specified then the operation is equivalent to creating
a block diagonal tensor.

Returns the ITensor representing the partial direct sum as well as the new
direct summed indices. The tags of the direct summed indices are specified
by the keyword arguments.

Optionally, pass the new direct summed indices of the output tensor as the
first argument (either a single Index or a collection), which must be proper
direct sums of the input indices that are specified to be direct summed.

See Section 2.3 of https://arxiv.org/abs/1405.7786 for a definition of a partial
direct sum of tensors.

# Examples
```julia
x = Index(2, "x")
i1 = Index(3, "i1")
j1 = Index(4, "j1")
i2 = Index(5, "i2")
j2 = Index(6, "j2")

A1 = randomITensor(x, i1)
A2 = randomITensor(x, i2)
S, s = directsum(A1 => i1, A2 => i2)
dim(s) == dim(i1) + dim(i2)

i1i2 = directsum(i1, i2)
S = directsum(i1i2, A1 => i1, A2 => i2)
hasind(S, i1i2)

A3 = randomITensor(x, j1)
S, s = directsum(A1 => i1, A2 => i2, A3 => j1)
dim(s) == dim(i1) + dim(i2) + dim(j1)

A1 = randomITensor(i1, x, j1)
A2 = randomITensor(x, j2, i2)
S, s = directsum(A1 => (i1, j1), A2 => (i2, j2); tags = ["sum_i", "sum_j"])
length(s) == 2
dim(s[1]) == dim(i1) + dim(i2)
dim(s[2]) == dim(j1) + dim(j2)
```
"""
function directsum(
  A_and_I::Pair{ITensor},
  B_and_J::Pair{ITensor},
  C_and_K::Pair{ITensor},
  itensor_and_inds...;
  tags=default_directsum_tags(A_and_I),
)
  return directsum(nothing, A_and_I, B_and_J, C_and_K, itensor_and_inds...; tags)
end

function directsum(
  output_inds::Nothing,
  A_and_I::Pair{ITensor},
  B_and_J::Pair{ITensor},
  C_and_K::Pair{ITensor},
  itensor_and_inds...;
  tags=default_directsum_tags(A_and_I),
)
  return directsum(
    output_inds,
    directsum(nothing, A_and_I, B_and_J; tags),
    C_and_K,
    itensor_and_inds...;
    tags,
  )
end

function directsum(
  output_inds::Union{Index,Indices},
  A_and_I::Pair{ITensor},
  B_and_J::Pair{ITensor},
  C_and_K::Pair{ITensor},
  itensor_and_inds...;
  tags=default_directsum_tags(A_and_I),
)
  return directsum(
    output_inds,
    directsum(nothing, A_and_I, B_and_J; tags),
    C_and_K,
    itensor_and_inds...;
    tags,
  )
end

function directsum(A_and_I::Pair{ITensor}, B_and_J::Pair{ITensor}; kwargs...)
  return directsum(nothing, A_and_I, B_and_J; kwargs...)
end

function directsum(
  output_inds::Nothing, A_and_I::Pair{ITensor}, B_and_J::Pair{ITensor}; kwargs...
)
  return _directsum(output_inds, A_and_I..., B_and_J...; kwargs...)
end

function directsum(
  output_inds::Union{Index,Indices},
  A_and_I::Pair{ITensor},
  B_and_J::Pair{ITensor};
  kwargs...,
)
  return first(_directsum(output_inds, A_and_I..., B_and_J...; kwargs...))
end

const ⊕ = directsum

"""
    apply(A::ITensor, B::ITensor)
    (A::ITensor)(B::ITensor)
    product(A::ITensor, B::ITensor)

Get the product of ITensor `A` and ITensor `B`, which
roughly speaking is a matrix-matrix product, a
matrix-vector product, or a vector-matrix product,
depending on the index structure.

There are three main modes:

1. Matrix-matrix product. In this case, ITensors `A`
and `B` have shared indices that come in pairs of primed
and unprimed indices. Then, `A` and `B` are multiplied
together, treating them as matrices from the unprimed
to primed indices, resulting in an ITensor `C` that
has the same pairs of primed and unprimed indices.
For example:
```
s1'-<-----<-s1            s1'-<-----<-s1   s1'-<-----<-s1
      |C|      = product(       |A|              |B|      )
s2'-<-----<-s2            s2'-<-----<-s2 , s2'-<-----<-s2
```
Essentially, this is implemented as
`C = mapprime(A', B, 2 => 1)`.
If there are dangling indices that are not shared between
`A` and `B`, a "batched" matrix multiplication is
performed, i.e.:
```
       j                         j
       |                         |
s1'-<-----<-s1            s1'-<-----<-s1   s1'-<-----<-s1
      |C|      = product(       |A|              |B|      )
s2'-<-----<-s2            s2'-<-----<-s2 , s2'-<-----<-s2
```
In addition, if there are shared dangling indices,
they are summed over:
```
                                    j                j
                                    |                |
s1'-<-----<-s1               s1'-<-----<-s1   s1'-<-----<-s1
      |C|      = Σⱼ product(       |A|              |B|      )
s2'-<-----<-s2               s2'-<-----<-s2 , s2'-<-----<-s2
```
where the sum is not performed as an explicitly
for-loop, but as part of a single tensor contraction.

2. Matrix-vector product. In this case, ITensor `A`
has pairs of primed and unprimed indices, and ITensor
`B` has unprimed indices that are shared with `A`.
Then, `A` and `B` are multiplied as a matrix-vector
product, and the result `C` has unprimed indices.
For example:
```
s1-<----            s1'-<-----<-s1   s1-<----
     |C| = product(       |A|             |B| )
s2-<----            s2'-<-----<-s2 , s2-<----
```
Again, like in the matrix-matrix product above, you can have
dangling indices to do "batched" matrix-vector products, or
sum over a batch of matrix-vector products.

3. Vector-matrix product. In this case, ITensor `B`
has pairs of primed and unprimed indices, and ITensor
`A` has unprimed indices that are shared with `B`.
Then, `B` and `A` are multiplied as a matrix-vector
product, and the result `C` has unprimed indices.
For example:
```
---<-s1            ----<-s1   s1'-<-----<-s1
|C|     = product( |A|              |B|      )
---<-s2            ----<-s2 , s2'-<-----<-s2
```
Again, like in the matrix-matrix product above, you can have
dangling indices to do "batched" vector-matrix products, or
sum over a batch of vector-matrix products.

4. Vector-vector product. In this case, ITensors `A`
and `B` share unprimed indices.
Then, `B` and `A` are multiplied as a vector-vector
product, and the result `C` is a scalar ITensor.
For example:
```
---            ----<-s1   s1-<----
|C| = product( |A|             |B| )
---            ----<-s2 , s2-<----
```
Again, like in the matrix-matrix product above, you can have
dangling indices to do "batched" vector-vector products, or
sum over a batch of vector-vector products.
"""
function product(A::ITensor, B::ITensor; apply_dag::Bool=false)
  commonindsAB = commoninds(A, B; plev=0)
  isempty(commonindsAB) && error("In product, must have common indices with prime level 0.")
  common_paired_indsA = filterinds(
    i -> hasind(commonindsAB, i) && hasind(A, setprime(i, 1)), A
  )
  common_paired_indsB = filterinds(
    i -> hasind(commonindsAB, i) && hasind(B, setprime(i, 1)), B
  )

  if !isempty(common_paired_indsA)
    commoninds_pairs = unioninds(common_paired_indsA, common_paired_indsA')
  elseif !isempty(common_paired_indsB)
    commoninds_pairs = unioninds(common_paired_indsB, common_paired_indsB')
  else
    # vector-vector product
    apply_dag && error("apply_dag not supported for vector-vector product")
    return A * B
  end
  danglings_indsA = uniqueinds(A, commoninds_pairs)
  danglings_indsB = uniqueinds(B, commoninds_pairs)
  danglings_inds = unioninds(danglings_indsA, danglings_indsB)
  if hassameinds(common_paired_indsA, common_paired_indsB)
    # matrix-matrix product
    A′ = prime(A; inds=!danglings_inds)
    AB = mapprime(A′ * B, 2 => 1; inds=!danglings_inds)
    if apply_dag
      AB′ = prime(AB; inds=!danglings_inds)
      Adag = swapprime(dag(A), 0 => 1; inds=!danglings_inds)
      return mapprime(AB′ * Adag, 2 => 1; inds=!danglings_inds)
    end
    return AB
  elseif isempty(common_paired_indsA) && !isempty(common_paired_indsB)
    # vector-matrix product
    apply_dag && error("apply_dag not supported for matrix-vector product")
    A′ = prime(A; inds=!danglings_inds)
    return A′ * B
  elseif !isempty(common_paired_indsA) && isempty(common_paired_indsB)
    # matrix-vector product
    apply_dag && error("apply_dag not supported for vector-matrix product")
    return replaceprime(A * B, 1 => 0; inds=!danglings_inds)
  end
end

"""
    product(As::Vector{<:ITensor}, A::ITensor)

Product the ITensors pairwise.
"""
function product(As::Vector{<:ITensor}, B::ITensor; kwargs...)
  AB = B
  for A in As
    AB = product(A, AB; kwargs...)
  end
  return AB
end

# Alias apply with product
const apply = product

(A::ITensor)(B::ITensor) = apply(A, B)

const Apply{Args} = Applied{typeof(apply),Args}
