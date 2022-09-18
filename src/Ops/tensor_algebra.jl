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

# Helper tensors for performing a partial direct sum
function directsum_itensors(i::Index, j::Index, ij::Index)
  S1 = zeros(dim(i), dim(ij))
  for ii in 1:dim(i)
    S1[ii, ii] = 1
  end
  S2 = zeros(dim(j), dim(ij))
  for jj in 1:dim(j)
    S2[jj, dim(i) + jj] = 1
  end
  D1 = itensor(S1, dag(i), ij)
  D2 = itensor(S2, dag(j), ij)
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

function _directsum(A::ITensor, I, B::ITensor, J; tags=["sum$i" for i in 1:length(I)])
  N = length(I)
  (N != length(J)) &&
    error("In directsum(::ITensor, ::ITensor, ...), must sum equal number of indices")
  check_directsum_inds(A, I, B, J)
  IJ = Vector{Base.promote_eltype(I, J)}(undef, N)
  for n in 1:N
    In = I[n]
    Jn = J[n]
    In = dir(A, In) != dir(In) ? dag(In) : In
    Jn = dir(B, Jn) != dir(Jn) ? dag(Jn) : Jn
    IJn = directsum(In, Jn; tags=tags[n])
    D1, D2 = directsum_itensors(In, Jn, IJn)
    IJ[n] = IJn
    A *= D1
    B *= D2
  end
  C = A + B
  return C => IJ
end

function _directsum(A::ITensor, i::Index, B::ITensor, j::Index; tags="sum")
  C, (ij,) = _directsum(A, (i,), B, (j,); tags=[tags])
  return C => ij
end

function directsum(A_and_I::Pair{ITensor}, B_and_J::Pair{ITensor}; kwargs...)
  return _directsum(A_and_I..., B_and_J...; kwargs...)
end

function default_directsum_tags(A_and_I::Pair{ITensor})
  return ["sum$i" for i in 1:length(last(A_and_I))]
end

function default_directsum_tags(A_and_I::Pair{ITensor,<:Index})
  return "sum"
end

"""
    directsum(A::Pair{ITensor}, B::Pair{ITensor}, ...; tags)

Given a list of pairs of ITensors and indices, perform a partial
direct sum of the tensors over the specified indices. Indices that are
not specified to be summed must match between the tensors.

If all indices are specified then the operation is equivalent to creating
a block diagonal tensor.

Returns the ITensor representing the partial direct sum as well as the new
direct summed indices. The tags of the direct summed indices are specified
by the keyword arguments.

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
  return directsum(directsum(A_and_I, B_and_J; tags), C_and_K, itensor_and_inds...; tags)
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
