using ITensors

function fusion_tree(s::Vector{<:Index})
  n = length(s)
  Cs = Vector{ITensor}(undef, n - 1)
  cj = s[1]
  for j in 1:(n - 1)
    fuse_inds = (cj, s[j + 1])
    Cj = combiner(fuse_inds...)
    Cs[j] = Cj
    cj = uniqueind(Cj, fuse_inds)
  end
  return Cs
end

function fuse_inds(A::MPS, fusion_tree::Vector{ITensor})
  n = length(A)
  A_fused = A[1]
  for j in 2:n
    A_fused = A_fused * A[j] * fusion_tree[j - 1]
  end
  return A_fused
end

function fuse_inds(A::MPO, fusion_tree::Vector{ITensor})
  n = length(A)
  A_fused = A[1]
  for j in 2:n
    A_fused = A_fused * A[j] * dag(fusion_tree[j - 1]) * fusion_tree[j - 1]'
  end
  return A_fused
end

function fusion_tree_binary_layer(s::Vector{IndexT}; layer=1) where {IndexT<:Index}
  n = length(s)
  Cs = ITensor[]
  cs = IndexT[]
  for j in 1:2:(n - 1)
    fuse_inds = (s[j], s[j + 1])
    Cj = combiner(fuse_inds...; tags="n=$(j)⊗$(j + 1),l=$(layer)")
    push!(Cs, Cj)
    cj = uniqueind(Cj, fuse_inds)
    push!(cs, cj)
  end
  if isodd(n)
    push!(cs, last(s))
  end
  return Cs, cs
end

function fusion_tree_binary(s::Vector{<:Index}; depth=ceil(Int, log2(length(s))))
  Cs = Vector{ITensor}[]
  c_layer = s
  for layer in 1:depth
    C_layer, c_layer = fusion_tree_binary_layer(c_layer; layer)
    push!(Cs, C_layer)
  end
  return Cs
end

function fuse_tensors(A::MPS, fusion_tree_layer::Vector{ITensor}, j::Int)
  return A[j] * A[j + 1] * fusion_tree_layer[(j + 1) ÷ 2]
end

function fuse_tensors(A::MPO, fusion_tree_layer::Vector{ITensor}, j::Int)
  return A[j] *
         A[j + 1] *
         dag(fusion_tree_layer[(j + 1) ÷ 2]) *
         fusion_tree_layer[(j + 1) ÷ 2]'
end

function fuse_inds_binary_layer(A::Union{MPS,MPO}, fusion_tree_layer::Vector{ITensor})
  n = length(fusion_tree_layer)
  A_fused = ITensor[]
  for j in 1:2:(2n)
    push!(A_fused, fuse_tensors(A, fusion_tree_layer, j))
  end
  if isodd(length(A))
    push!(A_fused, A[end])
  end
  return typeof(A)(A_fused)
end

function fuse_inds_binary(A::Union{MPS,MPO}, fusion_tree::Vector{Vector{ITensor}})
  depth = length(fusion_tree)
  A_fused = A
  for layer in 1:depth
    A_fused = fuse_inds_binary_layer(A_fused, fusion_tree[layer])
  end
  return only(A_fused)
end
