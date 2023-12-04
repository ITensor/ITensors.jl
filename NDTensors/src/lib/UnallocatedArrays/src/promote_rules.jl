## TODO this section is not finished. I just pasted this from the previous PR.
## Still working here
function Base.promote_rule(z1::Type{<:UnallocatedZeros}, z2::Type{<:UnallocatedZeros})
  ElT = promote_type(eltype(z1), eltype(z2))
  @assert ndims(z1) == ndims(z2)
  Axs = axes(z1)
  Alloc = promote_type(alloctype(z1), alloctype(z2))
  return UnallocatedZeros{ElT,ndims(z1),Axs,Alloc}
end

function Base.promote_rule(z1::Type{<:UnallocatedZeros}, z2::Type{<:AbstractArray})
  ElT = promote_type(eltype(z1), eltype(z2))
  @assert ndims(z1) == ndims(z2)
  Axs = axes(z1)
  Alloc = promote_type(alloctype(z1), z2)
  set_eltype(Alloc, ElT)
  return UnallocatedZeros{ElT,ndims(z1),Axs,Alloc}
end

function Base.promote_rule(z1::Type{<:AbstractArray}, z2::Type{<:UnallocatedZeros})
  return promote_rule(z2, z1)
end
