# NDTensors.copyto!
function copyto!(R::AbstractArray, T::AbstractArray)
  copyto!(leaf_parenttype(R), R, leaf_parenttype(T), T)
  return R
end

# NDTensors.copyto!
function copyto!(
  ::Type{<:AbstractArray}, R::AbstractArray, ::Type{<:AbstractArray}, T::AbstractArray
)
  Base.copyto!(R, T)
  return R
end
