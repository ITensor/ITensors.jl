isroc(A::AbstractArray) = isroc(typeof(A))

function isroc(A::Type{<:AbstractArray})
  return (unwrap_type(A) == A ? false : isroc(unwrap_type(A)))
end
