
struct LastVal <: Number
  n::Int
end

# TODO: make these definition work for notation
# A[1, end-1]
#(l::LastVal + n::Number) = LastVal(l.n + n)
#(n::Number + l::LastVal) = l + n
#(l::LastVal * n::Number) = LastVal(l.n * n)
#(n::Number * l::LastVal) = l * n
#(l::LastVal - n::Number) = LastVal(l.n - n)
#(n::Number - l::LastVal) = LastVal(n - l.n)
#(-l::LastVal) = LastVal(-l.n)

# Implement when ITensors can be indexed by a single integer
#lastindex(A::ITensor) = dim(A)

lastval_to_int(n::Int, ::LastVal) = n
lastval_to_int(::Int, n::Int) = n
lastval_to_int(dimsT::Tuple, I::Tuple) = lastval_to_int.(dimsT, I)
lastval_to_int(A::Tensor, I::Tuple) = lastval_to_int(size(A), I)
