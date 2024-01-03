using Dictionaries: Dictionary
using ..SparseArrayInterface: Zero

default_zero() = Zero()
default_data(type::Type, ndims::Int) = Dictionary{default_keytype(ndims),type}()
default_keytype(ndims::Int) = CartesianIndex{ndims}
