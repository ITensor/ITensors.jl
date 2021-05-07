"""
    AliasStyle

A trait that determines the aliasing behavior of a constructor or function,
for example whether or not a function or constructor might return an alias
of one of the inputs (i.e. the output shares memory with one of the inputs,
such that modifying the output also modifies the input or vice versa).

See also [`AllowAlias`](@ref) and [`NeverAlias`](@ref).
"""
abstract type AliasStyle end

"""
    AllowAlias

Singleton type used in a constructor or function indicating
that the constructor or function may return an alias of the input data when
possible, i.e. the output may share data with the input. For a constructor
`T(AllowAlias(), args...)`, this would act like `Base.convert(T, args...)`.

See also [`AliasStyle`](@ref) and [`NeverAlias`](@ref).
"""
struct AllowAlias <: AliasStyle end

"""
    NeverAlias

Singleton type used in a constructor or function indicating
that the constructor or function will never return an alias of the input data,
i.e. the output will never share data with one of the inputs.

See also [`AliasStyle`](@ref) and [`AllowAlias`](@ref).
"""
struct NeverAlias <: AliasStyle end
