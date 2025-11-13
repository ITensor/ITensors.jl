for type in (:Algorithm, :Backend)
    @eval begin
        """
            $($type)

        A type representing a backend for a function.

        For example, a function might have multiple backends
        implementations, which internally are selected with a `$($type)` type.

        This allows users to extend functionality with a new implementation but
        use the same interface.
        """
        struct $type{Back, Kwargs <: NamedTuple} <: AbstractBackend
            kwargs::Kwargs
        end

        $type{Back}(kwargs::NamedTuple) where {Back} = $type{Back, typeof(kwargs)}(kwargs)
        $type{Back}(; kwargs...) where {Back} = $type{Back}(NamedTuple(kwargs))
        $type(s; kwargs...) = $type{Symbol(s)}(NamedTuple(kwargs))

        $type(backend::$type) = backend

        # TODO: Use `SetParameters`.
        backend_string(::$type{Back}) where {Back} = string(Back)
        parameters(backend::$type) = getfield(backend, :kwargs)

        function Base.show(io::IO, backend::$type)
            return print(io, "$($type) type ", backend_string(backend), ", ", parameters(backend))
        end
        Base.print(io::IO, backend::$type) = print(
            io, backend_string(backend), ", ", parameters(backend)
        )
    end
end

# TODO: See if these can be moved inside of `@eval`.
"""
    @Algorithm_str

A convenience macro for writing [`Algorithm`](@ref) types, typically used when
adding methods to a function that supports multiple algorithm
backends.
"""
macro Algorithm_str(s)
    return :(Algorithm{$(Expr(:quote, Symbol(s)))})
end

"""
    @Backend_str

A convenience macro for writing [`Backend`](@ref) types, typically used when
adding methods to a function that supports multiple
backends.
"""
macro Backend_str(s)
    return :(Backend{$(Expr(:quote, Symbol(s)))})
end
