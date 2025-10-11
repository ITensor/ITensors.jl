default_compile_dir() = joinpath(homedir(), ".julia", "sysimages")

default_compile_filename() = "sys_itensors.so"

default_compile_path() = joinpath(default_compile_dir(), default_compile_filename())

function compile_note(; dir = default_compile_dir(), filename = default_compile_filename())
    path = joinpath(dir, filename)
    return """
    You will be able to start Julia with a compiled version of ITensors using:

    ```
    ~ julia --sysimage $path
    ```

    and you should see that the startup times and JIT compilation times are substantially improved when you are using ITensors.

    In unix, you can create an alias with the Bash command:

    ```
    ~ alias julia_itensors="julia --sysimage $path -e 'using ITensors' -i"
    ```

    which you can put in your `~/.bashrc`, `~/.zshrc`, etc. This also executes
    `using ITensors` so that ITensors is loaded and ready to use, you can leave off `
    -e 'using ITensors' -i` if you don't want that. Then you can start Julia with a
    version of ITensors installed with the command:

    ```
    ~ julia_itensors
    ```

    Note that if you update ITensors to a new version, for example with `using
    Pkg; Pkg.update("ITensors")`, you will need to run the `ITensors.compile()`
    command again to recompile the new version of ITensors.
    """
end

function compile(; backend = Algorithm"PackageCompiler"(), kwargs...)
    return compile(backend; kwargs...)
end

@doc """
    ITensors.compile(; dir = "$(default_compile_dir())",
                       filename = "$(default_compile_filename())")

Compile ITensors.jl with [PackageCompiler.jl](https://julialang.github.io/PackageCompiler.jl/dev/).
This will take some time, perhaps a few minutes.

This will create a system image containing the compiled version of ITensors
located at `dir/filename`, by default `$(default_compile_path())`.

!!! compat "ITensors 0.7"
    As of ITensors 0.7, you must now install and load both the
    [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl) package
    and the [PackageCompiler.jl](https://github.com/JuliaLang/PackageCompiler.jl)
    package in order to use `ITensors.compile()`, since it relies on running MPS/MPO
    functionality as example code for Julia to compile and is based in a package
    extension in order to make `PackageCompiler.jl` an optional dependency.

$(compile_note())
""" compile
