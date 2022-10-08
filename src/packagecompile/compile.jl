
default_compile_dir() = joinpath(homedir(), ".julia", "sysimages")

default_compile_filename() = "sys_itensors.so"

default_compile_path() = joinpath(default_compile_dir(), default_compile_filename())

function compile_note(; dir=default_compile_dir(), filename=default_compile_filename())
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

function compile(;
  dir::AbstractString=default_compile_dir(),
  filename::AbstractString=default_compile_filename(),
)
  if !isdir(dir)
    println("""The directory "$dir" doesn't exist yet, creating it now.""")
    println()
    mkdir(dir)
  end
  path = joinpath(dir, filename)
  println(
    """Creating the system image "$path" containing the compiled version of ITensors. This may take a few minutes.""",
  )
  create_sysimage(
    :ITensors;
    sysimage_path=path,
    precompile_execution_file=joinpath(@__DIR__, "precompile_itensors.jl"),
  )
  println(compile_note(; dir=dir, filename=filename))
  return path
end

@doc """
    ITensors.compile(; dir = "$(default_compile_dir())",
                       filename = "$(default_compile_filename())")

Compile ITensors.jl with [PackageCompiler](https://julialang.github.io/PackageCompiler.jl/dev/).
This will take some time, perhaps a few minutes.

This will create a system image containing the compiled version of ITensors
located at `dir/filename`, by default `$(default_compile_path())`.

$(compile_note())
""" compile
