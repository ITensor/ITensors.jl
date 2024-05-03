using NDTensors: @Algorithm_str
using ITensors: ITensors
using PackageCompiler: PackageCompiler

function ITensors.compile(
  ::Algorithm"PackageCompiler";
  dir::AbstractString=ITensors.default_compile_dir(),
  filename::AbstractString=ITensors.default_compile_filename(),
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
  PackageCompiler.create_sysimage(
    :ITensors;
    sysimage_path=path,
    precompile_execution_file=joinpath(@__DIR__, "precompile_itensors.jl"),
  )
  println(ITensors.compile_note(; dir, filename))
  return path
end
