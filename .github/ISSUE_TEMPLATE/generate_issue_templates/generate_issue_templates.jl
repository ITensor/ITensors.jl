using FileUtils

template_package_name = "PACKAGE"

package_names = [
  "ITensors",
  "NDTensors",
  "ITensorGPU",
  "ITensorGaussianMPS",
  "ITensorVisualizationBase",
  "ITensorUnicodePlots",
  "ITensorMakie",
  "ITensorGLMakie",
]

package_ordering = Dict([
  "ITensors" => 1,
  "NDTensors" => 2,
  "ITensorGPU" => 3,
  "ITensorGaussianMPS" => 4,
  "ITensorVisualizationBase" => 5,
  "ITensorUnicodePlots" => 6,
  "ITensorMakie" => 7,
  "ITensorGLMakie" => 8,
])

function bug_report_file(package_name::String)
  return "$(package_name)_bug_report.md"
end
function feature_request_file(package_name::String)
  return "$(package_name)_feature_request.md"
end

for package_name in package_names
  @show package_name

  order = lpad(package_ordering[package_name], 2, "0")

  template_bug_report = bug_report_file(template_package_name)
  new_bug_report = order * "_" * bug_report_file(package_name)

  if isfile(new_bug_report)
    println("File $new_bug_report already exists, skipping")
  else
    println("Copying $template_bug_report to $new_bug_report")
    cp(template_bug_report, new_bug_report)

    println("Replace $template_package_name with $package_name in $new_bug_report")
    replace_in_file(new_bug_report, template_package_name => package_name)

    mv(new_bug_report, joinpath("..", new_bug_report); force=true)
  end

  template_feature_request = feature_request_file(template_package_name)
  new_feature_request = order * "_" * feature_request_file(package_name)

  if isfile(new_feature_request)
    println("File $new_feature_request already exists, skipping")
  else
    println("Copying $template_feature_request to $new_feature_request")
    cp(template_feature_request, new_feature_request)

    println("Replace $template_package_name with $package_name in $new_feature_request")
    replace_in_file(new_feature_request, template_package_name => package_name)

    mv(new_feature_request, joinpath("..", new_feature_request); force=true)
  end
end
