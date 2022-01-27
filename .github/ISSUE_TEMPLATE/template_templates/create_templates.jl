using FileUtils

template_package_name = "PACKAGE"

package_names = [
  "ITensors",
  "ITensorGaussianMPS",
  "ITensorGLMakie",
  "ITensorGPU",
  "ITensorMakie",
  "ITensorUnicodePlots",
  "ITensorVisualizationBase",
  "NDTensors",
]

bug_report_file(package_name::String) = "$(package_name)_bug_report.md"
feature_request_file(package_name::String) = "$(package_name)_feature_request.md"

for package_name in package_names
  @show package_name

  template_bug_report = bug_report_file(template_package_name)
  new_bug_report = bug_report_file(package_name)

  if isfile(new_bug_report)
    println("File $new_bug_report already exists, skipping")
  else
    println("Copying $template_bug_report to $new_bug_report")
    cp(template_bug_report, new_bug_report)

    println("Replace $template_package_name with $package_name in $new_bug_report")
    replace_in_file(new_bug_report, template_package_name => package_name)
  end

  template_feature_request = feature_request_file(template_package_name)
  new_feature_request = feature_request_file(package_name)

  if isfile(new_feature_request)
    println("File $new_feature_request already exists, skipping")
  else
    println("Copying $template_feature_request to $new_feature_request")
    cp(template_feature_request, new_feature_request)

    println("Replace $template_package_name with $package_name in $new_feature_request")
    replace_in_file(new_feature_request, template_package_name => package_name)
  end
end
