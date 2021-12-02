name: TestITensorGaussianMPS
on:
  push:
    branches:
      - main
    tags: '*'
  pull_request:
jobs:
  test:
    name: Julia ${{ matrix.version }} - ${{ matrix.os }} - ${{ matrix.arch }} - ${{ matrix.threads }} thread(s)
    runs-on: ${{ matrix.os }}
    env:
      JULIA_NUM_THREADS: ${{ matrix.threads }}
    strategy:
      matrix:
        version:
          - '1.3'
          - '1'
        os:
          - ubuntu-latest
        threads:
          - '1'
        arch:
          - x64
    steps:
      - uses: actions/checkout@v2
      - uses: julia-actions/setup-julia@latest
        with:
          version: ${{ matrix.version }}
          arch: ${{ matrix.arch }}
      - name: Install Julia dependencies
        shell: julia --project=monorepo {0}
        run: |
          using Pkg;
          pkg"dev ./NDTensors . ./ITensorGaussianMPS"
      - name: Run the tests
        run: >
          julia --project=monorepo --depwarn=yes -e 'using Pkg; Pkg.test("ITensorGaussianMPS")'
