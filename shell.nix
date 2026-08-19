{ pkgs ? import (
    if (builtins.tryEval <nixpkgs>).success
    then <nixpkgs>
    else fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-unstable.tar.gz"
  ) { }
}:

let
  libsamcore = import ./default.nix { inherit pkgs; };
in
pkgs.mkShell {
  inputsFrom = [ libsamcore ];

  packages = with pkgs; [
    gdb
    (python3.withPackages (ps: with ps; [
      numpy
      scipy
      h5py
      pytest
      pip
      nanobind
      scikit-build-core
    ]))
  ];

  shellHook = ''
    export SAMCORE_TEST_DATA_DIR="$PWD/tests/data"
    echo "libsamcore dev shell (v${libsamcore.version})"
    echo "  cmake -B build -G Ninja && cmake --build build && ctest --test-dir build"
    echo "  pip install --no-build-isolation -e . && pytest tests/python"
  '';
}