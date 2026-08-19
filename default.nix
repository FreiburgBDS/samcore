{ pkgs ? import (
    if (builtins.tryEval <nixpkgs>).success
    then <nixpkgs>
    else fetchTarball "https://github.com/NixOS/nixpkgs/archive/nixos-unstable.tar.gz"
  ) { }
}:

let
  version = pkgs.lib.fileContents ./version.txt;
in
pkgs.stdenv.mkDerivation {
  pname = "libsamcore";
  inherit version;
  src = pkgs.lib.cleanSource ./.;

  nativeBuildInputs = with pkgs; [
    cmake
    ninja
    pkg-config
  ];

  buildInputs = with pkgs; [
    hdf5
    nlohmann_json
    gtest
  ];

  cmakeFlags = [
    "-DSAMCORE_BUILD_TESTS=ON"
  ];

  doCheck = true;
}