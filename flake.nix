{
  description = "DOE Toolkit - Design of Experiments software (Streamlit app)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonDeps = ps: with ps; [
          numpy
          scipy
          pandas
          matplotlib
          plotly
          patsy
          statsmodels
          scikit-learn
          streamlit
          requests
        ];

        python = pkgs.python3.withPackages pythonDeps;
        pythonTest = pkgs.python3.withPackages (ps: pythonDeps ps ++ [ ps.pytest ]);

        src = pkgs.lib.cleanSourceWith {
          src = ./.;
          filter = path: type:
            let base = baseNameOf path; in
            !(pkgs.lib.hasSuffix ".nix" base || base == "flake.lock" || base == "result");
        };

        doe-toolkit = pkgs.stdenvNoCC.mkDerivation {
          pname = "doe-toolkit";
          version = builtins.head (builtins.match ''.*__version__ = "([^"]+)".*''
            (builtins.readFile ./src/__version__.py));
          inherit src;

          nativeBuildInputs = [ pkgs.makeWrapper ];

          dontBuild = true;

          installPhase = ''
            runHook preInstall

            share=$out/share/doe-toolkit
            mkdir -p $share $out/bin
            cp -r src docs .streamlit $share/

            # .streamlit/config.toml is only read from the working directory,
            # so its settings are passed as flags instead.
            makeWrapper ${python}/bin/python $out/bin/doe-toolkit \
              --prefix PYTHONPATH : $share \
              --add-flags "-m streamlit run $share/src/ui/app.py" \
              --add-flags "--server.headless false" \
              --add-flags "--browser.gatherUsageStats false" \
              --add-flags "--server.enableXsrfProtection true" \
              --add-flags "--theme.base light" \
              --add-flags "--theme.primaryColor '#4A90E2'" \
              --add-flags "--theme.secondaryBackgroundColor '#F0F2F6'"

            runHook postInstall
          '';

          meta = with pkgs.lib; {
            description = "Design of Experiments toolkit with a Streamlit web UI";
            homepage = "https://github.com/probar/doe-toolkit";
            license = licenses.mit;
            mainProgram = "doe-toolkit";
            platforms = platforms.unix;
          };
        };
      in
      {
        packages.default = doe-toolkit;
        packages.doe-toolkit = doe-toolkit;

        apps.default = {
          type = "app";
          program = "${doe-toolkit}/bin/doe-toolkit";
        };

        # Smoke test: every module under src/ imports with the packaged deps.
        # (Run the upstream pytest suite with `nix develop -c pytest tests`.)
        checks.imports = pkgs.runCommand "doe-toolkit-imports" { } ''
          export HOME=$TMPDIR MPLBACKEND=Agg
          cd ${doe-toolkit}/share/doe-toolkit
          ${python}/bin/python -B - <<'EOF'
        import importlib, pathlib
        mods = sorted(".".join(p.with_suffix("").parts) for p in pathlib.Path("src").rglob("*.py")
                      if p.name != "__init__.py" and "ui" not in p.parts)
        for m in mods:
            importlib.import_module(m)
        print(f"imported {len(mods)} modules")
        EOF
          touch $out
        '';

        devShells.default = pkgs.mkShell {
          packages = [ pythonTest ];
        };
      });
}
