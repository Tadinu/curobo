# NOTE: https://github.com/NVlabs/curobo/discussions/555
# - Assume correct Cuda toolkit (eg: 12.1) which correspond to your installed torch is installed at /usr/local/cuda-12.1
#   - https://developer.nvidia.com/cuda-12-1-0-download-archive
# - Make sure these are added in ~/.bashrc:
# export export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
# export CUDA_PATH=/usr/local/cuda-12.1
# Optional: export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
# - Verify by: `which nvcc && nvcc --version`

# Make sure a virtual env named [isaac] has been already created
#[Optional] workon isaac
#$ISAACSIM_PYTHON -m pip install tomli wheel ninja
#$ISAACSIM_PYTHON -m pip uninstall torch torchvision torchaudio
#$ISAACSIM_PYTHON -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
#$ISAACSIM_PYTHON -m pip install typing-extensions --upgrade
$ISAACSIM_PYTHON -m pip install -e .[isaacsim] --no-build-isolation
# If GLIBCXX_<version> error: https://stackoverflow.com/questions/68205760/install-glibcxx-3-4-29-in-anaconda
