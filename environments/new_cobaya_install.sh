#!/bin/bash
## $1 is the first argument to a bash script, which will be the name of the folder

## need to check that $1 has actually been supplied
if [ $# -eq 0 ]; then
  echo "please provide a name for the install folder"
  exit 2
fi

## make directory with name $1
mkdir $1
cd $1

## create python environment
module load python/3.8
python -m venv "env_${1}"
source "env_${1}"/bin/activate
python --version
python -m pip install --upgrade pip wheel setuptools
python -m pip --version

## install mpi, cobaya, pyside2 and pyqt5
python -m pip install "mpi4py>=3" --upgrade --no-binary :all:
python -m pip install git+https://github.com/williamjameshandley/cobaya@master ## here we want to install the specific cobaya
python -m pip install pyside2 pyqt5 ## need these to see cobaya-cosmo-generator

## install polychord before loading gcc
python -m pip install git+https://github.com/PolyChord/PolyChordLite@master
cobaya-install polychord --packages-path /home/ocn22/environments/cobaya_env/env/lib/python3.8/site-packages/

## load higher version of gcc
module load gcc

## install set of cosmology requisites (CAMB, CLASS, Planck, BICEP-Keck, BAO, SN) into packages folder within $1
cobaya-install cosmo -p packages

## ## pip install camb and pypolychord
## python -m pip install camb
## python -m pip install git+https://github.com/PolyChord/PolyChordLite@master

## copy basic camb polychord run
cp ../install_cobaya/camb_polychord.yaml camb_polychord.yaml

## make runs directory
mkdir runs

## copy submit script
cp ../install_cobaya/submit_cclake_template "submit_${1}"

sed -i "13s/.*/#SBATCH -J $1/" "submit_$1"

## as there are / in the paths, I have to use a different separator for sed (here |)
sed -i "61s|.*|source env_$1/bin/activate|" "submit_$1"

sed -i "65s|.*|application=\"/rds-d7/user/ano23/hpc-work/"$1"/"env_$1"/bin/python3\"|" "submit_$1"

sed -i '68s|.*|options="-m cobaya run -p packages -o runs/camb_poly -r camb_polychord.yaml"|' "submit_$1"
