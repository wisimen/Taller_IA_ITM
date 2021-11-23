# Taller_IA_ITM


docker rm -fv fruits_db_ctnr
Remove-Item .\schemas\* -Recurse -Force

docker-compose -f stack.yml up --force-recreate 

conda env remove -n fruits_env
conda env create -f .\conda_dependencies.yml
conda activate fruits_env
conda deactivate

ssh-keygen -t ed25519 -C "dahian01@gmail.com"

python fill_db.py

D:
\Universidad
\IA
\Taller_IA_ITM

fruits

Test

Apple Braeburn

3_100.jpg