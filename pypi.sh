#rm dist/*
#python -m build
#twine upload dist/*

pip install --upgrade pip build twine
rm -rf dist/
python -m build --sdist --wheel
twine upload --username=__token__ dist/*
