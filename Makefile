all: format doc

format:
	black -l 79 timepp.py setup.py

doc:
	${MAKE} -C docs html
