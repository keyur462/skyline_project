1. Clone nmslib from https://github.com/nmslib/nmslib
	*** If reinstalling makesure to remove python_bindings/build after uninstalling nmslib
	pip uninstall nmslib
	rm -r build/
	rm -r tmp/
	restart the python kernel
	pip install .
	restart the python kernel


2. Replace init_spaces.h at nmslib\similarity_search\include\factory with the provided.
3. Add space_weighted_jaccard.h to  nmslib\similarity_search\include\factory\space
4. movChange directory to nmslib\python_bindings folder
5. Run pip install .
6. Now you can specify "WeightedJaccard" as the space 
