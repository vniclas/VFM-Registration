.PHONY: cpp

install:
	@pip3 install --verbose ./python/

uninstall:
	@pip3 -v uninstall kiss_icp

editable:
	@pip3 install scikit-build-core pyproject_metadata pathspec pybind11 ninja cmake
	@pip3 install --no-build-isolation -ve ./python/

test:
	@pytest -rA --verbose ./python/

cpp:
	@cmake -Bbuild cpp/kiss_icp/
	@cmake --build build -j$(nproc --all)
