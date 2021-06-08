CXX := g++
CXXFLAGS := -std=c++11 -O3
NVFLAGS := $(CXXFLAGS)
TARGET := raytracer


.PHONY: all
all: $(TARGET)

.PHONY: raytracer
raytracer: raytracer.cu
	nvcc $(NVFLAGS) -o raytracer raytracer.cu
.PHONY: clean
clean:
	rm -f $(TARGET) $(SEQUENTIAL)


