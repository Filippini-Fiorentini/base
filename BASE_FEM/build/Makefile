CXXFLAGS += -Wall -Wextra -std=c++11 -O3
CPPFLAGS += -I${mkEigenInc} -DNDEBUG -I ../include
LDFLAGS += 
M = main
OBJ = main.o mesh_2d.o node_2d.o triangle.o interface_fem_2d.o quadrature.o reference.o

ERROR_BUILD_FILE = error_build.txt
OUTPUT_PATH = ../output/
INPUT_FILE = ../input_files/nput_fem_2d.txt
LOAD_SCRIPT = load_script
SHELL:=/bin/bash
.PHONY: all build run clean distclean

.DEFAULT_GOAL = all

all: build

#Should check before if output_path exists and if it is not the case it creates the directory
build: $(M)

run: build $(M)
	mkdir $(OUTPUT_PATH)
	./$(M) -f $(INPUT_FILE)

#LOADING MODULES
#load:
#	source $(LOAD_SCRIPT)
#	module load gcc-glibc
	
#LINKING

$(M): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) $^ $(OUTPUT_OPTION)

#COMPILATION

%.o: ../src/%.cpp   
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -c $^ >& $(ERROR_BUILD_FILE)

#CLEANING

clean:
	$(RM) $(OBJ)
	$(RM) $(ERROR_BUILD_FILE)
	$(RM) *.gch

distclean: clean
	$(RM) $(M)
	rm -r ../output
	$(RM) *~
