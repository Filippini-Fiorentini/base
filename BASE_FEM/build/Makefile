CXXFLAGS += -Wall -Wextra -std=c++11 -O3
CPPFLAGS += -I${mkEigenInc} -DNDEBUG -I ../include
LDFLAGS += 
M = main
OBJ = main.o mesh_2d.o node_2d.o triangle.o interface_fem_2d.o

OUTPUT_PATH = ../output/
INPUT_FILE = ../input_files/nput_fem_2d.txt
LOAD_SCRIPT = load_script
SHELL:=/bin/bash
.PHONY: all build run clean distclean

.DEFAULT_GOAL = all

all: build

build: $(M)
	mkdir $(OUTPUT_PATH)

run: build $(M)
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
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -c $^

#CLEANING

clean:
	$(RM) $(OBJ)
	$(RM) *.gch

distclean: clean
	$(RM) $(M)
	rm -r ../output
	$(RM) *~
