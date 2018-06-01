CXX = g++
CPPFLAGS = -I ../include
CXXFLAGS = -std=c++11 -Wall 
DEBUG = no

ifeq ($(DEBUG),yes)
  OPTFLAGS=-g
else
  OPTFLAGS=-O3
  CPPFLAGS+=-DNDEBUG
endif

BIN_DIR = ../bin
SRC_DIR = ../src
OBJ_DIR = ../obj
OUT_DIR = ../output
OUTPUT_FILE = $(OUT_DIR)/output_txt
EXE = main
OBJECTS = main.o abstract_generator.o generator_temp.o
OBJECTS_LINK = $(OBJ_DIR)/main.o $(OBJ_DIR)/abstract_generator.o $(OBJ_DIR)/generator_temp.o
.PHONY: all clean distclean run 

.DEFAULT_GOAL: all

all: $(EXE)

run:
	../bin/$(EXE)

$(EXE): $(OBJECTS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPTFLAGS) $(OBJECTS_LINK) -o $(EXE)
	mv $(EXE) $(BIN_DIR)

%.o: $(SRC_DIR)/%.cpp 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OPTFLAGS) $^ -c
	mv $@ $(OBJ_DIR)

clean:
	$(RM) $(OBJ_DIR)/*.o

distclean: clean
	$(RM) $(BIN_DIR)/$(EXE)	
