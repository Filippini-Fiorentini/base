CXXFLAGS += -Wall -std=c++11
M = main
OBJ = main.o create_file.o

.PHONY: all clean distclean

.DEFAULT_GOAL: all

all: $(M)

main.o: GetPot.hpp create_file.hpp
create_file.o: create_file.hpp

$(M): $(OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	$(RM) $(OBJ)

distclean: clean
	$(RM) $(M)
	$(RM) *~
