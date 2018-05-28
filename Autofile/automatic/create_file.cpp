#include "create_file.hpp"

namespace automatic {

void 
Create_file::header (bool isclass) const
    {
    std::ostringstream hname;
    hname << foldername << "/include/" << basename << ".hpp";
    
    std::ofstream ofs (hname.str());
    
    std::string hg = create_header_guard();
    ofs << "#ifndef " << hg << '\n';
    ofs << "#define " << hg << '\n' << '\n';
    if (isclass)
        {
        std::string clname = create_class_name();
        ofs << "class " << clname << " {" << '\n' << '\n' << "};" << '\n' << '\n';
        }
    ofs << "#endif /* " << hg << " */";   
    }

void 
Create_file::source (void) const
    {
    std::ostringstream sname;
    sname << foldername << "/src/" << basename << ".cpp";
    
    std::ofstream ofs (sname.str());
    
    ofs << "#include \"" << basename << ".hpp\"";  
    }   
    
void 
Create_file::txt (void) const
    {
    std::ostringstream tname;
    tname << foldername << "/data/" << basename << ".txt";
    
    std::ofstream ofs (tname.str());
    
    ofs << "26104";  
    } 
    
void 
Create_file::makefile (const std::string &fold, bool inbuild)
    {
    std::ostringstream mname;
    if (inbuild)
        mname << fold << "/build/Makefile";
    else
        mname << fold << "/Makefile";    
        
    std::ofstream ofs (mname.str());
    
    ofs << "CXXFLAGS += -Wall -std=c++11\n";
    ofs << "EXE =\n";
    ofs << "OBJS =\n";
    
    if (inbuild)
        {
        ofs << "\nHPATH = ../include\n";
        ofs << "SPATH = ../src\n";
        }
    
    ofs << "\n.PHONY: all clean distclean\n";
    ofs << "\n.DEFAULT_GOAL: all\n";
    ofs << "\nall: $(EXE)\n";
    ofs << "\n$(EXE): $(OBJS)\n\t$(CXX) $(CXXFLAGS) $^ -o $@\n";
    ofs << "\nclean:\n\t$(RM) $(OBJS)\n";
    ofs << "\ndistclean: clean\n\t$(RM) $(EXE)\n\t$(RM) *~" << std::endl;  
    }        

std::string 
Create_file::create_header_guard (void) const
    {
    std::locale loc;
    std::string hg(basename);
    for (unsigned j=0; j<hg.size(); j++)
        hg[j] = std::toupper(hg[j],loc);
    std::ostringstream os;
    os << hg << "_HH";    
    return os.str();
    }

std::string 
Create_file::create_class_name (void) const
    {
    std::locale loc;
    std::string hg(basename);
    hg[0] = std::toupper(hg[0],loc);    
    return hg;
    }   

}
