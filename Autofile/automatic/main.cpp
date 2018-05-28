#include <iostream>

#include "GetPot.hpp"

#include "create_file.hpp"

void print_help (void);
bool string_to_bool (const std::string &s);

int main (int argc, char **argv)
    {
    GetPot GP(argc,argv);
    
    if (GP.search(1,"-h"))
        {
        print_help();
        return 0;
        }
        
    std::string basename = GP("name","");
    std::string header = GP("header","y");
    std::string source = GP("source","y");
    std::string isclass = GP("class","y");
    std::string foldername = GP("folder",".");
    std::string data = GP("isdata","n");
    std::string make = GP("make","n");
    std::string makeonly = GP("makeonly","n");
    
    using Create_file = automatic::Create_file;
    
    bool ish = string_to_bool(header);
    bool iss = string_to_bool(source);
    bool isd = string_to_bool(data);
    bool mo = string_to_bool(makeonly);
             
    if ((ish || iss || isd) && !mo)
        {
        if (basename.empty())
            std::cerr << "ERROR: the name of the file must be provided" << std::endl;    
        else
            {
            Create_file CF (basename,foldername);
            if (header == "y")
                CF.header(string_to_bool(isclass));    
            if (source == "y")
                CF.source();  
            if (data == "y")
                CF.txt();      
            }
        }   
    
    if (make != "n")
        {
        bool inbuild = make == "inbuild" ? true : false;
        Create_file::makefile(foldername,inbuild);    
        }
               
    return 0;         
    }
    

void print_help (void)
    {
    std::cout << "Required parameters:\n";
    std::cout << "###\n";
    std::cout << "name=\n";
    std::cout << "This will be the name of the file; the relative header guard\n" 
              << "is constructed automatically\n";
    std::cout << "###\n";
    std::cout << "header=\n";
    std::cout << "Possible values: y/n; Default: y\n";
    std::cout << "###\n";
    std::cout << "source=\n";
    std::cout << "Possible values: y/n; Default: y\n";
    std::cout << "###\n";
    std::cout << "class=\n";
    std::cout << "Possible values: y/n; Default: y\n";
    std::cout << "###\n";
    /*
    std::cout << "folder=\n";
    std::cout << "This is the name of the folder in which you want to build the files\n";
    std::cout << "Default is the current folder\n";
    std::cout << "NOTE: you have to provide the full path (from ~)";
    */
    std::cout << "###\n";
    std::cout << "isdata=\n";
    std::cout << "Possible values: y/n; Default: n\n";
    std::cout << "###\n";
    std::cout << "make=\n";
    std::cout << "Possible values: inbuild/here/n; Default: n\n";
    std::cout << "Choose inbuild if there exist an include and a src folder\n";
    std::cout << "###\n";
    std::cout << "makeonly=\n";
    std::cout << "Possible values: y/n; Default: n\n";
    } 
    
bool string_to_bool (const std::string &s)
    {
    if (s == "y")
        return true;
    return false;    
    }          
