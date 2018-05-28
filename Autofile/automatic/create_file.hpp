#ifndef CREATE_FILE_H
#define CREATE_FILE_H

#include <fstream>
#include <sstream>
#include <string>

namespace automatic {

class Create_file {

private:
    
    std::string basename;
    std::string foldername;
    
    std::string create_header_guard (void) const;
    std::string create_class_name (void) const; 
    
public:

    Create_file (const std::string &b, const std::string &f): basename(b), foldername(f) {}

    void header (bool) const;
    void source (void) const;
    void txt (void) const;
    static void makefile (const std::string &, bool);

    };

}

#endif /* CREATE_FILE_H */
