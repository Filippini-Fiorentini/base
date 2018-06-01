////////////////////////////////////////////////////////////////////////////////
//! \file
//!
//! \author	Lorenzo Fiorentini
//!
//! \date	23/12/2018
//!
//! \par	Abstract_generator:
//!	Class for .
//!
////////////////////////////////////////////////////////////////////////////////

#include "abstract_generator.h"

namespace FG
	{
	bool Abstract_generator::file_exists(const std::string& fileName) const
		{
	    std::fstream file;
	    file.open(fileName.c_str(), std::ios::in);
	    if (file.is_open() == true)
	    	{
	    	file.close();
	    	return true;
	    	}
	    file.close();
	    return false;
		}

	int Abstract_generator::copy_file(const std::string& fileNameFrom, const std::string& fileNameTo) const
		{
		assert(file_exists(fileNameFrom));
		std::ifstream in (fileNameFrom.c_str());
	    std::ofstream out (fileNameTo.c_str());
		out << in.rdbuf();
		out.close();
		in.close();
		return 0;
		}

	int Abstract_generator::move_file(const std::string& old_name, const std::string& new_name) const
		{
		return rename ( old_name.c_str(), new_name.c_str() );
		}

	void Abstract_generator::set_path(const std::string& n_pth)
		{
		need_to_move = true;
		new_path  = n_pth;
		}

	void Abstract_generator::print_info(void) const
		{
		std::cout << "File: " << filename << " it is a " << (need_fct ? "function " : "class ");
		std::cout << "the cpp file " << (need_cpp ? "will be created too " : "won't be created ");
		std::cout << "the file(s) will be created in " << new_path << std::endl;
		}
	}