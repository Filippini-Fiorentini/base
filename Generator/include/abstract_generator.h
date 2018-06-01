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

#if !defined	_ABSTRACT_FILE_GENERATOR_H
#define	_ABSTRACT_FILE_GENERATOR_H

#include <cstdio>		// rename
#include <string>		// string
#include <cstdlib>  	// system
#include <iostream> 	// cin cout
#include <fstream> 		// fstream in general
#include <cassert>		// to check sthg

namespace FG
	{
	class Abstract_generator
		{
		protected:
			//! \property	filename
    		//! name of the file that we want to write
    		std::string filename = "test";
    		//! \property	new_path
    		//! new path to the file that we want to write
    		std::string new_path = "./";
			//! \property	need_fct
			//! If true we will create a function if false a class
			bool need_fct = false;		
			//! \property	need_cpp
			//! If true we will create a cpp file too
			bool need_cpp = false; 		
			//! \property	need_to_move
			//! If true we will move the file(s) to a directory
			bool need_to_move = false; 	

			//! \property	quit_str
			//! Default string used to quit in menus
			constexpr static char quit_str[] =  "quit";
			
			//! \function
			//! Checks if a file exists and if we can open it
			bool file_exists(const std::string& fileName) const;
	
			//! \function
			//! copy file fileNameFrom in fileNameTo
			int copy_file(const std::string& fileNameFrom, const std::string& fileNameTo) const;

			//! \function
			//! move file from old_name to new_name (wrapper of rename)
			int move_file(const std::string& old_name, const std::string& new_name) const;

		public:
			//! \function
			//! constructor with only the name
			Abstract_generator(const std::string& name):
				filename(name), need_fct(false), need_cpp(false), need_to_move(false) {}
			//! \function
			//! constructor with all parameters
			Abstract_generator(const std::string& name,	const std::string& n_pth, bool fct, bool cpp, bool mv):
				filename(name), new_path(n_pth), need_fct(fct), need_cpp(cpp), need_to_move(mv) {}
			//!  \function
			//!  Default constructor
			Abstract_generator(void) = default;
			//!  \function
			//!  Copy constructor
			Abstract_generator(const Abstract_generator& rhs) = default;
			//!  \function
			//!  Move constructor
			Abstract_generator(Abstract_generator&& rhs) = default;
			//!  \function
			//!  Destructor
			~Abstract_generator(void) = default;
			//!  \function
			//!  Copy assignement
			Abstract_generator& operator =(const Abstract_generator& rhs) = default;
			//!  \function
			//!  Move assignement
			Abstract_generator& operator =(Abstract_generator&& rhs) = default;
			
			//!  \function
			//!  Generate file and return 0 if everythinf went fine, an integer different from 0 otherwise
			virtual int generate(void) const = 0;
			//!  \function
			//!  Print basic information about the file that we want to generate
			void set_path(const std::string& n_pth);
			//!  \function
			//!  Print basic information about the file that we want to generate
			virtual void print_info(void) const;
		};
	}
#endif
