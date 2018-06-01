////////////////////////////////////////////////////////////////////////////////
//! \file
//!
//! \author	Lorenzo Fiorentini
//!
//! \date	23/12/2018
//!
//! \par	Generator_temp:
//!	Class for generating files from  template files .
//!
////////////////////////////////////////////////////////////////////////////////

#if !defined	_GENERATOR_TEMP_H
#define	_GENERATOR_TEMP_H

#include "abstract_generator.h"
#include <unordered_map>

namespace FG
	{	
	class Generator_temp : public Abstract_generator
		{
		protected:
			//! \property	DEF_CLASS_H
			//! Default header file used for classes
			static constexpr char DEF_CLASS_H[] = "../base_cpp/class.h";
			
			//! \property	DEF_CLASS_CPP
			//! Default header file used for classes
			static constexpr char DEF_CLASS_CPP[] = "../base_cpp/class.cpp";
			
			//! \property	DEF_FCT_H
			//! Default header file used for classes
			static constexpr char DEF_FCT_H[] = "../base_cpp/base.h";
			
			//! \property	DEF_FCT_CPP
			//! Default header file used for classes
			static constexpr char DEF_FCT_CPP[] = "../base_cpp/base.cpp";
		
			//! \property	base_header
    		//! name of the base file used for the header file
			std::string base_header = DEF_CLASS_H;
			//! \property	base_src
    		//! name of the base file used for the source file
			std::string base_src = DEF_CLASS_CPP;

			//! \property	subs_map_h
    		//! Map of the strings that have to be substituted in the header file
			std::unordered_map<std::string, std::string> subs_map_h;
			//! \property	subs_map_src
    		//! Map of the strings that have to be substituted in the source file
			std::unordered_map<std::string, std::string> subs_map_src;

			int substitute(const std::string& file, const std::unordered_map<std::string, std::string>& map) const;
			
		public:
			//!  \function
			//!  Constructor with only the filename changed
			Generator_temp(const std::string& name,	const std::string& n_pth, bool fct, bool cpp, bool mv);
			//!  \function
			//!  Constructor with only the filename changed
			Generator_temp(const std::string& name): Abstract_generator(name) {}
			//!  \function
			//!  Default constructor
			Generator_temp(void) = default;
			//!  \function
			//!  Copy constructor
			Generator_temp(const Generator_temp& rhs) = default;
			//!  \function
			//!  Move constructor
			Generator_temp(Generator_temp&& rhs) = default;
			//!  \function
			//!  Destructor
			~Generator_temp(void) = default;
			//!  \function
			//!  Copy assignement
			Generator_temp& operator =(const Generator_temp& rhs) = default;
			//!  \function
			//!  Move assignement
			Generator_temp& operator =(Generator_temp&& rhs) = default;

			//!  \function
			//!  Add to set base header file
			void set_base_h(const std::string& arg);
			
			//!  \function
			//!  Add to set base header file
			void set_base_cpp(const std::string& arg);
			
			//!  \function
			//!  Add a substitution in header file
			void add_subs_h(const std::string& key, const std::string& value);
			
			//!  \function
			//!  Add a substitution in source file
			void add_subs_cpp(const std::string& key, const std::string& value);
			
			//!  \function
			//!  Add a substitution in both files
			void add_subs_both(const std::string& key, const std::string& value);
			
			//!  \function
			//!  Generate file and return 0 if everythinf went fine, an integer different from 0 otherwise
			int generate(void) const override;
			
			//!  \function
			//!  Print basic information about the file that we want to generate
			void print_info(void) const override;
		
			};
	}
#endif
