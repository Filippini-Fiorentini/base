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

#include "generator_temp.h"

namespace FG
	{
	constexpr char Generator_temp::DEF_FCT_H[];
	constexpr char Generator_temp::DEF_FCT_CPP[];
	//Reason: You have to provide the definition of the static member as well as the declaration.
	//The declaration and the initializer go inside the class definition,
	//but the member definition has to be separate.
	
	int Generator_temp::substitute(const std::string& file, const std::unordered_map<std::string, std::string>& map) const
		{
		int res = 0;
 		for (auto iter = map.cbegin(); iter != map.cend(); ++iter)
	 		{
	 		std::string command = "sed -i -e 's/";
	 		command += 	iter->first + "/" + iter->second + "/g' " + file;
	 		std::cout << "command: " << command << std::endl;
	 		res = system(command.c_str());
	 		std::cout << "result of the command: " << res << std::endl;
	 		}
		return res;
		}

	Generator_temp::Generator_temp(const std::string& name,	const std::string& n_pth, bool fct, bool cpp, bool mv):
				 Abstract_generator(name, n_pth, fct, cpp, mv)
				 {
				 if(fct)
				 	{
				 	base_header = DEF_FCT_H;
				 	base_src = DEF_FCT_CPP;
				 	}
				 }

	void Generator_temp::set_base_h(const std::string& arg)
		{
		base_header = arg;
		}

	void Generator_temp::set_base_cpp(const std::string& arg)
		{
		base_src = arg;
		}

	void Generator_temp::add_subs_h(const std::string& key, const std::string& value)
		{
		subs_map_h.insert(std::make_pair(key, value));
		}
			
	void Generator_temp::add_subs_cpp(const std::string& key, const std::string& value)
		{
		if(!need_cpp)
			{
			need_cpp = true;
			std::cout << "The source file will now be created too" << std::endl;
			}
		subs_map_src.insert(std::make_pair(key, value));
		}
			
	void Generator_temp::add_subs_both(const std::string& key, const std::string& value)
		{
		add_subs_h(key,value);
		add_subs_cpp(key,value);
		}
			
	int Generator_temp::generate(void) const
		{
		const std::string out_file_h = need_to_move? new_path + filename + ".h": filename + ".h";
		const std::string out_file_cpp = need_to_move? new_path + filename + ".cpp": filename + ".cpp";

		int res = copy_file(base_header,out_file_h);
		if (res)
			return res;
		if(need_cpp)
			{
			res = copy_file(base_src,out_file_cpp);
			if (res)
				return res;
			}
		res = substitute(out_file_h,subs_map_h);
		if (res)
			return res;
		if(need_cpp)
			{
			res = substitute(out_file_cpp,subs_map_src);
			if (res)
				return res;
			}
		return res;
		}
			
	void Generator_temp::print_info(void) const
		{
		Abstract_generator::print_info();
		std::cout << "The file used as template for the header is " << base_header;
		if (need_cpp)
			std::cout << "The file used as template for the header is " << base_header;
		std::cout << std::endl;

		for (auto iter_pair = subs_map_h.cbegin(); iter_pair != subs_map_h.cend(); ++iter_pair)
			std::cout << iter_pair->first << " (in header) will be substituted with " << iter_pair->second << std::endl;
		if (need_cpp)
			for (auto iter_pair = subs_map_src.cbegin(); iter_pair != subs_map_src.cend(); ++iter_pair)
				std::cout << iter_pair->first << " (in source) will be substituted with " << iter_pair->second << std::endl;
		}
	}