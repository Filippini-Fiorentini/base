#include "generator_temp.h"

int main(int argc, char const *argv[])
	{
	FG::Generator_temp gen("proviamoci","../",true,true,true);
	gen.add_subs_h("_FILE_NAME_H","CIOAIAOIAO");
	gen.add_subs_h("Fiorentini","Cafu");
	gen.print_info();
	std::cout << "Exiting with result: " << gen.generate() << std::endl;
	return 0;
	}