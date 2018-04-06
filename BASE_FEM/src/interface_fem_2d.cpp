#include "interface_fem_2d.hpp"

namespace fem
{

void 
write_solution_file(const Mesh_2d& mesh, const Eigen::Matrix<fem::Real, 
            Eigen::Dynamic, 1>& vect, const std::string& filename)
	{
	std::size_t vect_size = mesh.get_n_nodes();
	if(vect_size != vect.rows())
		{
		std::cerr << "The Eigen vector and the mesh must have the same dimension" 
		          << std::endl;
		return;
		}
	std::ofstream output;
	output.open(filename);
	for (std::size_t i = 0; i < vect_size; ++i)
		{
		Node_2d node = mesh.get_node(i);
		output << node.get_x() << ',' << node.get_y() << ',' << vect(i,0) << std::endl;
		}
	output.close();
	}

Real
read_value(const std::string& filename, const std::string& name, Real default_value)
	{
	GetPot file( filename.c_str() );
	Real value = file(name.c_str(), default_value);
	if(value == default_value)
		std::cout << "Warning: " << name << " is equal to default_value = " 
		          << default_value << std::endl;
	return value;
	}
	
	
/*
std::function<fem::Real(fem::Real, fem::Real) 
read_2d_function(const std::string& filename, const std::string& name);
*/

std::vector<std::string>
input_list_file(const std::string& default_name, int argc, char* argv [])
	{
	GetPot command_line(argc, argv);
	std::string filename = command_line.follow(default_name.c_str(), 2, "-f","--file");
	GetPot file( filename.c_str() );
	//GetPot file( "input_fem_2d.txt" );
	std::vector<std::string> vect(N_FILES);
	if(filename == default_name)
		std::cerr << "Warning: " << filename << " not found, the input_list_file is created using instead "
			<< default_name << std::endl;
	else
		{
		vect[OUT_ERR] = file("OUT_ERR", DEF_OUT_ERR.c_str());
		vect[DBG] = file("DBG", DEF_DBG.c_str());
		vect[DBG] = file("MESH", DEF_MESH.c_str());
		vect[IN] = file("IN", DEF_IN.c_str());
		vect[SOL] = file("SOL", DEF_SOL.c_str());
		vect[EXACT] = file("EXACT", DEF_EXACT.c_str());
		}
	return vect;
	}

}
