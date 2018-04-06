#include <cmath>	
#include "interface_fem_2d.hpp"

struct f1
	{
	fem::Real operator()(fem::Real x, fem::Real y)
		{return 8*M_PI*M_PI*sin(2*M_PI*x)*sin(2*M_PI*y);}
	};

struct exact_1
	{
	fem::Real operator()(fem::Real x, fem::Real y)
			{return sin(2*M_PI*x)*sin(2*M_PI*y);}		
	};

int main (int argc, char* argv [])
	{
	// MESH TEST
	std::vector<std::string> files = fem::input_list_file("non_esisto.txt", argc, argv);

    fem::Real Lx = fem::read_value(files[fem::IN], "Lx", 10.0);
    fem::Real Ly = fem::read_value(files[fem::IN], "Ly", 10.0);
    
    fem::Mesh_2d mesh(Lx,Ly);
        
    std::ofstream dbg(files[fem::DBG]);
        
    mesh.print(dbg);
    
    // SOLVER TEST    
	std::vector<std::function<fem::Real(fem::Real, fem::Real)>> vect_functions;
	vect_functions.emplace_back(f1());
	vect_functions.emplace_back(exact_1());
	vect_functions.emplace_back(fem::function_2d_0());
	vect_functions.emplace_back(fem::function_2d_1());
	
	fem::Fdm_2d_solver solver_1(mesh, vect_functions[0], vect_functions[2], 
	            vect_functions[1]);
	Eigen::Matrix<fem::Real, Eigen::Dynamic, 1> u = solver_1.solve();
	//std::cout << "result:" << std::endl << u << std::endl;
	std::cout << "result calculated" << std::endl;
	
	
	Eigen::Matrix<fem::Real, Eigen::Dynamic, 1> exact_solution = solver_1.exact_sol();
	Eigen::Matrix<fem::Real, Eigen::Dynamic, 1> diff = u-exact_solution;
	
	fem::Real norml2 = diff.norm();
	std::cout << "l2 norm of the difference:" << norml2 << std::endl;	
	
	write_solution_file(mesh, u, files[fem::SOL]);
	write_solution_file(mesh, exact_solution, files[fem::EXACT]);
	}
