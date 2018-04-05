#include <cmath>	
#include "fdm_2d_solver.hpp"
#include "interface_fdm_2d.hpp"

struct f1
	{
	fdm::Real operator()(fdm::Real x, fdm::Real y)
		{return 8*M_PI*M_PI*sin(2*M_PI*x)*sin(2*M_PI*y);}
	};

struct exact_1
	{
	fdm::Real operator()(fdm::Real x, fdm::Real y)
			{return sin(2*M_PI*x)*sin(2*M_PI*y);}		
	};

int main (int argc, char* argv [])
	{
	// MESH TEST
	std::vector<std::string> files = fdm::input_list_file("non_esisto.txt", argc, argv);

    fdm::Real Lx = fdm::read_value(files[fdm::IN], "Lx", 10.0);
    fdm::Real Ly = fdm::read_value(files[fdm::IN], "Ly", 10.0);
    
    fdm::Mesh_2d mesh(Lx,Ly);
        
    std::ofstream dbg(files[fdm::DBG]);
        
    mesh.print(dbg);
    
    // SOLVER TEST    
	std::vector<std::function<fdm::Real(fdm::Real, fdm::Real)>> vect_functions;
	vect_functions.emplace_back(f1());
	vect_functions.emplace_back(exact_1());
	vect_functions.emplace_back(fdm::function_2d_0());
	vect_functions.emplace_back(fdm::function_2d_1());
	
	fdm::Fdm_2d_solver solver_1(mesh, vect_functions[0], vect_functions[2], 
	            vect_functions[1]);
	Eigen::Matrix<fdm::Real, Eigen::Dynamic, 1> u = solver_1.solve();
	//std::cout << "result:" << std::endl << u << std::endl;
	std::cout << "result calculated" << std::endl;
	
	
	Eigen::Matrix<fdm::Real, Eigen::Dynamic, 1> exact_solution = solver_1.exact_sol();
	Eigen::Matrix<fdm::Real, Eigen::Dynamic, 1> diff = u-exact_solution;
	
	fdm::Real norml2 = diff.norm();
	std::cout << "l2 norm of the difference:" << norml2 << std::endl;	
	
	write_solution_file(mesh, u, files[fdm::SOL]);
	write_solution_file(mesh, exact_solution, files[fdm::EXACT]);
	}
